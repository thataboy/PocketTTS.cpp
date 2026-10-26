#!/usr/bin/env python3
"""Exploratory static Conv quantization; never modifies the source bundle.

Run collect, quantize, optionally combined, then evaluate/benchmark.
Models/corpus/reports live in --output.
Quantization is calibrated on real, stateful speech trajectories. Evaluation
uses held-out utterances and the exact same latents for every decoder.
"""
import argparse
import collections
import json
import math
import shutil
import subprocess
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from scipy.io import wavfile


def session(path, threads=1, optimized=None):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    if optimized:
        options.optimized_model_filepath = str(optimized)
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


def initial(s):
    types = {'tensor(float)': np.float32, 'tensor(int64)': np.int64, 'tensor(bool)': np.bool_}
    return {i.name: np.zeros(i.shape, types[i.type]) for i in s.get_inputs() if i.name.startswith('state_')}


def advance(s, latent, state):
    outputs = s.run(None, {'latent': latent, **state})
    return outputs[0].reshape(-1), {i.name: v for i, v in zip(s.get_inputs()[1:], outputs[1:])}


def collect(args):
    import torch
    from scipy.signal import resample_poly
    sys.path.insert(0, str(args.source.resolve()))
    from pocket_tts import TTSModel
    torch.set_num_threads(4)
    model = TTSModel.load_model(language='english', temp=0.3).eval()
    model.max_decoder_frames_per_call = 1
    texts = [
        'The morning sun lit the quiet street as a narrator began telling the next part of the story.',
        'Please close the window gently, then put the blue notebook beside the lamp.',
        'In nineteen forty three, her father moved to New York and started a small business.',
        'Are you really coming tomorrow? I thought the meeting had been cancelled.',
        'She was born in New York in 1943. Her father was a Hungarian Jewish immigrant.',
        'What a wonderful surprise! I never expected to see you here again.',
    ]
    records = []
    original = model.mimi.decode_from_latent
    segments, identities = [], []
    def capture(latent, state):
        if not identities or identities[-1] is not state:
            identities.append(state)
            segments.append([])
        z = (latent.detach().cpu() - model.flow_lm.emb_mean.cpu()) / model.flow_lm.emb_std.cpu()
        segments[-1].append(z.numpy().copy())
        return original(latent, state)
    model.mimi.decode_from_latent = capture
    corpus = args.output / 'corpus'
    corpus.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for vi, voice in enumerate(args.voices):
            rate, audio = wavfile.read(voice)
            if np.issubdtype(audio.dtype, np.integer):
                audio = audio.astype(np.float32) / max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
            audio = audio.astype(np.float32)
            if audio.ndim == 2:
                audio = audio.mean(1)
            if rate != 24000:
                gcd = math.gcd(rate, 24000)
                audio = resample_poly(audio, 24000 // gcd, rate // gcd).astype(np.float32)
            prompt = model.get_state_for_audio_prompt(torch.from_numpy(audio[:720000])[None])
            for ti, text in enumerate(texts):
                segments.clear(); identities.clear()
                torch.manual_seed(100 + vi * 10 + ti)
                model.generate_audio(prompt, text)
                for si, blocks in enumerate(segments):
                    name = f'{voice.stem}-{ti}-{si}'
                    latent = np.concatenate(blocks, axis=1)
                    np.save(corpus / (name + '.npy'), latent)
                    records.append(dict(name=name, voice=str(voice.resolve()), text=text,
                                        split='calibration' if ti < 4 else 'evaluation', frames=latent.shape[1]))
                print(voice.stem, ti, sum(b.shape[1] for blocks in segments for b in blocks), flush=True)
    manifest = json.loads((args.bundle / 'manifest.json').read_text())
    (corpus / 'manifest.json').write_text(json.dumps(dict(upstream_revision=manifest['upstream_revision'], records=records), indent=2)+'\n')


def quantize(args):
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process
    source = args.bundle / 'mimi_decoder.onnx'
    prepared = args.output / 'decoder-prepared.onnx'
    if not prepared.exists():
        quant_pre_process(source, prepared, skip_optimization=True, skip_symbolic_shape=True)
    records = json.loads((args.output/'corpus/manifest.json').read_text())['records']
    class Reader(CalibrationDataReader):
        def __init__(self):
            self.reference = session(source)
            self.items = self.generate()
        def generate(self):
            for r in records:
                if r['split'] != 'calibration':
                    continue
                latent = np.load(args.output/'corpus'/(r['name']+'.npy'))
                state = initial(self.reference)
                starts = list(range(1, latent.shape[1], 4))
                selected = set(np.linspace(0, max(0,len(starts)-1), min(12,len(starts)), dtype=int))
                for step, start in enumerate([0]+starts):
                    end = 1 if start == 0 else min(start+4, latent.shape[1])
                    z = latent[:,start:end].copy()
                    feed = {'latent':z, **state}
                    _, state = advance(self.reference, z, state)
                    if step == 0 or step-1 in selected:
                        yield feed
        def get_next(self):
            return next(self.items, None)
    graph = onnx.load(prepared)
    convs = [n.name for n in graph.graph.node if n.op_type == 'Conv']
    phases = [n for n in convs if n.startswith('/decoder/') and '/cpu_phase/' in n]
    heavy = [n for n in convs if n in phases or n == '/decoder/model.0/conv/Conv']
    candidates = [
        ('conv-all-qdq-u8s8', QuantFormat.QDQ, QuantType.QUInt8, convs),
        ('conv-inner-qdq-u8s8', QuantFormat.QDQ, QuantType.QUInt8, convs[1:-1]),
        ('conv-all-qop-u8s8', QuantFormat.QOperator, QuantType.QUInt8, convs),
        ('conv-phase-qdq-u8s8', QuantFormat.QDQ, QuantType.QUInt8, phases),
        ('conv-heavy-qdq-u8s8', QuantFormat.QDQ, QuantType.QUInt8, heavy),
    ]
    for name, fmt, activation, nodes in candidates:
        target = args.output/(name+'.onnx')
        if (args.output/name/'manifest.json').exists():
            continue
        print('Calibrating', name, len(nodes), 'convolutions', flush=True)
        quantize_static(prepared, target, Reader(), quant_format=fmt, op_types_to_quantize=['Conv'],
                        nodes_to_quantize=nodes, per_channel=True, activation_type=activation,
                        weight_type=QuantType.QInt8, calibrate_method=CalibrationMethod.MinMax,
                        extra_options={'ActivationSymmetric':False,'WeightSymmetric':True})
        onnx.checker.check_model(onnx.load(target))
        optimized = args.output/(name+'-optimized.onnx')
        s = session(target,3,optimized)
        print(name,dict(collections.Counter(n.op_type for n in onnx.load(optimized).graph.node)),flush=True)
        manifest = json.loads((args.bundle/'manifest.json').read_text())
        bundle = args.output/name
        bundle.mkdir(exist_ok=True)
        for component in manifest['files']:
            p = bundle/component
            if not p.exists():
                p.symlink_to((args.bundle/component).resolve())
        destination = bundle/'mimi_decoder_quant.onnx'
        shutil.copy2(target,destination)
        manifest['files'][destination.name]=destination.stat().st_size
        for variant in manifest['variants'].values():
            variant['mimi_decoder']=destination.name
        manifest['experimental_decoder_quantization']=name
        manifest['quantization']['decoder']='experimental static Conv U8S8; other decoder ops FP32'
        (bundle/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def combined(args):
    """Add the same weight-only dynamic MatMul quantization as the main model."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.onnx_model import ONNXModel
    cases=[
        ('conv-none-linear-dynamic',args.bundle/'mimi_decoder.onnx'),
        ('conv-all-linear-dynamic',args.output/'conv-all-qdq-u8s8.onnx'),
        ('conv-phase-linear-dynamic',args.output/'conv-phase-qdq-u8s8.onnx'),
    ]
    for name,source in cases:
        target=args.output/(name+'.onnx')
        if (args.output/name/'manifest.json').exists():continue
        model=ONNXModel(onnx.load(source))
        model.replace_gemm_with_matmul()
        quantize_dynamic(model.model,target,per_channel=True,weight_type=QuantType.QInt8,
                         op_types_to_quantize=['MatMul'],extra_options={'MatMulConstBOnly':True})
        graph=onnx.load(target)
        count=sum(n.op_type=='MatMulInteger' for n in graph.graph.node)
        assert count==8,(name,count)
        onnx.checker.check_model(graph)
        session(target,3,args.output/(name+'-optimized.onnx'))
        manifest=json.loads((args.bundle/'manifest.json').read_text())
        bundle=args.output/name;bundle.mkdir(exist_ok=True)
        for component in manifest['files']:
            link=bundle/component
            if not link.exists():link.symlink_to((args.bundle/component).resolve())
        destination=bundle/'mimi_decoder_quant.onnx';shutil.copy2(target,destination)
        manifest['files'][destination.name]=destination.stat().st_size
        for variant in manifest['variants'].values():variant['mimi_decoder']=destination.name
        manifest['experimental_decoder_quantization']=name
        manifest['quantization']['decoder']='dynamic INT8 linear layers; '+('FP32 convolutions' if 'none' in name else 'static INT8 convolutions')
        (bundle/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(name,count,'quantized linear layers',flush=True)


def decode(s, latent, chunk):
    state = initial(s)
    audio=[]; times=[]
    for start in range(0,latent.shape[1],chunk):
        z=latent[:,start:start+chunk].copy()
        before=time.perf_counter()
        a,state=advance(s,z,state)
        times.append((time.perf_counter()-before)*1000)
        if not np.isfinite(a).all():
            raise ValueError('Non-finite decoder output')
        audio.append(a)
    return np.concatenate(audio),times


def evaluate(args):
    records=json.loads((args.output/'corpus/manifest.json').read_text())['records']
    records=[r for r in records if r['split']=='evaluation']
    paths=[('fp32',args.bundle/'mimi_decoder.onnx')]+[(p.stem,p) for p in sorted(args.output.glob('conv-*.onnx')) if not p.stem.endswith('-optimized') and (not args.candidates or p.stem in args.candidates)]
    references={}
    s=session(paths[0][1],1)
    for r in records:
        latent=np.load(args.output/'corpus'/(r['name']+'.npy'))
        for chunk in [1,4]:
            references[r['name'],chunk]=decode(s,latent,chunk)[0]
    del s
    results=[]
    # Only one threaded session exists during timing; idle ORT pools must not
    # spin in the background and contaminate other variants' measurements.
    for name,path in paths:
        s=session(path,3)
        decode(s,np.load(args.output/'corpus'/(records[0]['name']+'.npy'))[:,:8],4)
        for r in records:
            latent=np.load(args.output/'corpus'/(r['name']+'.npy'))
            for chunk in [1,4]:
                reference=references[r['name'],chunk]
                audio,times=decode(s,latent,chunk)
                assert audio.shape==reference.shape
                error=audio.astype(np.float64)-reference
                snr=float(10*np.log10(max(1e-30,np.mean(reference.astype(np.float64)**2))/max(1e-30,np.mean(error**2))))
                row=dict(case=r['name'],variant=name,chunk=chunk,frames=latent.shape[1],snr_db=snr,
                         relative_rmse=float(np.sqrt(np.mean(error**2)/max(1e-30,np.mean(reference.astype(np.float64)**2)))),
                         max_error=float(np.max(np.abs(error))),total_ms=sum(times),first_ms=times[0],
                         median_ms=float(np.median(times[2:])))
                results.append(row)
                if chunk==4:
                    wavfile.write(args.output/(r['name']+'-'+name+'.wav'),24000,audio)
                print(json.dumps(row),flush=True)
        del s
    summary={}
    for name,_ in paths:
        summary[name]={}
        for chunk in [1,4]:
            rows=[r for r in results if r['variant']==name and r['chunk']==chunk]
            summary[name][chunk]=dict(ms_per_frame=sum(r['total_ms'] for r in rows)/sum(r['frames'] for r in rows),
                                     median_snr_db=float(np.median([r['snr_db'] for r in rows])),
                                     worst_snr_db=min(r['snr_db'] for r in rows))
    (args.output/'evaluation.json').write_text(json.dumps(dict(results=results,summary=summary),indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


def benchmark(args):
    """Production C++ engine, seeded AR latents, sequential independent engines."""
    cases=[('baseline-int8',args.bundle)]
    cases += [(p.name,p) for p in sorted(args.output.glob('conv-*')) if p.is_dir() and (p/'manifest.json').exists() and (not args.candidates or p.name in args.candidates)]
    rows=[]
    expected={}
    for repeat in range(3):
        for name,bundle in (cases if repeat%2==0 else cases[::-1]):
            result=subprocess.run([str(args.benchmark_binary.resolve()),str(bundle.resolve()),str(args.voices[0].resolve())],
                                  capture_output=True,text=True,check=True,timeout=240)
            report=json.loads(result.stdout)
            for row in report['runs']:
                key=row['run']
                expected.setdefault(key,row['audio_s'])
                assert row['audio_s']==expected[key], 'Decoder change altered the AR generation length'
                rows.append(dict(row,variant=name,repeat=repeat,load_s=report['load_s']))
            print(name,repeat,flush=True)
    summary={}
    for name,_ in cases:
        warm=[r for r in rows if r['variant']==name and r['stage']=='warm']
        summary[name]=dict(aggregate_rtfx=sum(r['audio_s'] for r in warm)/sum(r['generation_s'] for r in warm),
                           median_ttfa_ms=statistics.median(r['ttfa_ms'] for r in warm),warm_requests=len(warm))
    (args.output/'benchmark.json').write_text(json.dumps(dict(seed=42,temperature=0.3,ar_threads=1,decoder_threads=3,
                                                           max_chunk=4,rows=rows,summary=summary),indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['collect','quantize','combined','evaluate','benchmark'])
    p.add_argument('--bundle',type=Path,default=Path('models-v2-2026-10/english'))
    p.add_argument('--candidates',nargs='+',help='Restrict evaluation/benchmark to named candidates')
    p.add_argument('--benchmark-binary',type=Path,default=Path('.cache/conv-quantization/benchmark-native'))
    p.add_argument('--source',type=Path,default=Path('../pocket3'))
    p.add_argument('--output',type=Path,default=Path('.cache/conv-quantization'))
    p.add_argument('--voices',nargs='+',type=Path,default=[Path('.cache/year-study/voice.wav'),Path('voices/cate.wav')])
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    globals()[args.stage](args)

if __name__=='__main__':main()
