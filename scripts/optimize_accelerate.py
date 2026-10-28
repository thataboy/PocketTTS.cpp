#!/usr/bin/env python3
"""Build an experimental Apple Accelerate V2 bundle.

Requires the optional pocket-custom-ops CMake target and matching Python ORT.
Unchanged components are symlinked to the source bundle. The source is read-only.
Convolution rewrite adapted from PocketTTS-RAVEN (native_accelerate/RAVEN_LICENSE).
"""
import argparse
import collections
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper


def rewrite_convolutions(source, target):
    model = onnx.load(source)
    g = model.graph
    weights = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    consumers = collections.defaultdict(list)
    for node in g.node:
        for name in node.input:
            consumers[name].append(node)
    outputs = {o.name for o in g.output}
    replacements, remove = {}, set()
    for node in g.node:
        if node.op_type != 'Conv' or node.input[1] not in weights:
            continue
        # Preserve the projection before the INT8 transformer. Changing its
        # FP32 reduction order can move dynamic-quantization rounding boundaries.
        if node.name == '/quantizer/output_proj/Conv':
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        if (attrs.get('strides', [1]) != [1] or attrs.get('dilations', [1]) != [1]
                or attrs.get('group', 1) != 1 or any(attrs.get('pads', [0, 0]))
                or attrs.get('auto_pad', b'NOTSET') != b'NOTSET'):
            continue
        weight = weights[node.input[1]]
        if weight.ndim != 3 or weight.dtype != np.float32:
            raise ValueError(f'Unexpected convolution weights: {node.name}')
        prefix = node.name + '/accelerate'
        packed_name = prefix + '/weight'
        g.initializer.append(numpy_helper.from_array(
            np.ascontiguousarray(weight.transpose(2, 0, 1)), packed_name))
        if len(node.input) > 2:
            bias = node.input[2]
        else:
            bias = prefix + '/bias'
            g.initializer.append(numpy_helper.from_array(np.zeros(weight.shape[0], np.float32), bias))
        op, output = 'AccelConv', node.output[0]
        users = consumers[output]
        if len(users) == 1 and users[0].op_type == 'Elu' and output not in outputs:
            alpha = next((helper.get_attribute_value(a) for a in users[0].attribute if a.name == 'alpha'), 1.0)
            if alpha == 1.0:
                op, output = 'AccelConvElu', users[0].output[0]
                remove.add(users[0].name)
        replacements[node.name] = helper.make_node(
            op, [node.input[0], packed_name, bias], [output],
            name=prefix, domain='pockettts')
    if len(replacements) != 8:
        raise ValueError(f'Expected eight post-transformer convolutions, found {len(replacements)}')
    nodes = [replacements.get(n.name, n) for n in g.node if n.name not in remove]
    del g.node[:]
    g.node.extend(nodes)
    used = {i for n in nodes for i in n.input} | outputs
    retained = [t for t in g.initializer if t.name in used]
    del g.initializer[:]
    g.initializer.extend(retained)
    if not any(o.domain == 'pockettts' for o in model.opset_import):
        model.opset_import.append(helper.make_opsetid('pockettts', 1))
    # Stale intermediate annotations can disagree after fusing ELU.
    del g.value_info[:]
    onnx.checker.check_model(model)
    onnx.save(model, target)
    return dict(collections.Counter(n.op_type for n in replacements.values()))


def rewrite_attention(source, target):
    m = onnx.load(source)
    nodes = {n.name: n for n in m.graph.node}
    replacements = {}
    if sum(n.op_type == 'FusedMatMul' for n in m.graph.node) != 12:
        raise ValueError('Expected the strided-attention V2 graph')
    for layer in range(6):
        p = f'/transformer/layers.{layer}/self_attn/'
        for name, op in [('Mul', 'Mul'), ('Mul_1', 'Mul'), ('ScatterElements', 'ScatterElements'),
                         ('ScatterElements_1', 'ScatterElements'), ('Where', 'Where'), ('Reshape_2', 'Reshape')]:
            if nodes[p+name].op_type != op:
                raise ValueError(f'Unexpected attention pattern: {p+name}')
        end = nodes[p+'Reshape_2']
        replacements[end.name] = helper.make_node('SplitAttention',
            [nodes[p+'Mul'].output[0], nodes[p+'ScatterElements'].output[0],
             nodes[p+'ScatterElements_1'].output[0], nodes[p+'Where'].output[0],
             nodes[p+'Mul_1'].input[1]], list(end.output), name=p+'SplitAttention', domain='pockettts')
    new = [replacements.get(n.name, n) for n in m.graph.node]
    needed = {o.name for o in m.graph.output}
    kept = []
    for n in reversed(new):
        if any(o in needed for o in n.output):
            kept.append(n)
            needed.update(n.input)
    del m.graph.node[:]
    m.graph.node.extend(reversed(kept))
    del m.graph.value_info[:]
    referenced = {i for n in kept for i in n.input} | {o.name for o in m.graph.output}
    initializers = [i for i in m.graph.initializer if i.name in referenced]
    del m.graph.initializer[:]
    m.graph.initializer.extend(initializers)
    if not any(o.domain == 'pockettts' for o in m.opset_import):
        m.opset_import.append(helper.make_opsetid('pockettts', 1))
    onnx.checker.check_model(m)
    onnx.save(m, target)
    return {'SplitAttention': 6, 'nodes_before': len(new), 'nodes_after': len(kept)}


def session(path, library, threads=1):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    opts.register_custom_ops_library(str(library))
    return ort.InferenceSession(str(path), opts, providers=['CPUExecutionProvider'])


def initial(s):
    types = {'tensor(float)': np.float32, 'tensor(int64)': np.int64, 'tensor(bool)': np.bool_}
    return {i.name: np.zeros(i.shape, types[i.type]) for i in s.get_inputs()[1:]}


def validate_convolutions(source, target, library):
    sessions = [session(p, library) for p in (source, target)]
    for method in ('get_inputs', 'get_outputs'):
        interfaces = [[(i.name, i.type, i.shape) for i in getattr(s, method)()] for s in sessions]
        if interfaces[0] != interfaces[1]:
            raise ValueError('Decoder interface changed')
    states = [initial(s) for s in sessions]
    rng = np.random.default_rng(123)
    signal = error = max_audio = max_state = 0.0
    batches = [1, 2, 4, 8, 15, 16] * 12
    for frames in batches:
        z = rng.normal(size=(1, frames, 32)).astype(np.float32)
        results = [s.run(None, {'latent': z, **state}) for s, state in zip(sessions, states)]
        for state, result in zip(states, results):
            state.update(zip(state, result[1:]))
        for index, (ref, actual) in enumerate(zip(*results)):
            if ref.dtype == np.float32:
                np.testing.assert_allclose(actual, ref, atol=2e-5, rtol=2e-4)
                diff = float(np.max(np.abs(actual - ref), initial=0))
                if index:
                    max_state = max(max_state, diff)
            else:
                np.testing.assert_array_equal(actual, ref)
        ref, actual = (r[0].astype(np.float64) for r in results)
        signal += float(np.sum(ref**2))
        error += float(np.sum((actual-ref)**2))
        max_audio = max(max_audio, float(np.max(np.abs(actual-ref))))
    snr = 10 * np.log10(signal / max(error, 1e-30))
    if snr < 95 or max_audio > 2e-5:
        raise ValueError(f'Audio acceptance failed: SNR={snr}, max={max_audio}')
    return dict(snr_db=float(snr), max_audio_error=max_audio, max_state_error=max_state,
                latent_frames=sum(batches), calls=len(batches), ort_version=ort.__version__)


def validate_attention(paths, library):
    # Four independent, teacher-forced trajectories: distinguish FP32 kernel
    # parity from amplification by later dynamic INT8 rounding boundaries.
    sessions = {name: session(path, library) for name, path in paths.items()}
    inputs = sessions['fp32_reference'].get_inputs()[2:]
    reference_interface = [[(i.name, i.type, i.shape) for i in getattr(sessions['fp32_reference'], method)()]
                           for method in ('get_inputs', 'get_outputs')]
    for s in sessions.values():
        interface = [[(i.name, i.type, i.shape) for i in getattr(s, method)()]
                     for method in ('get_inputs', 'get_outputs')]
        if interface != reference_interface:
            raise ValueError('Attention changed the streaming interface')

    def initial_state(offset=0):
        return {i.name: np.full(i.shape, offset if i.type == 'tensor(int64)' else 0,
                               np.int64 if i.type == 'tensor(int64)' else np.float32) for i in inputs}

    states = {name: initial_state() for name in sessions}
    rng = np.random.default_rng(123)
    metrics = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
    cases = [(None, 0, 48), (None, 0, 24)] + [(None, 1, 0)]*48 + [(None, 4, 0), (None, 0, 17), (None, 1, 0)]
    capacity = inputs[0].shape[1]
    cases += [(1, 1, 0), (capacity//2, 1, 0), (capacity-1, 1, 0), (capacity-8, 0, 8)]
    for step, (offset, frames, tokens) in enumerate(cases):
        if offset is not None:
            states = {name: initial_state(offset) for name in sessions}
        sequence = rng.normal(0, .1, (1, frames, 32)).astype(np.float32)
        if step == 2:
            sequence.fill(np.nan)
        text = rng.normal(0, .1, (1, tokens, 1024)).astype(np.float32)
        results = {name: s.run(None, dict(sequence=sequence, text_embeddings=text, **states[name]))
                   for name, s in sessions.items()}
        for name, result in results.items():
            for output in result:
                if not np.isfinite(output).all():
                    raise ValueError(f'Nonfinite {name} attention output at call {step}')
            states[name] = dict(zip((i.name for i in inputs), result[2:]))
        for ref, actual in zip(results['fp32_reference'], results['fp32_candidate']):
            if ref.dtype == np.float32:
                np.testing.assert_allclose(actual, ref, atol=2e-5, rtol=2e-4)
            else:
                np.testing.assert_array_equal(actual, ref)
        for ref, actual in zip(results['int8_reference'][2:], results['int8_candidate'][2:]):
            if ref.dtype != np.float32:
                np.testing.assert_array_equal(actual, ref)
        for precision in ('fp32', 'int8'):
            for index, component in enumerate(('conditioning', 'eos')):
                ref = results[precision+'_reference'][index].astype(np.float64)
                actual = results[precision+'_candidate'][index].astype(np.float64)
                values = metrics[precision+'_'+component]
                values[0] += float(np.sum(ref**2))
                values[1] += float(np.sum((actual-ref)**2))
                values[2] = max(values[2], float(np.max(np.abs(actual-ref))))
        for name in ('int8_reference', 'int8_candidate'):
            ref = results['fp32_reference'][0].astype(np.float64)
            actual = results[name][0].astype(np.float64)
            values = metrics[name+'_vs_fp32']
            values[0] += float(np.sum(ref**2))
            values[1] += float(np.sum((actual-ref)**2))
            values[2] = max(values[2], float(np.max(np.abs(actual-ref))))
    report = {name: dict(snr_db=float(10*np.log10(signal/max(error, 1e-30))), max_error=maximum)
              for name, (signal, error, maximum) in metrics.items()}
    report.update(calls=len(cases), ort_version=ort.__version__,
                  status='experimental; INT8 rounding changes require audio evaluation')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('bundle', type=Path)
    p.add_argument('output_dir', type=Path)
    p.add_argument('--custom-ops-library', type=Path, required=True)
    p.add_argument('--components', choices=['conv', 'attention', 'both'], default='conv')
    args = p.parse_args()
    source, target = args.bundle.resolve(), args.output_dir.resolve()
    library = args.custom_ops_library.resolve()
    if target.exists():
        p.error('Output must be a new directory')
    manifest = json.loads((source/'manifest.json').read_text())
    if manifest.get('schema_version') != 2:
        p.error('Expected a V2 bundle')
    target.parent.mkdir(parents=True, exist_ok=True)
    reports = {}
    with tempfile.TemporaryDirectory(prefix='accelerate-', dir=target.parent) as tmp:
        out = Path(tmp)
        for name in manifest['files']:
            (out/name).symlink_to(source/name)
        def select(precision, component, name, feature):
            manifest['variants'][precision][component] = name
            manifest['files'][name] = (out/name).stat().st_size
            features = manifest.setdefault('required_runtime_features', {}).setdefault(precision, [])
            if feature not in features:
                features.append(feature)

        if args.components in ('conv', 'both'):
            reports['convolutions'] = {}
            for precision in manifest['precision']:
                original = manifest['variants'][precision]['mimi_decoder']
                name = f'mimi_decoder_accelerate_{precision}.onnx'
                counts = rewrite_convolutions(source/original, out/name)
                report = validate_convolutions(source/original, out/name, library)
                reports['convolutions'][precision] = dict(kernels=counts, **report)
                select(precision, 'mimi_decoder', name, 'accelerate_conv_v1')
                print('convolutions', precision, json.dumps(report), flush=True)
            manifest['decoder_optimization'] = 'phase_conv_v1+accelerate_conv_v1'
        if args.components in ('attention', 'both'):
            if set(manifest['precision']) != {'fp32', 'int8'}:
                p.error('Attention experiment requires FP32 and INT8 controls')
            paths = {}
            for precision in manifest['precision']:
                original = manifest['variants'][precision]['flow_lm_main']
                name = f'flow_lm_main_accelerate_{precision}.onnx'
                rewrite_attention(source/original, out/name)
                paths[precision+'_reference'] = source/original
                paths[precision+'_candidate'] = out/name
                select(precision, 'flow_lm_main', name, 'accelerate_attention_v1')
            reports['attention'] = validate_attention(paths, library)
            manifest['main_optimization'] = 'split_attention_v1'
            print('attention', json.dumps(reports['attention']), flush=True)
        (out/'accelerate_validation.json').write_text(json.dumps(reports, indent=2)+'\n')
        manifest['files']['accelerate_validation.json'] = (out/'accelerate_validation.json').stat().st_size
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        # Publish only after numerical/interface validation completes.
        shutil.move(str(out), str(target))


if __name__ == '__main__':
    main()
