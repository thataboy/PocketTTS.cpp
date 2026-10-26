#!/usr/bin/env python3
"""Verify decoder selection, integer kernels, and unchanged upstream components."""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import onnx
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundles', type=Path, default=Path('models-v2-int8-linear-decoder'))
    parser.add_argument('--baseline', type=Path, default=Path('models-v2-2026-10'))
    parser.add_argument('--binary', type=Path, default=Path('ptts'))
    parser.add_argument('--voice', type=Path, required=True)
    args = parser.parse_args()
    for language, text in [('english', 'She was born in New York in 1943. Her father was a Hungarian Jewish immigrant.'),
                           ('french', 'Elle est née à New York en 1943. Son père était un immigrant juif hongrois.')]:
        bundle = (args.bundles / language).resolve()
        manifest = json.loads((bundle/'manifest.json').read_text())
        fp32 = manifest['variants']['fp32']['mimi_decoder']
        int8 = manifest['variants']['int8']['mimi_decoder']
        assert fp32 != int8
        decoder_ops = [n.op_type for n in onnx.load(bundle/int8).graph.node]
        assert decoder_ops.count('Conv') == 13, decoder_ops
        assert not {'QuantizeLinear', 'DequantizeLinear', 'QLinearConv'}.intersection(decoder_ops)
        for name in [fp32, int8]:
            assert name in manifest['files'] and (bundle/name).is_file()
        with tempfile.TemporaryDirectory(dir='.cache', prefix='decoder-bundle-') as temp:
            root = Path(temp)
            optimized = root/'optimized.onnx'
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.optimized_model_filepath = str(optimized)
            s = ort.InferenceSession(str(bundle/int8), opts, providers=['CPUExecutionProvider'])
            ops = [n.op_type for n in onnx.load(optimized).graph.node]
            assert ops.count('QLinearConv') == 0, ops
            assert ops.count('DynamicQuantizeMatMul') == 8, ops
            del s
            float_ops = {n.op_type for n in onnx.load(bundle/fp32).graph.node}
            assert not float_ops.intersection({'QuantizeLinear', 'MatMulInteger', 'QLinearConv'})
            # Isolate AR/sampler parity by selecting the unchanged FP32 decoder
            # in a temporary INT8 control bundle. The real bundle is never edited.
            control = root/'control'
            control.mkdir()
            for name in manifest['files']:
                (control/name).symlink_to(bundle/name)
            manifest['variants']['int8']['mimi_decoder'] = fp32
            (control/'manifest.json').write_text(json.dumps(manifest))
            def audio(path, precision):
                return subprocess.check_output([str(args.binary.resolve()), '--models-dir', str(path),
                    '--precision', precision, '--threads', '1', '3', '--no-cache',
                    '--seed', '42', '--temperature', '0.3', text, str(args.voice.resolve())],
                    stderr=subprocess.PIPE, timeout=120)
            for precision, candidate in [('fp32', bundle), ('int8', control)]:
                assert audio(args.baseline/language, precision) == audio(candidate, precision), (language, precision)
        print(f'PASS {language}: FP32 Conv + 8 INT8 linear kernels; FP32 and AR/sampler audio preserved', flush=True)


if __name__ == '__main__':
    main()
