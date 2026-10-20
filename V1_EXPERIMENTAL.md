# Legacy English CPU optimization

`v1-experimental` starts from `mine`. It keeps the v1 English weights,
SentencePiece tokenizer and split FP16 KV caches. It requires no Rust or v2
runtime. The original `models` directory remains unchanged.

The upstream numeric-year omission is filed as
[kyutai-labs/pocket-tts#395](https://github.com/kyutai-labs/pocket-tts/issues/395).

## Try it

The optimized directories have already been generated locally:

| Directory | Changes |
| --- | --- |
| `models` | Original legacy models |
| `models-v1-cpu` | Phase-convolution decoder, FP32 and INT8 |
| `models-v1-attention` | Strided main attention, FP32 and INT8 |
| `models-v1-kv` | Both optimizations |

```sh
cmake -S . -B .build -DCMAKE_BUILD_TYPE=Release
cmake --build .build -j4
./ptts --models-dir models-v1-kv "Hello, world." narrator output.wav
```

INT8 remains the default. Maximum decoder chunk size now defaults to four,
matching the setting used for previous CPU tests. Explicit `--tokenizer` paths
still work; otherwise the tokenizer is read from `--models-dir`. Existing
server and Android commands work with the new model directory. No Android
device measurements have been collected for these graphs yet.

The Mac shared-library loader fixes from v2 are included: loader-relative ORT
lookup, copying the versioned dylib, and applying `patchelf` only on Linux.
Thread allocation and sampling defaults retain `mine` behavior.

## Generate the graphs

Use an existing validated v1 ONNX directory. No checkpoint download or fresh
PyTorch export is needed:

```sh
pip install -r requirements-optimize-v1.txt
python optimize_onnx_v1.py --models models --output models-v1-kv
```

Outputs must be new directories. For independent A/B tests, add `--stage decoder`
or `--stage attention`. Applying attention to a decoder-only directory also works;
the report retains both stages. `cpu_optimization.json` records component error
metrics and the rewrites applied. The runtime does not need that report.

## Applicable v2 gains

- Four ConvTranspose nodes per decoder become ordinary convolutions over output
  phases, followed by interleaving. INT8 matrix products stay quantized; this
  does not replace the legacy INT8 decoder with a FP32 decoder.
- Twelve main-transformer attention products read the existing sequence-major
  K/V storage using CPU `com.microsoft.FusedMatMul`. The two score scale factors
  remain unchanged; twelve history transposes become unnecessary.

V1 already slices attention to the populated KV prefix and already uses split,
single-buffered FP16 caches. The corresponding v2 runtime changes add nothing
here. Multilingual configuration, JSON tokenization, fused prefill, single-frame
specializations and voice-cache experiments are excluded.

## Mac measurements

2026-10-20, M1 CPU, native ORT 1.25.0, narrator voice, INT8, temperature zero,
seed 42, AR1/decoder3, first chunk one and subsequent chunks four. Each process
performs one cold voice request and three warm requests with disk caching off.
Three processes per variant/workload use alternating model order. Entries below
are medians of nine warm requests. Synthesis uses the same runtime for all graphs.

| Workload | Original RTFX | Decoder | Attention | Combined | Combined gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short greeting | 15.31x | 15.78x | 15.55x | 16.26x | +6.2% |
| Two sentences | 14.87x | 14.92x | 15.85x | 16.17x | +8.8% |
| Supplied Glück/year passage | 12.30x | 12.45x | 13.65x | 13.69x | +11.2% |

Warm TTFA is essentially unchanged: original/combined are 19.5/19.3 ms,
31.0/31.0 ms and 76.7/75.6 ms respectively. Attention accounts for most of the
throughput improvement. Decoder-only gains are small and workload-dependent.
Short/medium generated durations match; the optimized-attention year passage
is one frame shorter (14.64 versus 14.72 seconds), so throughput is normalized
to the actual audio duration. These are Mac results, not Android speedup claims.

## Numerical and speech checks

The optimizer checks model interfaces, 57 main-transformer calls including
prefill, BOS, repeated generation, multi-frame input and final-context writes,
and 44 decoder calls / 300 latent frames with variable chunk sizes and circular
cache wraparound. Main FP32 results are bit-exact; main INT8 maximum error is
`9.54e-7` across the tested calls.

Decoder arithmetic changes rounding. Synthetic streaming audio SNR is 92.6 dB
for FP32 and 36.2 dB for INT8 relative to the original matching precision.
The original strict v2 bit-exact/tiny-error gates did not transfer to the legacy
FP16 decoder. The v1 validator explicitly reports numerical differences and
checks relative RMS drift and output SNR; INT8 dynamic scales amplify the small
convolution differences. A real listening comparison remains appropriate.

Whisper large-v3 transcribed the short, medium and year passages from the
original, attention-only and combined models. All nine transcripts contain
the same words after punctuation normalization. The complete numeric year
`1943` and following `her father` are preserved in all three year samples.
The legacy-default temperature 0.7 / seed 42 CLI samples also have matching
transcriptions, including the complete year and following phrase. A native FP32
waveform comparison on the medium text measured 93.1 dB SNR versus the original.
The normal Mac CLI/shared build passes, and the shared library successfully
creates an optimized v1 engine.

Raw timing reports, reference/optimized speech, and ASR transcripts are in
`.cache/v1-study/`. The benchmark source is `scripts/benchmark_v1_cpu.cpp`.

To build that harness using the normal Mac build dependencies:

```sh
c++ -O3 -std=c++17 \
  -I.build/_deps/onnxruntime-src/include \
  -I.build/_deps/sentencepiece-src/src -I.build/_deps/dr_libs-src \
  scripts/benchmark_v1_cpu.cpp -o .cache/v1-study/benchmark \
  .build/_deps/onnxruntime-src/lib/libonnxruntime.dylib \
  .build/_deps/sentencepiece-build/src/libsentencepiece.a \
  -Wl,-rpath,"$PWD/.build/_deps/onnxruntime-src/lib"
.cache/v1-study/benchmark models-v1-kv voices/narrator.wav int8 \
  "Hello, world." .cache/v1-study/report.json .cache/v1-study/warm.wav
```
