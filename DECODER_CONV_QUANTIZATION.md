# Decoder convolution quantization exploration

Final listening update (2026-10-25): convolution quantization is **rejected
for production**. The user heard background hiss in the selective-convolution
sample and stronger hiss in the combined sample; the linear-only sample sounded
clean. Prosody, transcription and waveform checks did not reveal this problem.
The production exporter now keeps all convolutions FP32 and quantizes only the
decoder's eight linear operators, without calibration inputs. See `V2.md`.

The user subsequently confirmed the rebuilt linear-only bundle sounds good and
provides a small but real improvement. This is the accepted configuration.
The exploratory matched run measured +2.5% aggregate warm RTFX; treat that as
workload-specific, not a guaranteed gain. Rebuilt English narrator audio is
byte-identical to `narrator-int8-linear-decoder.wav`. English/French kernel and
component-parity checks pass; all 13 convolutions remain FP32 and eight linear
operators use DynamicQuantizeMatMul. No C++ runtime changes are needed.

The following measurements describe historical experiments, including the
subsequently withdrawn convolution implementation. Experimental bundles and
samples remain available for reference. Work is uncommitted on `v2` at
`382a1e9`; small-model work is preserved on `v2-small-model` at `5460ad2`.

**Every native end-to-end test uses the 100M INT8 backbone and sampler.**
The earlier "FP32 reference" label refers only to the decoder, matching the
current production INT8 default. Convolution-only and combined decoder tests
keep the same INT8 backbone/sampler; they do not compare a fully FP32 model
against a fully INT8 model.

## Result

On M1 CPU, static convolution quantization improves isolated four-frame decoder
execution by about 15–20%, but only about 3–5% carries through to warm end-to-end
RTFX. Waveform differences are substantial. Retain the FP32 decoder as the
production default; these candidates need a listening comparison before any
quality judgement or adoption. Android performance has not been measured.
The combined Conv/linear INT8 follow-up below reaches about 7% end-to-end gain;
it also uses the existing INT8 backbone/sampler throughout.

| Decoder | Quantized convs | Decoder ms/frame, chunk 4 | Warm RTFX | Warm median TTFA | Held-out median SNR |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 reference | 0 | 4.68 | 15.05x | 33.69 ms | reference |
| All, QDQ | 13 | 3.73 | 15.75x (+4.7%) | 32.34 ms | 15.00 dB |
| All, QOperator | 13 | 4.00 | 15.82x (+5.1%) | 32.55 ms | 15.00 dB |
| Inner, QDQ | 11 | 3.82 | 15.65x (+4.0%) | 33.39 ms | 15.95 dB |
| Heavy, QDQ | 4 | 3.84 | 15.49x (+3.0%) | 33.43 ms | 17.78 dB |
| Upsampling phases, QDQ | 3 | 3.97 | 15.52x (+3.2%) | 33.35 ms | 20.29 dB |

The broad QDQ/QOperator candidates produce the same held-out waveforms. Small
speed differences between their timings should not be overinterpreted. Across
three end-to-end rounds, the all-QOperator gain ranged from 4.0% to 6.8%; the
phase-only gain ranged from 0.9% to 6.5%. Decoder speed alone overstates the
benefit because the autoregressive generator runs in parallel with decoding.

SNR measures waveform error against the same-latent FP32 decoder, not perceived
quality or voice similarity. Phase-only quantization has the smallest error
among these candidates, but still has approximately 10% relative RMS waveform
error (worst held-out SNR 18.83 dB). The all-convolution candidate has approximately
18% relative RMS error (worst 13.18 dB). This is a fidelity tradeoff, not a
transparent numeric rewrite like the earlier phase-convolution optimization.

## Scope and method

### Combined decoder INT8 follow-up

The follow-up adds dynamic per-channel INT8 to the decoder's eight constant-weight
linear/MatMul operations, both alone and together with the calibrated Conv
quantization. Runtime optimization fuses those operations into eight
DynamicQuantizeMatMul kernels. Attention matrix products with variable weights,
normalization, activations and streaming state remain FP32; "quantized decoder"
does not mean every operator runs in INT8.

All rows below use the same existing 100M INT8 backbone and sampler. These are
a new matched run; compare within this table rather than absolute RTFX from
the earlier table. Same hardware, settings, seeds, texts and 18 warm requests
per variant apply. The generated audio duration agrees for every seeded request.

| Decoder | Warm RTFX | Gain vs current INT8 default | Warm median TTFA | Held-out median SNR |
| --- | ---: | ---: | ---: | ---: |
| Current FP32 decoder | 14.47x | reference | 35.02 ms | reference |
| Eight INT8 linear layers only | 14.83x | +2.5% | 34.66 ms | 29.26 dB |
| Thirteen INT8 Conv nodes only | 15.05x | +4.0% | 34.40 ms | 15.00 dB |
| Thirteen Conv + eight linear layers INT8 | 15.49x | +7.1% | 33.61 ms | 14.77 dB |
| Three phase Conv + eight linear layers INT8 | 15.03x | +3.9% | 33.56 ms | 19.60 dB |

The combined all-Conv/linear gain ranged from 5.2% to 8.9% across the three
rounds. Quantizing linear layers only has much smaller waveform error, but
its speed gain ranged from 0.1% to 4.3%. SNR alone does not determine audible
quality; matched native samples are available for listening:

- `narrator-current-int8.wav`: current INT8 default, FP32 decoder.
- `narrator-int8-linear-decoder.wav`: INT8 decoder linear layers only.
- `narrator-int8-combined-decoder.wav`: INT8 decoder Conv and linear layers.
- `narrator-int8-selective-decoder.wav`: three INT8 phase Conv nodes and INT8 linear layers.

All other model components, input voice, text, seed and temperature agree in
these samples. No production defaults were changed. The earlier run is preserved
as `benchmark-convs.json`/`evaluation-convs.json`; follow-up reports are
`benchmark-combined.json`/`evaluation-combined.json` under `.cache/conv-quantization/`.

Reproduce after the convolution quantization stage:

```sh
.venv/bin/python scripts/explore_decoder_quantization.py combined
.venv/bin/python scripts/explore_decoder_quantization.py evaluate --candidates conv-none-linear-dynamic conv-all-qdq-u8s8 conv-phase-qdq-u8s8 conv-all-linear-dynamic conv-phase-linear-dynamic
.venv/bin/python scripts/explore_decoder_quantization.py benchmark --candidates conv-none-linear-dynamic conv-all-qdq-u8s8 conv-phase-qdq-u8s8 conv-all-linear-dynamic conv-phase-linear-dynamic
```

Historical command for the rejected combined decoder (audible hiss):

```sh
./ptts --models-dir .cache/conv-quantization/conv-all-linear-dynamic --threads 1 3
```

### Convolution calibration

The source is `models-v2-2026-10/english` (100M English checkpoint, pocket3
`a922b912dfa366975294afb568d815b305acde5d`). Backbone and sampler remain the
existing dynamic INT8 variants during native tests. The decoder's linear
layers, attention, state buffers and non-Conv operators remain FP32.

All candidates use per-output-channel signed INT8 weights and statically
calibrated asymmetric unsigned INT8 activations, with MinMax calibration.
The existing phase-convolution rewrite turns ConvTranspose into Conv; those
Conv nodes are eligible too. Runtime optimization selects QLinearConv kernels.

- All: all 13 Conv nodes, including the pre-transformer projection/upsampling.
- Inner: keeps the initial latent projection and final waveform Conv in FP32.
- Heavy: the post-transformer seven-tap Conv plus three decoder phase Conv nodes.
- Phases: only the three decoder phase Conv nodes; the pre-transformer grouped
  upsampler and all other convolutions remain FP32.

Calibration uses eight utterances (37.20 seconds) from narrator and cate;
four different held-out utterances total 20.96 seconds. Raw sampled latents
are captured from Python generation at temperature 0.3, seeds 100–115. Each
utterance resets decoder state. The calibrator receives real FP32 streaming
states, including first-frame and four-frame calls. This is a small pilot
corpus, not broad production calibration.

Evaluation decodes identical held-out latents through each candidate with
one-frame and four-frame chunks, checks finite output and matching lengths,
and writes matched WAV files. Decoder timings use Python ORT 1.24.4, three
threads, no CPU arena, and one active session at a time. The baseline is warmed
before timing. Optimized ONNX files are inspection artifacts only; experimental
bundles use the portable, unoptimized quantized graphs.

End-to-end measurements use the unchanged C++ engine with ORT 1.25.0,
AR1/decoder3, first chunk 1, maximum chunk 4, temperature 0.3 and fixed per-request
seeds. Each candidate has three independent engine runs with one cold and six
warm requests (18 timed warm requests total), alternating candidate order.
Disk caches are disabled; resident voice/KV reuse remains enabled. Every
candidate produces the same audio duration per seeded request: only decoding
changes. RTFX is sum(audio duration) / sum(generation time); TTFA is the median
first callback time. Model load and first-voice requests are excluded from the
warm numbers.

## Reproduce

The current workspace uses `.build-cpp-tokenizer` for the normal Mac build and
existing dependency sources under `.build-v2/_deps`. Adjust those paths for
another build directory. No production C++ modifications or new build dependency
are required.

```sh
cmake --build .build-cpp-tokenizer -j4
HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 .venv/bin/python scripts/explore_decoder_quantization.py collect
.venv/bin/python scripts/explore_decoder_quantization.py quantize
.venv/bin/python scripts/explore_decoder_quantization.py evaluate

c++ -O3 -funroll-loops -march=native -std=c++17 -DPTT_JSON_TOKENIZER \
  -I.build-v2/_deps/onnxruntime-src/include \
  -I.build-v2/_deps/sentencepiece-src/src -I.build-v2/_deps/dr_libs-src \
  scripts/benchmark_decoder_quantization.cpp .build-cpp-tokenizer/libpocket-text.a \
  .build-v2/_deps/onnxruntime-src/lib/libonnxruntime.dylib \
  .build-cpp-tokenizer/_deps/sentencepiece-build/src/libsentencepiece.a \
  -Wl,-rpath,"$PWD/.build-v2/_deps/onnxruntime-src/lib" \
  -o .cache/conv-quantization/benchmark-native
.venv/bin/python scripts/explore_decoder_quantization.py benchmark
```

Run evaluation/benchmark without concurrent builds, generation or ASR. Collection
can take explicit WAV paths with `--voices`. Use a new `--output` directory when
changing the calibration data; quantization resumes by skipping existing bundle
manifests. Original model bundles are never overwritten. Experimental bundles
contain symlinks to unchanged source components and their own quantized decoder;
copy with symlink dereferencing if moving them to another machine.

Raw reports, calibration corpus, candidate bundles and matched audio are under
`.cache/conv-quantization/`. The native narrator samples use the same text, voice,
AR/sampler INT8 weights, seed 42, temperature 0.3 and chunk settings:

- `narrator-fp32-codec.wav`: production reference.
- `narrator-conv-phase.wav`: three quantized phase convolutions.
- `narrator-conv-all.wav`: all 13 convolutions, QOperator format.

A local CLI listening test for the selective candidate:

```sh
./ptts --models-dir .cache/conv-quantization/conv-phase-qdq-u8s8 \
  --seed 42 "Your listening-test text." narrator output.wav
```

The original model path continues to select the FP32 decoder. No production
setting or model file has been replaced.
