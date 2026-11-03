# Optimization findings — consolidated checkpoint

As of 2026-11-02. This document supersedes `V2.md`, `RAVEN_REVIEW.md` and
`DECODER_CONV_QUANTIZATION.md`; their originals are retained in the local dated
trash inventory. Current use is in [README.md](README.md), model reproduction in
[MODEL_EXPORT.md](MODEL_EXPORT.md), and text policies in
[TEXT_NORMALIZATION.md](TEXT_NORMALIZATION.md).

## Current decisions

| Work | Decision and evidence |
| --- | --- |
| Portable v2 split FP32 KV + populated prefix | Keep. Persistent input/output cache allocations avoid whole-cache copies; main cache storage fell from about 94 to 47 MiB at capacity 1,000. |
| Strided main attention | Keep in portable v2 and v1. Twelve `com.microsoft::FusedMatMul` products avoid history layout transposes. No cache precision change. |
| Decoder phase convolutions | Keep. Four ConvTranspose operations become ordinary Conv over output phases, then interleave; preserves streaming state and weights. |
| Dynamic INT8 dense operators | Keep. Backbone/sampler INT8 and eight decoder linear operators; state, attention products, encoder and convolutions stay float. |
| Apple fused AR attention | Best Mac English configuration: `models-v2-attention`. Roughly 5.6–5.7% warm normalized gain over the portable linear-decoder control. |
| Apple FP32 convolution custom ops | Set aside. Eight kernels improved isolated decoder time about 10%, but attention alone won end-to-end at AR1/decoder3. |
| Static INT8 decoder convolutions | Reject. Listening found background hiss, worse with broad coverage, despite reasonable ASR/prosody results and higher throughput. |
| Portable ARM64 fused attention / NEON | Set aside. User device results were mixed and did not beat the chosen v1 model. Code is preserved at `3118980` on `codex/v2-neon-attention`. |
| Android model | User-selected `models-v1-kv-linear`; legacy weights/tokenizer and split FP16 cache, plus attention/phase rewrites and decoder linear INT8. |
| Native JSON tokenizer | Keep. English/French Unigram/Metaspace parity, no Cargo or Rust target needed. |
| Voice cache | Keep single resident named-voice snapshot and timestamp-based disk freshness. Compact/multi-voice LRU experiments were reverted. |
| First/max chunks | One/four frames retained. Larger batches and different thread splits must win full-pipeline tests, not just decoder tests. |
| One-frame/static graph and fused-prefill prototypes | Not in the selected runtime. Earlier one-frame prototypes did not establish useful native gains; fused-prefill work remains in historical experiments. |
| Small 18.2M model | Separate `v2-small-model` branch at `5460ad2`; different interface, not the selected quality/performance configuration. |

`POCKET_ACCELERATE` controls Apple custom-op availability at build time. It does
not rewrite a model or make every attention call use Apple BLAS. The fused tail
uses NEON for a single query; multi-token prefill uses Accelerate. A bundle's
`required_runtime_features` determines whether it needs this build. The separate
`POCKET_FUSED_ATTENTION` switch belongs to the NEON experiment, not current `v2`.

## M1 measurements

### Accepted Apple attention comparison, 2026-10-26

CPU only, native ORT 1.25.0, narrator/INT8, temperature 0.3, seed 42,
AR1/decoder3, first chunk one and maximum four. Three order-rotated blocks,
five requests per block/variant, first two discarded: nine retained warm
requests per case. These measurements predate speech normalization.
RTFx is aggregate output duration / aggregate generation time; wall time,
TTFA and RSS below are medians from the recorded summary.

| Fixture / variant | RTFx | Wall time | First audio | RSS |
| --- | ---: | ---: | ---: | ---: |
| 250 chars, portable baseline | 18.37× | 721.7 ms | 35.27 ms | 743 MiB |
| 250 chars, attention | **19.40×** | 698.8 ms | 34.48 ms | 746 MiB |
| 250 chars, attention + conv | 19.19× | 708.4 ms | 34.61 ms | 751 MiB |
| 440 chars, portable baseline | 18.13× | 1283.6 ms | 35.54 ms | 747 MiB |
| 440 chars, attention | **19.16×** | 1201.0 ms | 34.61 ms | 750 MiB |
| 440 chars, attention + conv | 18.98× | 1213.6 ms | 34.62 ms | 755 MiB |

Attention-only normalized gains: 5.6%/5.7%; median wall-time reductions:
3.2%/6.4%. Generated duration changed 13.28→13.60 seconds and 23.28→23.04
seconds. Thus a shared text/seed is not an identical workload. A fixed-input,
fixed-96-frame AR benchmark at histories 128/384/768 measured 7.9%/5.3%/1.9%
higher throughput, supporting a compute gain independent of speech length.
No CPU/power/energy study has been done; a mobile energy gain is not established.

Durable summary and exact text fixtures:
[JSON](docs/benchmarks/m1-attention-2026-10-26.json),
[250 characters](docs/benchmarks/250.txt), [440 characters](docs/benchmarks/440.txt).
Detailed local reports: `.cache/raven-study/{final-summary,main-summary}.json`;
component/state results are also in each bundle's `accelerate_validation.json`.

Build and collect current raw runs (these are not automatically the historical
numbers, since the runtime now normalizes text):

```sh
cmake --build .build-v2 -j4 --target benchmark-accelerate
./benchmark-accelerate models-v2-int8-linear-decoder/english voices/narrator.wav \
  1 3 4 docs/benchmarks/250.txt 5 > /tmp/portable-250.json
./benchmark-accelerate models-v2-attention/english voices/narrator.wav \
  1 3 4 docs/benchmarks/250.txt 5 > /tmp/attention-250.json
./benchmark-accelerate --main models-v2-int8-linear-decoder/english/flow_lm_main_int8.onnx \
  > /tmp/portable-main.json
./benchmark-accelerate --main models-v2-attention/english/flow_lm_main_accelerate_int8.onnx \
  > /tmp/attention-main.json
```

Repeat the end-to-end blocks with reversed/rotated variant order, include the
440-character fixture, discard the first two requests per block, and aggregate
actual output seconds and generation seconds. Run serially without builds,
ASR or profiler tracing competing for CPU. The harness disables disk caching
but retains resident voice/KV reuse. It records first callback and output samples.

### Earlier portable, v1 and cold-process results

These are separate experiments with different prompts and must not be stacked
into a claimed cumulative speedup:

- Portable v2 strided attention, 2026-10-18: English medium 16.11→17.46×,
  long 15.90→17.40×; French medium 14.06→15.41×, long 14.17→15.61×.
  Short English was effectively flat. INT8, AR1/decoder3, temperature zero,
  first/max chunks 1/4, warmed resident voice. Reports: `.cache/main-layout-study/`.
- Legacy v1 CPU rewrites, 2026-10-20: original→combined phase/attention
  15.31→16.26× (short), 14.87→16.17× (two sentences), 12.30→13.69×
  (long year passage). Native ORT 1.25.0, INT8, temperature zero, seed 42,
  AR1/decoder3, chunks 1/4, nine warm requests. `.cache/v1-study/summary.json`.
- Subsequent v1 linear-only decoder check: 14.614→14.592× aggregate warm
  throughput, TTFA 33.48→32.69 ms on a shorter year prompt. Audio durations
  matched. Linear INT8 does not establish an additional Mac throughput gain;
  the Android selection reflects user device testing. Report:
  `.cache/v1-study/linear-audit/comparison.json`.
- Portable consolidation, 2026-10-20: cold-process median v2 English INT8
  8.56×, French INT8 8.88×, English FP32 3.36×, French FP32 3.86×;
  legacy INT8 8.72×. Voice encoding was included in TTFA, disk caching off,
  temperature zero, seed 42, AR1/decoder3, chunks 1/4. All 36 paired outputs
  matched the preserved experimental executable byte for byte.
  Reports: `.cache/v2-consolidation/`.
- October checkpoint follow-up: old/new English INT8 aggregate 8.54/8.62×,
  median cold first audio 350/348 ms. Five alternating pairs after discarding
  the first pair; the same cold-process settings. No measurable regression.
  Reports: `.cache/year-fix-2026-10/`.

## Decoder quantization: why the fastest candidate was rejected

M1 native end-to-end follow-up: INT8 backbone/sampler throughout, narrator/cate
calibration and held-out latents, AR1/decoder3, chunks 1/4, temperature 0.3,
three alternating rounds with six retained warm requests per variant (18 total).
All candidate durations matched for each seeded request.

| Decoder | Warm aggregate RTFx | Gain | Warm median TTFA | Held-out median SNR |
| --- | ---: | ---: | ---: | ---: |
| FP32 codec reference | 14.47× | — | 35.02 ms | Reference |
| Eight linear INT8 operators | 14.83× | +2.5% | 34.66 ms | 29.26 dB |
| Thirteen Conv INT8 operators | 15.05× | +4.0% | 34.40 ms | 15.00 dB |
| All Conv + linear INT8 | 15.49× | +7.1% | 33.61 ms | 14.77 dB |
| Three phase Conv + linear INT8 | 15.03× | +3.9% | 33.56 ms | 19.60 dB |

User listening found hiss in selective Conv quantization and stronger hiss in
combined quantization; linear-only sounded clean. ASR, SNR and prosody checks
had missed it. Keep all convolutions FP32. The eight-linear gain varied from
0.1% to 4.3% between rounds; it is workload-specific. The original all-Conv
pilot improved isolated four-frame decoder time by about 15–20%, but only
3–5% reached whole-pipeline throughput because generation and decoding overlap.

Python decoder timings used ORT 1.24.4; native end-to-end ORT 1.25.0. Calibration
covered eight utterances / 37.2 seconds and held-out evaluation four utterances /
20.96 seconds. Report/sample corpus: `.cache/conv-quantization/`, especially
`benchmark-combined.json`, `evaluation-combined.json` and matched narrator WAVs.
The research scripts remain available, but are not the production export recipe:
`scripts/explore_decoder_quantization.py` and `scripts/benchmark_decoder_quantization.cpp`.

## Raven and Android conclusions

The user originally observed Raven about 6% faster at 250 chars and 12% at 440
chars. Rebuilding this fork with ORT 1.25.0 did not remove that lead, and Raven's
`--keep-commas` did not materially change performance or speaking rate. Neither
ORT version nor comma removal explains the observed result.

The useful adaptation was fused AR attention while preserving our split caches,
valid-prefix bound, causal masks and both score scaling factors. Raven's packed
cache/delta interface was not copied wholesale. Prefill uses Accelerate; the
single-query tail uses SIMD. Fused scores/softmax/value accumulation removes
intermediates and dispatch. Small reduction-order changes amplify through later
dynamic INT8 quantization, explaining non-identical generated lengths.

Eight FP32 Apple convolutions achieved 118–119 dB matched-latent audio SNR over
552 frames / 72 calls per precision/language. The input projection was left on
ORT to avoid perturbing downstream INT8 rounding. Good component timing and
numerical parity still did not beat attention alone at the chosen thread split.

English attention checks retained the long passage and 1943 regression in ASR;
FP32 temperature-zero free-running speech matched the control at 73.4 dB SNR
with equal sample count. French prose controls passed, but a numeric-year
case worsened from 20% to 46.7% ASR WER. An ORT-prefill prototype restored
baseline WER but still spoke the wrong year and was not adopted for Apple.

The portable ARM64 attention experiment used a separate feature flag/bundle and
was tested on the user's Android device. Supplied mixed results were around
v2 baseline performance, with roughly 1–2% aggregate benefit in one comparison;
v1-kv remained faster in most warm cases and produced longer/slower speech.
The latest user-selected Android model is v1-kv-linear. This checkpoint does
not claim a new device benchmark or measured energy benefit. Older Galaxy Tab
S5e controls measured about 1.77× v1 versus 1.675× v2-kv, but predate the final
linear model; do not label those as final v1-kv-linear numbers.

Do not force equal audio lengths to compare compute: checkpoint/precision
changes alter EOS and prosody even at a fixed seed. Report actual frames,
duration, wall time and RTFx, and use teacher-forced fixed-frame tests for kernels.
A slower speaking pace is a separate preference from faster inference.

## Other findings and remaining work

- Decoder allocation/arena changes did not establish a material native gain;
  retain memory-retention safeguards. Larger batches can hurt whole-pipeline
  throughput even when one decoder call becomes cheaper.
- Raven's fused upsampling/overlap and decoder delta-cache approach remain
  unimplemented opportunities. Our phase rewrite already covers the basic
  polyphase math; only incremental backend/state/plumbing gains would count.
- Merged sampler and cross-layer positional deduplication were lower priority;
  Raven's reported native results were neutral. Fewer graph nodes is not a
  throughput measurement. No claim is made that these experiments were all
  implemented or measured in this fork.
- Full-model startup hashing was removed from the selected native tokenizer
  path. Historical 1.80→0.27-second startup results describe old hash-on/off
  experiments; the old `--verify-model-hashes` option is not available now.
- Native tokenizer: initial 26,164 English/French reference checks plus later
  bundle-config checks passed; ten CLI pairs matched the old Rust bridge.
  See `native_tokenizer/README.md` and `.cache/tokenizer-cpp/`.
- October English checkpoint: twelve Python/native FP32/INT8 ASR transcripts
  retained 1943 and following words; native FP32/Python waveform SNR was
  51.5/61.8 dB on the temperature-zero cases. French control audio was unchanged.
- Normalization: 299 policy/CSV tests plus CLI chunking/bypass checks. Earlier
  ordinary EN/FR/v1 and bypass controls preserved WAV bytes. Scientific,
  date/time and phone policies need no model re-export. Date/time warm median
  was about 0.40 ms, dense 440-character mixed text 1.29 ms on this Mac;
  Android normalization timing has not been measured.
- Power/energy, sustained thermal behavior, and device phase profiling remain
  open. Improve first-audio latency and throughput only with matching runtime,
  inputs, thread split, warmup and cache policy; confirm perceptual quality by listening.

## Validation tools and evidence

- `export_onnx_v2.py`: pinned checkpoint export, numerical/state checks,
  portable CPU rewrites and linear-only INT8.
- `optimize_onnx_v1.py`: restored legacy CPU and linear-quantization stages.
- `scripts/optimize_accelerate.py`: validated Apple attention/conv bundle rewrite.
- `scripts/test_decoder_bundle.py`: eight dynamic linear kernels, no QLinearConv,
  FP32 and controlled backbone/sampler audio parity (portable bundles).
- `scripts/test_cpp_tokenizer.py`, `scripts/tokenizer_probe.cpp`: native/reference
  tokenizer parity; `scripts/test_voice_cache.py`: named cache invalidation.
- `scripts/test_text_normalization.py`, `scripts/normalizer_probe.cpp`,
  `scripts/benchmark_text_normalization.cpp`: text policy and latency.
- `scripts/test_cli_regression.py`: paired executable audio comparisons;
  `scripts/validate_milestone.py`, `scripts/validate_int8.py`: broader speech/FFI/ASR checks.

Keep `.cache/` reports and original controls for reproducibility. Most older
scripts default to historical control paths; pass model/binary arguments
explicitly when validating a new export. Checkpoint-specific validation and
cleanup are recorded in [CHECKPOINT.md](CHECKPOINT.md).
