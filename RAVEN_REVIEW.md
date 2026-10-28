# Raven optimization review — 2026-10-26

## Follow-up: implemented and measured

The user rebuilt V2 with ORT 1.25.0 and confirmed Raven's lead persisted;
`--keep-commas` did not materially change speed or speaking rate. Treat those
as ruled-out explanations for their observed gap. The binary/version and text
handling findings later in this initial audit are background, not explanations.

The first implemented experiment was ordinary FP32 Accelerate convolutions.
The final eight-convolution version passes English/French FP32/INT8 state and
audio parity, at 118–119 dB audio SNR over 552 latent frames and 72 calls per
precision/language, including wraparound. The input projection stays on ORT:
changing its reduction order perturbed subsequent INT8 rounding enough to fail
the strict decoder-only parity goal. That is distinct from a convolution bug.

At chunk four the eight kernels improved isolated decoder time by roughly 10%,
but did not improve full synthesis at AR1/decoder3. Moving threads to AR also
lost overall throughput. Keep this backend experimental; it is not the best
whole-pipeline configuration on this machine.

The useful change is **fused AR attention**, adapted to our split FP32 caches.
It consumes the existing scatter outputs, bounds reads to the valid prefix,
uses a NEON single-query tail, and fuses scores/mask/softmax/value accumulation.
Key scaling happens on SIMD loads rather than through a full scaled-prefix
copy. Multi-token prefill uses Accelerate. Original cache storage, causal masks
and both original scaling factors are retained.

Native ORT 1.25.0, narrator, INT8, temperature 0.3, seed 42, AR1/decoder3, chunk
four; three order-rotated blocks, five requests per block/variant, first two
discarded (nine retained warm requests per case/variant):

| Fixture | Baseline wall time | Attention wall time | Baseline RTFX | Attention RTFX |
| --- | ---: | ---: | ---: | ---: |
| 250 characters | 721.7 ms | 698.8 ms | 18.37x | 19.40x |
| 440 characters | 1283.6 ms | 1201.0 ms | 18.13x | 19.16x |

Normalized throughput improves 5.6%/5.7%. Median wall time falls 3.2%/6.4%.
Output duration changes from 13.28 to 13.60 seconds and from 23.28 to 23.04
seconds respectively; do not equate these wall-time and throughput percentages.
These are new local fixtures, not the user's original benchmark prompts.
Attention plus convolutions was slightly slower than attention alone (19.19x
and 18.98x), so attention-only is the current experimental recommendation.
Median warm TTFA falls from about 35.3–35.5 ms to 34.5–34.6 ms; resident memory
increases about 3 MiB in these runs.

A separate native AR benchmark emits exactly 96 teacher-forced frames per run,
using identical inputs and histories starting at 128/384/768 tokens. Across
three alternating blocks it measures 7.9%/5.3%/1.9% higher throughput. This
confirms a compute gain independent of generated speech duration.

Both language attention graphs passed 57-call FP32 output/state checks,
including BOS, multi-token/text prefill, cache midpoint and final legal write.
Conditioning SNR is 117.4 dB English and 120.8 dB French. INT8 results are not
bit-identical: later dynamic quantization amplifies small changes. Relative to
the FP32 control, original/candidate INT8 conditioning SNR is 26.15/26.21 dB in
English and 25.52/25.54 dB in French on the teacher-forced cases. This supports
similar numerical fidelity; it does not establish perceptual equivalence.
Python component checks used ORT 1.25.1; native timing used 1.25.0 throughout.

Audio follow-up: large-v3 ASR retained the English long passage and the complete
1943 regression passage in baseline, attention-only and combined outputs. The
long-case WER is 1.25% for all three (ASR writes `travelers` for `travellers`);
the short year case has zero word errors. Free-running FP32 attention speech at
temperature zero matches the control at 73.4 dB SNR with equal sample count.
French short/prose controls also have zero ASR word errors, but the numeric
French year case is already poor in the baseline and worsens with attention
(20% to 46.7% WER). **Recommend the English experiment only; keep the existing
French bundle for normal use pending further evaluation.**

An additional prototype preserving original attention during multi-token
prefill restored the French numeric case to baseline WER but still spoke the
wrong year. It did not resolve that failure and was not adopted into the
rewriter. Its graphs/reports remain under `.cache/raven-study/hot-*`. Numerical
and ASR checks cannot establish absence of hiss or unchanged prosody; listen
to `.cache/raven-study/english-long-{baseline,attention}.wav` before making the
experimental model a personal default.

Implementation: `native_accelerate/`, `scripts/optimize_accelerate.py`, and
`scripts/benchmark_accelerate.cpp`. Build support is opt-in with
`-DPOCKET_ACCELERATE=ON`. The normal portable path remains the default.
Models declare required runtime features, with a clear error when unsupported.
The rewriter validates before publishing and symlinks unchanged components to
the source bundle. Preserve that source bundle when using these local outputs.

Ready local artifacts:

- Binary/shared library/custom-op library: `.cache/raven-study/bin/`.
- Attention-only bundles: `models-v2-accelerate-attention/{english,french}`.
- Convolution-only controls: `models-v2-accelerate-conv/{english,french}`.
- Combined English control: `models-v2-accelerate/english`.
- Raw logs, fixtures, orchestration scripts and JSON reports:
  `.cache/raven-study/` (`final-summary.json`, `main-summary.json`, and each
  bundle's `accelerate_validation.json`).

Regenerate in a Python environment with ONNX and ORT 1.25 or newer:

```sh
cmake -S . -B .build-accelerate -DCMAKE_BUILD_TYPE=Release \
  -DPOCKET_ACCELERATE=ON -DBUILD_SHARED_LIB=ON \
  -DPOCKET_OUTPUT_DIR="$PWD/.cache/accelerate-bin"
cmake --build .build-accelerate -j4
python scripts/optimize_accelerate.py models-v2-int8-linear-decoder/english \
  models-v2-accelerate-new/english --components attention \
  --custom-ops-library .cache/accelerate-bin/libpocket-custom-ops.dylib
```

Run the already-built experiment:

```sh
./.cache/raven-study/bin/ptts --models-dir models-v2-accelerate-attention \
  --language english --threads 1 3 --temperature 0.3 --seed 42 \
  "She was born in New York in 1943. Her father was a Hungarian Jewish immigrant." \
  narrator attention.wav
```

## Initial source audit

Reviewed local PocketTTS.cpp `c413061` and pocket-tts-raven `983c909`, including
the installed ONNX graphs. This is a source/graph audit, not a new performance
benchmark. Raven's published timings below are its measurements on its own
baseline and hardware, not predicted gains over our current V2 bundle.

The best candidates are FP32 Accelerate decoder convolutions, fused AR
attention, and decoder cache updates. Keep our October English checkpoint,
multilingual tokenizer/configuration, and linear-only INT8 policy.

## What is actually different

| Area | Our current V2 | Raven | Assessment |
| --- | --- | --- | --- |
| AR cache | Split FP32, persistent single-buffer allocations, ScatterElements; valid-prefix attention | Packed persistent cache, delta outputs scattered by host; custom attention reads old/new KV separately | Much of the original cache win is already covered here. Do not replace our layout just to match Raven. |
| AR attention | Twelve strided `com.microsoft::FusedMatMul` products, six softmaxes | Six `pockettts::AttentionTail` kernels | Worth testing: removes remaining intermediate attention work, not just transposes. |
| Decoder upsampling | Four ConvTranspose nodes rewritten as phase convolutions | Three large ConvTranspose+overlap blocks fused into Accelerate custom ops; one ordinary ConvTranspose remains | Same polyphase mathematics, but Raven also fuses streaming state/output assembly and changes the compute backend. |
| Ordinary decoder convolution | Nine ordinary FP32 convolutions, plus four phase convolutions | Five AccelConv and four AccelConvElu | Most attractive first experiment for Apple silicon. |
| Decoder linear layers | Eight dynamic INT8 MatMuls | Eight dynamic INT8 MatMuls | Already equivalent optimization coverage. |
| Decoder KV | Combined FP32 cache, regular state output, double-buffered runner | Delta outputs and persistent host scatter | Remaining opportunity in our decoder, especially cache copying/repacking. |
| Flow sampling | Separate main and sampler sessions | Sampler merged into main when LSD steps = 1 | Lower priority: Raven reports neutral native wall-clock impact. |
| Position/mask computation | Repeated per transformer layer | Cross-layer common-subexpression elimination | Lower priority: Raven reports no measurable native gain. |
| Decoder allocation | CPU arena disabled; memory pattern disabled on desktop | Normal arena-enabled decoder session | Cheap controlled experiment, with RSS measurements. |

Inspected raw graph counts: our main has 1,604 nodes and decoder 700; Raven's
merged main has 1,627 and decoder 558. These are not optimized-session node
counts and main totals are not directly comparable because Raven includes the
sampler. Node count alone is not a speed metric.

## Recommended experiments, in order

### 1. Accelerate ordinary convolutions, keeping FP32

Raven's `src/pocket_tts_custom_attention.hpp:800` implements Conv1d as one
Accelerate SGEMM per kernel tap. Offline weights become `[K,Cout,Cin]`; input
submatrices use their existing strides, avoiding im2col. A sole following ELU
can be fused. This does not quantize convolution weights or activations.

Our existing decoder has exactly nine ordinary convolutions matching the
rewriter's stride-one, dilation-one, group-one, zero-padding requirements.
The four phase convolutions have padding, and one is grouped, so they should
be left alone in this first experiment. Raven's generic
`tools/make_decoder_accel_conv_onnx.py` is a useful starting point for this
isolated adaptation. Its stock eligibility checks already skip those four.

Raven reports decoder time falling from 173.5 ms to roughly 137 ms when adding
AccelConv to its already-fused decoder. That is encouraging evidence for the
mechanism, not a transferable percentage for our M1/chunk-four configuration.

Add custom-op registration to the native engine and Python validation path,
link Accelerate on Apple, and select the experimental model through an explicit
manifest capability/variant. Keep the current portable decoder available for
Android and other platforms. Preserve MIT attribution if copying Raven code.

### 2. Fused upsampling and streaming overlap

Raven's `DecoderConvTransposeOverlapKernel` combines polyphase GEMMs, previous
overlap addition, emitted output layout, and next overlap state. Our
`export_onnx_v2.py:640` already uses the same polyphase identity, but expresses
it as Conv/Reshape/Transpose and leaves streaming overlap operations separate.

Adapt the fusion at export time before our phase rewrite, or explicitly match
our rewritten pattern. Raven's existing ConvTranspose pattern matcher cannot
simply be run on a bundle where those nodes have already disappeared. Keep the
small grouped upsampler on the existing path initially.

Raven's reported ~1.44x decoder speedup was over its original ConvTranspose
path, not our phase-convolution baseline. Expect only the incremental benefit
of its backend and fusion, which needs measurement.

### 3. Fused AR attention using our split-cache interface

Raven's attention kernel reads the populated old cache and new K/V separately,
computes scores, mask, softmax and weighted values, and emits the flattened
attention result. The installed source has an explicit ARM NEON specialization
for query length one and head dimension 64; the general/prefill path uses its
BLAS abstraction. It is not simply an AMX attention kernel on every call.

Our strided attention rewrite already removes expensive transposes, so the
remaining possible gains are avoiding intermediate cache/score plumbing,
scaling passes and dispatch. Adapt the kernel to separate K/V pointers rather
than returning to packed caches. Preserve scaling, causal masks, BOS behavior,
multi-token prefill and the valid-prefix bound.

This is also worth investigating for Android because Raven has ARM SIMD
attention code. Its Apple convolution results do not establish Android gains.

### 4. Decoder state and allocation experiments

Our decoder's two attention caches are `[2,1,512,8,64]` (2 MiB each), and its
runner does not enable the main model's single-buffer mode. Raven's decoder
uses a different layout/capacity and host delta scatter with ring wraparound.
Adapt the idea to our time axis and 512-slot capacity; do not copy its offsets.

Compare either split/in-place decoder caches or explicit delta outputs.
Maintain causal sliding-window behavior across multi-frame updates and wrap.
Raven's historical delta-only decoder change saved roughly half a millisecond
on the first call but did not improve aggregate RTFX; this is a secondary
candidate rather than a guaranteed throughput gain.

Separately test decoder arena on/off, then thread splits and chunk sizes.
Our arena setting was chosen for memory retention, so any timing gain needs
long-lived RSS checks across short/long alternating requests. Raven defaults
to max chunk 15 versus our four, but its own larger-batch experiment slowed
the full pipeline. Re-tune after changing kernels, not by copying defaults.

### Lower priority

Merged flow saves one session call per latent and can fold fixed `s=0,t=1`
work. Raven explicitly reports neutral macOS wall-clock results. Integration
must retain the multi-step fallback and avoid changing random-number
consumption at EOS/prefill boundaries. Do not present this as a likely large
native win.

Positional dedup is mathematically valid only for counters known to advance
together. Do not canonicalize every scalar integer state blindly in V2.
Raven reports removing hundreds of nodes without measurable native gain and
warns that ORT per-node profiling overstates tiny-node overhead. Prefer stage
timings without per-node tracing for performance decisions.

Avoid prioritizing norm/GELU fusion, static AR specialization, state-only
prefill, decoder custom attention, dequantizing decoder linear layers, or
queue rewrites: Raven tried these without a useful gain (some were slower).
Its README and optimization journal contain historical defaults superseded
later in the same file; installed code/models are the stronger evidence.

## Interpreting the reported 6–12% advantage

The local binaries are not all linked to the same ORT version:

| Binary | Linked ORT install name |
| --- | --- |
| `PocketTTS.cpp/ptts` | 1.23.2 |
| `PocketTTS.cpp/ptts-v2` | 1.25.0 |
| `pocket-tts-raven/pocket-tts` | 1.25.0 |

These were checked with `otool -L`; record the actual loaded library in the
benchmark too. The Python environment has ORT 1.24.4, another distinction for
component timings. The user's benchmark executable/flags are not established
by this audit, so this is a possible confound, not an explanation of the gap.

The bundles also differ: Raven's preparation pipeline identifies its source as
`english_2026-04` with SentencePiece, while our selected manifest uses the
October checkpoint and JSON tokenizer. Our model default temperature is 0.3;
Raven defaults to 0.7. The same voice name/text therefore does not establish an
equal latent-generation workload.

Raven additionally replaces commas, semicolons and colons with spaces before
preparing text in both generation paths. `--keep-commas` disables all three
replacements despite its name. This is a concrete possible contributor to
reduced pauses and faster pacing, not proof of the cause for the user's samples.
Its streaming decoder also applies 120-sample/5-ms boundary crossfades; emitted
duration can vary with batching. Neither behavior should be copied as a compute
optimization.

Record wall time, raw generated latent frames, decoded samples, delivered WAV
duration, TTFA, and stage time. For example, 10% less generated audio at equal
per-frame compute already gives about 10% less synthesis work. RTFX normalizes
by audio duration but still reflects trimming/crossfades; fixed-frame component
tests are needed to isolate the kernels. The growing advantage on longer text
is consistent with a steady-state advantage, but does not identify its source.

## Validation and acceptance

Use our October bundle as the baseline for each isolated change. Replay fixed
latent sequences through old/new decoders at chunk sizes 1/2/4/8/15, including
resets and enough frames to cross cache wrap. Compare all state outputs and raw
audio before streaming fades. Exercise English and French. For attention, use
identical teacher-forced inputs and compare conditioning, EOS and cache states
across voice/text prefill and growing history before free-running synthesis.

Use numerical tolerances for FP32 backend changes: different reduction order
need not produce bit-identical output. Require listening for hiss/clicks and
prosody, not just ASR or SNR. Our rejected convolution-INT8 experiment already
demonstrated why transcription checks alone are insufficient.

For end-to-end timing, alternate warmed baseline/candidate requests in the same
process configuration using narrator, fixed text and seed, matched ORT and
thread settings. Include the user's 250/440-character prompts when available,
and a longer multi-chunk case. Measure without ORT per-node profiling. Accept
only reproducible gains on our machine with stable audio quality and measured
memory behavior. No runtime, exporter or model changes were made in this audit.
