# PocketTTS.cpp — CPU inference fork

This fork runs Kyutai Pocket TTS locally through ONNX Runtime, with streaming,
voice cloning, a CLI, an HTTP server with the iTalk reader, and a C shared-library
API. The current `v2` branch supports English/French v2 bundles and legacy English
v1 models. Python is needed to export models, not to synthesize speech.

The runtime is `ptts.cpp` plus `ptts_engine.cpp`, with native tokenizer,
normalization and optional Apple attention kernels. It has diverged from the
upstream single-file program: the executable, server endpoints, audio formats
and command-line options below describe **this fork**.

## Recommended models

| Platform | Model directory | Configuration |
| --- | --- | --- |
| Apple silicon Mac | `models-v2-attention/english` | October English checkpoint, split FP32 KV, fused Apple attention, phase-convolution decoder, INT8 dense layers |
| Android ARM64 | `models-v1-kv-linear` | Legacy English weights, split FP16 KV, strided attention, phase convolutions, linear-only INT8 decoder |
| Portable v2 / French control | `models-v2-int8-linear-decoder/<language>` | Same portable v2 optimizations, without Apple custom attention |

The local aliases `models-v2` and `models-v1` point to the first two model sets.
Use explicit directories when deploying. Android's recommended v1 model is
English-only; use a portable v2 French bundle when French is required.
The French attention bundle exists, but a historical numeric-year speech test
regressed; it needs separate listening validation before replacing the French control.

[Model export and deployment](MODEL_EXPORT.md) gives reproducible commands for
both recommended models. [Experiment findings](EXPERIMENTS.md) records accepted,
rejected and deferred work, including the Android comparison and benchmark caveats.

## What this fork adds

- Pipelined autoregressive generation and Mimi decoding, separate thread budgets,
  configurable first/subsequent decoder chunks, and deterministic seeds.
- Valid-prefix attention and persistent split KV buffers; strided attention
  products and polyphase decoder convolutions avoid repeated layout work.
- Dynamic per-channel INT8 dense layers in backbone, flow sampler and decoder.
  Decoder convolutions remain FP32: convolution INT8 caused audible hiss.
- Optional Apple silicon fused attention. The Android NEON experiment is saved
  on `codex/v2-neon-attention`, rather than enabled in this branch.
- Native C++ JSON tokenization for v2, SentencePiece for v1; no Rust runtime.
- English/French speech normalization: numbers, currency, units, scientific
  notation, dates/times, phone grouping and configurable pronunciation lexicons.
- Voice embedding/KV disk caches plus resident reuse, cache refresh controls,
  text inspection and operation profiling.
- Concurrent HTTP engine pool and bundled iTalk browser UI for reading text,
  favorites and session history.

## M1 Mac performance

These are recorded **CPU-only warm in-process measurements**, not new checkpoint
benchmarks. Native ORT 1.25.0, narrator, INT8, temperature 0.3, seed 42, AR1 /
decoder3, first chunk one, maximum chunk four. Nine warm requests per variant
and text were retained across three order-rotated blocks.

| Text | Portable v2 RTFx | Attention-only RTFx | Median generation time, portable → attention | Warm first audio, portable → attention |
| --- | ---: | ---: | ---: | ---: |
| 250 characters | 18.37× | **19.40×** | 722 → 699 ms | 35.3 → 34.5 ms |
| 440 characters | 18.13× | **19.16×** | 1284 → 1201 ms | 35.5 → 34.6 ms |

RTFx is generated audio seconds divided by synthesis seconds; higher is better.
Model loading and cold voice preparation are excluded. Fused attention changes
INT8 numerical trajectories and sometimes audio duration, even with the same
text/seed: the throughput gains are 5.6–5.7%, while wall-time gains differ.
These tests predate text normalization. A separate fixed-96-frame attention test
confirmed a compute gain independently of output duration. RSS was about
746–750 MiB for the attention variant in these runs. CPU utilization, power and
energy per generated second have **not** been measured.

For comparison, older cold-process CLI tests with portable v2 INT8 measured
about 8.6×, including cold voice preparation in synthesis and first-audio timing.
Do not compare that figure directly with warm server throughput or the old
upstream README's Ryzen figures. Fixtures, results and reproduction steps are
in [the experiment record](EXPERIMENTS.md#m1-measurements).

## Build

Use a C++17 compiler and a recent CMake with FetchContent support (3.28+ is a
practical baseline). Native builds fetch ORT 1.25.0, SentencePiece 0.2.1 and
dr_libs. Speech-normalization rules are vendored and embedded in the binary.
The current macOS download targets Apple silicon; Intel Mac is not configured.

```sh
cmake -S . -B .build-v2 -DCMAKE_BUILD_TYPE=Release \
  -DPOCKET_ACCELERATE=ON -DBUILD_SHARED_LIB=ON
cmake --build .build-v2 -j4
```

Outputs default to the repository root: `ptts`, `libptts.dylib` on macOS
(`libptts.so` on Android/Linux), the ORT library, and, with Accelerate enabled,
`libpocket-custom-ops.dylib` for Python model-rewrite validation.
Use `-DPOCKET_OUTPUT_DIR=/absolute/output/directory` to isolate a build.

For Linux, Windows, or portable builds, explicitly set `-DPOCKET_ACCELERATE=OFF`.
The current CMake default is ON for non-Android targets and only works on Apple
silicon. This checkpoint was validated on Mac and Android ARM64; other platforms
have CMake paths but were not revalidated here. CMake caches option values:
changing a source default does not override an existing build cache.

For an already-populated offline build, use
`-DFETCHCONTENT_UPDATES_DISCONNECTED=ON` to avoid dependency update checks.
See [Android build/deployment](MODEL_EXPORT.md#android-arm64-build-and-deployment)
for the NDK and external ORT layout. Do not enable fast-math: NaN sentinels are
part of model-default and BOS handling.

## Generate speech

Place voice samples in `voices/` (the local workspace uses a symlink to the
voice collection). Voice names resolve to WAV samples there; explicit audio
paths can be used for cloning. Output is mono 24 kHz.

```sh
./ptts --models-dir models-v2-attention --language english 'Hello, world.' narrator output.wav

./ptts --models-dir models-v1-kv-linear 'Hello, world.' oliver output.wav

./ptts --models-dir models-v2-int8-linear-decoder --language french \
  'Bonjour le monde.' narrator bonjour.wav
```

A direct v2 language directory also works. INT8 is the default; `--precision fp32`
selects the FP32 variant. V2 temperature comes from the bundle (normally 0.3);
v1 defaults to 0.7. A fixed seed is reproducible within a configuration, not
across different checkpoints, precisions or attention backends.

Omit OUTPUT to stream **16-bit little-endian PCM in a WAV container** to stdout.
The header uses a streaming length; diagnostics go to stderr. There is no
`--stdout` flag and stdout is not raw float PCM.

```sh
./ptts --models-dir models-v2-attention 'Hello.' narrator | ffplay -nodisp -autoexit -i -
```

## Server and iTalk

No positional arguments starts the server.

```sh
./ptts --models-dir models-v2-attention
```

Open `http://localhost:9000/italk` (or `/`) for the bundled reader. Run from the
repository root so `static/italk` is found. iTalk stores settings, favorites and
session history in `~/italk.json`; the local `static/books` symlink exposes the
book collection through `/books`. The server binds all interfaces and has no
authentication; use it on a trusted network or behind an authenticated proxy.
Each engine has its own model memory; choose the pool size to suit available RAM.

| Endpoint | Behavior |
| --- | --- |
| `POST /synthesize` | JSON `text`, optional `voice` (default `narrator`); complete WAV |
| `POST /stream` | Same JSON; chunked raw signed-16-bit PCM, mono 24 kHz |
| `GET /voices` | JSON list of voice names |
| `GET /voices/refresh` | Rescan voices and clear idle engine voice caches |
| `GET /health` | Health response |
| `GET /italk`, `/` | Reader UI; its state API supports settings, tags, favorites and sessions |

```sh
curl http://localhost:9000/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello world!","voice":"narrator"}' --output speech.wav
```

This server does not implement upstream's `/tts` or `/v1/audio/speech` routes.

## Text normalization and caches

Normalization runs once before sentence splitting/tokenization, in CLI, server
and FFI paths for both model generations. Default optional lexicons are
`lexicon.english.csv` and `lexicon.french.csv` in the working directory.
Literal/regex rules run in CSV order; the first matching rule wins. Prefixes
`/i`, `/r`, `/R` select case-insensitive literals, sensitive regex, insensitive
regex; `//` escapes a literal leading slash. See [the normalization guide](TEXT_NORMALIZATION.md)
for boundary rules, capture substitutions, language limits and examples.

```sh
./ptts --models-dir models-v2-attention --inspect-text \
  'Meet Mon. Jan. 1, 1980 at 11:06 PM. Call (310) 799-1029.'
```

`--inspect-text` prints normalized text, prepared chunks and token IDs without
loading the synthesis sessions; it currently requires a v2 bundle.
Use `--no-lexicon` to disable CSVs or `--no-normalize-text` for the old text path.

Voice embeddings (`.emb`) and conditioning snapshots (`.kv`) live under
`voices/.cache/`; v2 uses readable `v2/<language>/<precision>/<voice>` paths.
Freshness uses timestamps, not content hashes. Keep distinct voice filenames;
after replacing models, refresh caches explicitly. `--no-cache` disables disk
caching, while warm resident reuse still applies. `--refresh-cache` regenerates
stale caches. There is no `--verify-model-hashes` option in the current CLI.

## Command-line reference

Run `./ptts --help` for the executable's current list. All options below are
parsed by this fork; AR/decoder thread counts must both be supplied.

| Option | Default / meaning |
| --- | --- |
| `--models-dir PATH` | `models`; v1 directory, v2 root, or direct v2 language bundle |
| `--language NAME` | `english`; child bundle when using a v2 root |
| `--voices-dir PATH` | `voices` |
| `--tokenizer PATH` | V1 override; otherwise `<models-dir>/tokenizer.model`; v2 owns its tokenizer |
| `--precision int8\|fp32` | `int8` |
| `--temperature FLOAT` | Bundle recommendation; 0.7 for v1 |
| `--lsd-steps INT` | `1`; sampler ODE steps |
| `--threads AR DECODER` | `0 0`: automatic split of a capped half-core budget, favoring decoder |
| `--first-chunk INT` | `1` latent frame |
| `--max-chunk INT` | `4` frames; v2 supports 1–16 |
| `--seed UINT` | `0`: seed from clock |
| `--eos-threshold FLOAT` | `-4.0` |
| `--eos-extra INT` | `-1`: automatic extra frames |
| `--noise-clamp FLOAT` | `0`: disabled |
| `--lexicon PATH` | Explicit CSV instead of automatic language CSV |
| `--no-lexicon` | Disable CSV substitutions |
| `--no-normalize-text` | Disable normalization and CSV substitutions |
| `--inspect-text` | V2 text/chunk/token inspection |
| `--no-cache` | Disable disk voice caches |
| `--refresh-cache` | Refresh stale caches |
| `--port INT` | `9000` |
| `--max-engines INT` | `3`: concurrent server engines |
| `--keep-engines INT` | `1`: retained idle engines |
| `--verbose`, `--trace` | Diagnostics; disabled for stdout streaming |
| `--profile` | Operation timing; disabled for stdout streaming |
| `-h`, `--help` | Help |

## Shared library and validation

Build with `-DBUILD_SHARED_LIB=ON`. The C ABI in `ptts_engine.cpp` exposes
`ptt_create`, `ptt_warmup`, `ptt_set_text_options`, `ptt_stream_start`,
`ptt_stream_read`, `ptt_stream_end`, `ptt_free_audio`, and `ptt_destroy`.
Use a direct language bundle path for v2. `ptt_create_bundle` is an older
convenience entry point that explicitly selects FP32; use `ptt_create` with
`"int8"` to match the normal INT8 CLI. Stream buffers contain float samples;
free them with `ptt_free_audio`. End the stream before destroying its engine.
Text options must be set before synthesis, not concurrently with it.

```sh
cmake --build .build-v2 -j4 --target normalizer-probe benchmark-text-normalization
.venv/bin/python scripts/test_text_normalization.py --probe ./normalizer-probe --cli ./ptts
./benchmark-text-normalization 1000
```

Additional checks cover tokenizer parity, decoder quantization, voice cache
invalidation, CLI regression and FP32/INT8 speech. See [EXPERIMENTS.md](EXPERIMENTS.md)
for the appropriate scripts and historical results.

## License and attribution

MIT; see [LICENSE](LICENSE). Model weights and voice recordings retain their
own licenses and access conditions. The original C++ runtime and Kyutai model
work remain credited in source/history. Apple kernels include adaptations from
PocketTTS-RAVEN; see [its license](native_accelerate/RAVEN_LICENSE).
The native Unigram tokenizer follows Hugging Face tokenizers (Apache-2.0;
[native tokenizer notice](native_tokenizer/NOTICE)). Numbertext's embedded rules
and Soros interpreter use the BSD option; see [COPYING](native_text/numbertext/COPYING).
