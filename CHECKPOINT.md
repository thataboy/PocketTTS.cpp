# Workspace checkpoint — 2026-11-02

Prepared on `v2` after `2ca24a0` (speech normalization). This checkpoint changes
working files only; nothing was staged or committed by the checkpoint task.

## Documentation layout

- [README.md](README.md): this fork's features, platform/model selection, M1
  figures, build, CLI, actual HTTP routes, audio formats, FFI and caches.
- [MODEL_EXPORT.md](MODEL_EXPORT.md): pinned v2 export and attention rewrite;
  restored v1 CPU/linear recipe; Android build, packaging and deployment.
- [EXPERIMENTS.md](EXPERIMENTS.md): consolidated accepted/rejected/deferred
  findings, methods, limitations and evidence from the previous experiment notes.
- [TEXT_NORMALIZATION.md](TEXT_NORMALIZATION.md): detailed speech/lexicon policy;
  retained as the specialist guide.
- `native_tokenizer/README.md`: retained tokenizer implementation/reference notes.
- `docs/benchmarks/`: durable M1 attention summary, exact 250/440-character
  fixtures and SHA256 evidence for the restored v1 model recipe.

Current selections: **v2-attention on Apple silicon**, **v1-kv-linear on Android**.
French attention retains a numeric-year caveat; the portable v2 linear-decoder
bundle is the French control. No new throughput or Android device benchmark is
claimed by this checkpoint.

## Model reproducibility repair

`optimize_onnx_v1.py` and `requirements-optimize-v1.txt` were recovered from the
existing `v1-experimental` stash without applying that stash or switching branches.
The optimizer contains both CPU graph rewrites and the subsequent `--stage linear`
implementation used by `models-v1-kv-linear`. This fills a missing runnable recipe
in the current branch; the retained model data was not modified.

Starting from the original `models` files, stage `all` reproduced the four existing
`models-v1-kv` FP32/INT8 main and decoder graphs byte for byte. Starting from
`models-v1-kv`, stage `linear` reproduced the selected linear decoder byte for byte.
See [SHA256 results](docs/benchmarks/v1-reproduction-2026-11-02.json).

## Recoverable cleanup

Local trash is `.trash/checkpoint-2026-11-02/`, inside the writable workspace;
no system Trash access was required. `.gitignore` excludes this local trash.
Nothing was permanently deleted. `inventory.json` records original paths,
new paths and reasons. The previous README is also saved there as
`README.upstream-before-checkpoint.md`.

Moved files/directories:

| Items | Reason |
| --- | --- |
| `V2.md`, `RAVEN_REVIEW.md`, `DECODER_CONV_QUANTIZATION.md` | Superseded notes consolidated into the three main guides |
| `pocket_tts.cpp`, `pocket-tts` | Upstream single-file runtime and executable, unused by current CMake |
| `libptts.dylib`, `libptts.so` | Stale October 25/24 shared-library outputs; rebuild current source for FFI |
| `out.wav` | Unlabelled old output; labelled research WAVs remain in `.cache` |
| `models-v2-int8-decoder` | Rejected convolution INT8 / hiss experiment |
| `models-v2-neon-attention` | Paused NEON experiment, code preserved on its branch |
| `models-v2-small` | Separate small-model experiment, code preserved on its branch |
| `.build`, `.build-1x`, `.build-android` | Retired build trees, unused by the active build configurations |
| `__pycache__`, `.ruff_cache` | Disposable generated caches |

To restore an item, move it from the dated trash directory to its original path,
first checking that a newer item has not been created at that path. Do not merge
an old build tree into a new one. Git still contains the tracked source/document
versions; ignored binaries and model directories are preserved only locally.

Retained deliberately:

- `models-v2-attention` and `models-v1-kv-linear`, plus their `models-v2`/`models-v1`
  aliases. Attention bundle file-presence checks pass in both languages.
- `models-v2-int8-linear-decoder`: required absolute-symlink source for the active
  attention bundle and the portable control. Removing it would break Mac models.
- `models`, `models-v1-kv`, `models-v2-2026-10`, `models-v2-kv`: recipe sources and
  controls still used by reproduction or validation scripts.
- `.build-v2`, `.build-android-text`, `.build-android-v2`: active build trees;
  Android text builds explicitly reuse dependency source trees in `.build-android-v2`.
- `ptts`, ORT dylibs and `libpocket-custom-ops.dylib`: current runtime and
  Apple model-validation dependencies. The custom-op dylib is not the old runtime library.
- `.cache` experiment reports/audio, current Android binaries/archive, the Python
  environment, voice collection and application data/symlinks.

## Verification

- Restored v1 CPU and linear stages completed numerical/interface/state checks;
  all five reproduced graphs matched existing SHA256 hashes.
- 299 normalization/CSV cases plus English/French CLI chunking/bypass checks pass.
- After cleanup, CLI synthesis succeeds with `models-v2-attention` / narrator
  and `models-v1-kv-linear` / oliver. These are smoke tests, not benchmark results.
- Recommended attention manifests have no missing component files after cleanup.
- CLI flags/routes/formats were checked against `ptts.cpp` and `ptts_engine.cpp`;
  runtime linkage was checked with `otool -L` (ORT 1.25.0 and Accelerate).
- Documentation links and syntax/whitespace were checked. Full checkpoint
  downloads, Apple rewrite regeneration, Android device testing, and new power
  measurements were not repeated for this documentation/cleanup task.

Local proof logs/reproduced models: `.cache/checkpoint-2026-11-02/`.
Historical performance inputs/results are in `docs/benchmarks/`; larger report
and audio collections remain under `.cache` as indexed by `EXPERIMENTS.md`.
