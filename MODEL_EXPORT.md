# Model creation and deployment

Checkpoint: 2026-11-02. The recommended configurations are `models-v2-attention`
for Apple silicon and `models-v1-kv-linear` for Android English. Directory names
are conventions; manifests and graph interfaces select behavior. Do not mix v1
and v2 component files. Every optimizer below writes a **new** directory.
If the documented output already exists, choose a fresh name rather than overwrite it.

## Apple silicon: v2-attention

### 1. Export the portable v2 base

Use Python 3.12 and the pinned upstream source. Keep export environments separate
from any legacy v1 installation. The exporter checks the source Git revision.
Obtain the upstream checkout at `../pocket3` if it is not already available:

```sh
git clone https://github.com/kyutai-labs/pocket-tts.git ../pocket3
git -C ../pocket3 checkout a922b912dfa366975294afb568d815b305acde5d
python3.12 -m venv .venv-export-v2
.venv-export-v2/bin/pip install -r requirements-export-v2.txt
.venv-export-v2/bin/pip install -e ../pocket3
.venv-export-v2/bin/python export_onnx_v2.py --source ../pocket3 \
  --language english --output-dir models-v2-int8-linear-decoder/english
.venv-export-v2/bin/python export_onnx_v2.py --source ../pocket3 \
  --language french --output-dir models-v2-int8-linear-decoder/french
```

The commands may download checkpoint/tokenizer assets and require access to
voice-cloning weights. For an existing checkout, inspect its status before
switching revisions; a separate checkout avoids disturbing other work.
The October English checkpoint is selected by `--language english` at this
revision. It fixes upstream sentence-final number/year truncation. French
weights are unchanged. The current exporter supports these 100M LSD models;
the 18.2M experiment is preserved on `v2-small-model`, not in this exporter.

A fresh export produces FP32 and INT8, split FP32 main KV, valid-prefix
attention, twelve strided main-attention products, and four phase-convolution
decoder rewrites. INT8 covers backbone/sampler dense layers plus the decoder's
eight constant-weight linear operators. All decoder convolutions, encoder,
text conditioner and streaming state stay FP32. No calibration corpus is needed.
Independent component/state and rewrite checks run before publication.

To upgrade an existing validated FP32/split-cache bundle instead of re-exporting:

```sh
.venv-export-v2/bin/python export_onnx_v2.py --source ../pocket3 \
  --quantize-from models-v2-2026-10/english \
  --output-dir models-v2-int8-linear-next/english
```

`--optimize-from BUNDLE` applies the portable CPU rewrites to an unoptimized
split-cache bundle; `--fp32-only` omits INT8. Neither changes checkpoint weights.
Keep the original source for comparisons.

### 2. Add attention only

Build the custom-op validation library against ORT 1.25.0. Native runtime custom
ops are registered inside `ptts`; the separate dylib is used by Python ORT.
The export requirements pin Python ORT 1.24.4 for the portable workflow; upgrade
it in this isolated environment for the custom-op ABI before this step:

```sh
cmake -S . -B .build-v2 -DCMAKE_BUILD_TYPE=Release \
  -DPOCKET_ACCELERATE=ON -DBUILD_SHARED_LIB=ON
cmake --build .build-v2 -j4
.venv-export-v2/bin/pip install 'onnxruntime==1.25.1'
.venv-export-v2/bin/python scripts/optimize_accelerate.py \
  models-v2-int8-linear-decoder/english models-v2-attention/english \
  --components attention --custom-ops-library ./libpocket-custom-ops.dylib
```

This rewrites both precisions' main-transformer graphs and validates prefill,
BOS, state growth and final legal cache writes. The manifest declares
`accelerate_attention_v1`; a portable or Android executable rejects it.
Use `--components attention` explicitly: the helper's default is `conv`, which
is not the accepted configuration. `conv`/`both` remain available for research.

French can be rewritten with the same command and `french` paths, but its
historical numeric-year regression remains unresolved. Prefer the portable
French control until its attention output passes your listening cases.

```sh
./ptts --models-dir models-v2-attention --language english \
  --threads 1 3 --max-chunk 4 --seed 42 'Hello.' narrator output.wav
```

### Files and symlinks

The attention rewriter **symlinks unchanged components to absolute source paths**.
In this workspace that source is `models-v2-int8-linear-decoder`. Both language
bundles depend on it; do not trash or move it while using these outputs.
For a self-contained deployment copy, dereference links:

```sh
mkdir -p deploy/mac
cp -RL models-v2-attention deploy/mac/models-v2-attention
```

Copy the native executable, ORT dylibs and chosen voice samples too. The bundle
contains both precisions; its manifest chooses the correct files. Inspect it
rather than renaming individual component files.

## Android: v1-kv-linear

This keeps the legacy English `b6369a24` checkpoint and SentencePiece tokenizer.
It is not a smaller v2 checkpoint or a conversion of the October model.
The source `models/` directory is the validated legacy export in this workspace.
`optimize_onnx_v1.py` was recovered from the v1 experiment stash for this
checkpoint, including its linear-only quantization stage.

### 1. Start from legacy ONNX files

If `models/` already contains the validated v1 ONNX files and `tokenizer.model`,
skip the PyTorch export. For a fresh legacy export, the locally installed source
was Pocket TTS 1.1.1 at `ef69ab86521d4bedcd1fda861d70d5c05e3a939a`; use a separate
environment because its model interfaces differ from pocket3:

```sh
python3.12 -m venv .venv-export-v1
.venv-export-v1/bin/pip install -r requirements-export-v2.txt
.venv-export-v1/bin/pip install \
  'pocket-tts @ git+https://github.com/kyutai-labs/pocket-tts.git@ef69ab86521d4bedcd1fda861d70d5c05e3a939a'
.venv-export-v1/bin/python export_onnx.py --output-dir models-v1-original
```

This optional command uses the legacy exporter, its `b6369a24` config and the
ungated weight/tokenizer download path encoded in it. A fresh checkpoint download
was not repeated during this documentation checkpoint. The verified reproduction
path starts from the existing validated legacy graphs.

### 2. Optimize attention/decoder, then quantize linear layers

Only NumPy, ONNX and ORT are needed when starting from existing graphs:

```sh
python3.12 -m venv .venv-optimize-v1
.venv-optimize-v1/bin/pip install -r requirements-optimize-v1.txt
.venv-optimize-v1/bin/python optimize_onnx_v1.py \
  --models models --output models-v1-kv --stage all
.venv-optimize-v1/bin/python optimize_onnx_v1.py \
  --models models-v1-kv --output models-v1-kv-linear --stage linear
```

Substitute `models-v1-original` for `models` after the optional fresh export.
The first step rewrites four decoder ConvTranspose operations and twelve main
attention products. It retains legacy split FP16 KV storage, weights, tokenizer
and input/output names. Unlike v2, v1 already had populated-prefix attention.
The second step regenerates `mimi_decoder_int8.onnx` from the optimized FP32
decoder, quantizing eight constant-weight MatMuls per channel. Convolutions stay
FP32. The FP32 decoder remains available under `--precision fp32`.

`--stage all` means the two CPU graph rewrites, **not** linear quantization;
both commands are required for the recommended directory. If `models-v1-kv`
already exists and is validated, run only the second command with a new output
name. Stage/component records and numerical error metrics are written to
`cpu_optimization.json`. Streaming validation includes 57 main calls and
300 decoder latent frames with changing chunks; linear quantization reports
drift rather than claiming byte-identical speech. Listen to new exports.

## Android ARM64 build and deployment

The current CMake imports Android ORT from a sibling directory rather than
fetching it. Supply the **Android ARM64** 1.25.0 library and matching headers:

```text
../onnxruntime-android/
  libonnxruntime.so
  headers/onnxruntime_cxx_api.h
  headers/onnxruntime_c_api.h
  ... other matching ORT headers
```

Example with NDK 27.3.13750724 (adjust the NDK location):

```sh
cmake -S . -B .build-android-release \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/android-ndk/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-30 \
  -DCMAKE_BUILD_TYPE=Release -DPOCKET_ACCELERATE=OFF -DBUILD_SHARED_LIB=ON \
  -DPOCKET_OUTPUT_DIR="$PWD/deploy/android"
cmake --build .build-android-release -j4
mkdir -p deploy/android/models deploy/android/voices
cp -RL models-v1-kv-linear/. deploy/android/models/
cp ../onnxruntime-android/libonnxruntime.so deploy/android/
cp voices/oliver.wav deploy/android/voices/
cp lexicon.english.csv deploy/android/
```

Copy the resulting directory to a writable/executable device location. `libptts.so`
is needed for FFI; standalone CLI needs `ptts`, ORT, models and voices. Ensure any
ORT SONAME required by your supplied library is available on the device. If the
NDK build links shared libc++, include its ARM64 `libc++_shared.so` too.

```sh
cd /device/path/to/deployment
LD_LIBRARY_PATH=. ./ptts --models-dir models --voices-dir voices \
  --threads 1 3 --seed 42 'Hello.' oliver output.wav
```

Tune threads on the actual device and measure sustained runs. Do not substitute
`models-v2-attention`: Apple attention is unavailable on Android. The NEON
experiment is paused on its branch and was not selected for this checkpoint.

Use `cp -RL` or `tar -h` when packaging any model directory containing symlinks.
A prior device archive contained a 29-byte encoder link/path instead of model
contents, causing protobuf parsing failure. Verify unpacked file sizes and
SHA256 hashes against the source. V1 and v2 encoder sizes differ; there is no
universal encoder checksum across these model sets.

## Local directory map

| Directory | Retained role |
| --- | --- |
| `models` | Original v1 export; source for v1 graph reproduction |
| `models-v1-kv` | Optimized v1 before linear-only decoder regeneration; comparison/source |
| `models-v1-kv-linear` | Recommended Android English set |
| `models-v2-attention` | Recommended Mac set; English accepted, French caveat above |
| `models-v2-int8-linear-decoder` | Required attention symlink source and portable French/control |
| `models-v2-2026-10` | Portable FP32-decoder control for decoder parity tests |
| `models-v2-kv` | Earlier portable control still used by tokenizer/cache test defaults |

Other experiments moved aside at this checkpoint are listed in
[CHECKPOINT.md](CHECKPOINT.md). Build directories and `.cache` reports are local
artifacts, not committed model distributions.
