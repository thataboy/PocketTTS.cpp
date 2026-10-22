# C++ JSON tokenizer

`pocket_text.cpp` implements the Unigram/Metaspace profile used by the exported
Pocket TTS English and French tokenizers. It reads their existing `tokenizer.json`
files directly. No model re-export, extra library, Rust toolchain, or runtime
Python is required. The existing C ABI in `pocket_text.h` is unchanged.

The implementation preserves double-precision Viterbi scores and tie ordering,
UTF-8 byte fallback, special-token extraction/skipping, Metaspace decoding, and
the model-specific text preparation/chunking rules. Unsupported tokenizer
pipelines are rejected instead of being interpreted as this profile.
Preparation follows the pinned Python reference, including its whitespace,
digit classification and trailing-punctuation rules; this also corrects a few
rare whitespace/Unicode edge cases in the previous Rust bridge.

Unicode whitespace, digit and uppercase data are committed in `unicode_data.hpp`.
This avoids locale/platform differences on Mac and Android. The header is
generated with Python 3.12 / Unicode 15.0 by
`scripts/generate_tokenizer_unicode.py`; generation is not a build step.

The Unigram algorithm follows Hugging Face tokenizers 0.22.2 (Apache-2.0);
see `NOTICE` and `LICENSE.tokenizers`.

## Validation

From the repository root, using the export/validation Python environment and
the pinned upstream `pocket3` source:

```sh
mkdir -p .cache/tokenizer-cpp
c++ -O3 -std=c++17 scripts/tokenizer_probe.cpp -o .cache/tokenizer-cpp/probe
.venv/bin/python scripts/test_cpp_tokenizer.py
```

Each language has 13,082 checks: 6,044 encoding cases, 5,005 decoding cases,
and 2,033 preparation/chunking cases. Fixtures include every vocabulary entry,
random Unicode text and token sequences, incomplete UTF-8 byte sequences,
all nontrivial Unicode 15 uppercase mappings, special tokens, whitespace,
punctuation, decimal boundaries, alternate preparation settings and long text.
All 26,164 checks pass against Python HF tokenizers and pocket3 preparation.
The same cases also pass with address/undefined-behavior sanitizers enabled.

Mac CLI comparisons against the preserved Rust executable cover legacy INT8
and v2 English/French in FP32 and INT8, with two alternating pairs per case.
All ten pairs produce identical streamed WAV bytes. Raw reports are under
`.cache/tokenizer-cpp/`.

Fresh Mac and Android arm64 CLI/shared-library builds pass without Cargo or a
Rust target configuration. Android device execution is pending. The host CLI
is restored at `ptts`; the Android executable is
`.cache/tokenizer-cpp/ptts-android`.
