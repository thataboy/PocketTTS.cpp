#!/usr/bin/env python3
"""Independent Python/native text, waveform, and cold/warm cache acceptance checks."""

import argparse
import ctypes
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=Path("../pocket3"))
    ap.add_argument("--binary", type=Path, default=Path("./ptts"))
    ap.add_argument("--bundles", type=Path, default=Path("models-v2"))
    ap.add_argument("--english-voice", type=Path, required=True)
    ap.add_argument("--french-voice", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path(".cache/milestone-validation"))
    args = ap.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    from pocket_tts import TTSModel
    from pocket_tts.models.text_chunking import (
        prepare_text_prompt,
        split_into_best_sentences,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    binary = str(args.binary.resolve())
    revision = subprocess.check_output(
        ["git", "-C", str(args.source.resolve()), "rev-parse", "HEAD"], text=True
    ).strip()
    results = []
    fixtures = [
        "Hello, world!",
        "  école : « Bonjour ! » Le café coûte 3.14 euros.  ",
        "L’été, c’est agréable; n’est-ce pas ?",
        "Äpfel, Straße, œuf, cafe\u0301, café.",
        "¿Dónde está? ¡Aquí!",
        '"Hi?", she said. Another sentence...',
        'A quote ends here?"',
        "No punctuation",
        "Ending with a dash—",
        "one\ntwo\rthree\t four",
        "a sentence, " * 80,
    ]
    for language, voice in (
        ("english", args.english_voice),
        ("french", args.french_voice),
    ):
        manifest = json.loads((args.bundles / language / "manifest.json").read_text())
        assert manifest["upstream_revision"] == revision, (
            "Reference source revision differs from bundle"
        )
        config = manifest["config"]
        model = TTSModel.load_model(language=language, temp=0).eval()
        tokenizer = model.flow_lm.conditioner.tokenizer
        options = {
            k: config[k]
            for k in (
                "pad_with_spaces_for_short_inputs",
                "remove_semicolons",
                "append_terminal_punctuation",
                "capitalize_first_letter",
                "replace_characters",
            )
        }
        for text in fixtures:
            chunks = split_into_best_sentences(tokenizer, text, 50, **options)
            expected = []
            for c in chunks:
                prepared, guess = prepare_text_prompt(c, **options)
                expected.append(
                    {
                        "text": prepared,
                        "ids": tokenizer.encode(prepared),
                        "eos_extra": config["model_recommended_frames_after_eos"]
                        or guess + 2,
                    }
                )
            native = json.loads(
                subprocess.check_output(
                    [
                        binary,
                        "--models-dir",
                        str(args.bundles),
                        "--language",
                        language,
                        "--inspect-text",
                        text,
                    ],
                    text=True,
                )
            )
            assert native["chunks"] == expected, (language, text, native, expected)
        results.append(
            {
                "language": language,
                "check": "text_and_token_ids",
                "fixtures": len(fixtures),
                "passed": True,
            }
        )
        rate, samples = wavfile.read(voice)
        if np.issubdtype(samples.dtype, np.integer):
            samples = samples.astype(np.float32) / max(
                abs(np.iinfo(samples.dtype).min), np.iinfo(samples.dtype).max
            )
        samples = samples.astype(np.float32)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        if rate != 24000:
            import math

            gcd = math.gcd(rate, 24000)
            samples = resample_poly(samples, 24000 // gcd, rate // gcd).astype(
                np.float32
            )
        samples = samples[: 30 * 24000]
        voicepath = (args.output / (language + "-voice.wav")).resolve()
        wavfile.write(voicepath, 24000, samples)
        text = (
            "Hello, world. This is a test of the new speech model."
            if language == "english"
            else "Bonjour le monde. Ceci est un test de synthèse vocale en français."
        )
        with torch.no_grad():
            state = model.get_state_for_audio_prompt(torch.from_numpy(samples)[None, :])
            model.max_decoder_frames_per_call = 1
            reference = model.generate_audio(state, text).cpu().numpy().squeeze()
        wavfile.write(args.output / (language + "-python.wav"), 24000, reference)
        with tempfile.TemporaryDirectory(prefix="pocket-cache-") as cachedir:
            outputs = []
            for run in ("cold", "warm", "no-cache", "corrupt"):
                if run == "corrupt":
                    for cache in Path(cachedir).rglob("*.kv"):
                        cache.write_bytes(b"KVC1")
                output = args.output / (language + "-" + run + ".wav")
                command = [
                    binary,
                    "--models-dir",
                    str(args.bundles),
                    "--language",
                    language,
                    "--temperature",
                    "0",
                    "--threads",
                    "1",
                    "1",
                    "--voices-dir",
                    cachedir,
                    "--verbose",
                    text,
                    str(voicepath),
                    str(output),
                ]
                if run == "no-cache":
                    command.insert(1, "--no-cache")
                start = time.monotonic()
                p = subprocess.run(
                    command, check=False, text=True, capture_output=True, timeout=240
                )
                (args.output / (language + "-" + run + ".log")).write_text(
                    p.stdout + p.stderr
                )
                assert p.returncode == 0, p.stdout + p.stderr
                _, native = wavfile.read(output)
                outputs.append(native)
                assert native.shape == reference.shape, (
                    language,
                    run,
                    native.shape,
                    reference.shape,
                )
                error = native - reference
                rmse = float(np.sqrt(np.mean(error**2)))
                maxerr = float(np.max(np.abs(error)))
                snr = float(
                    10 * np.log10(np.mean(reference**2) / max(1e-30, np.mean(error**2)))
                )
                assert snr > 40, (language, run, snr, rmse, maxerr)
                if run == "warm":
                    assert (
                        "Loaded KV cache" in p.stderr
                        and "Loaded cached embedding" in p.stderr
                    ), p.stderr
                results.append(
                    {
                        "language": language,
                        "check": run,
                        "samples": len(native),
                        "seconds": time.monotonic() - start,
                        "rmse": rmse,
                        "max_abs_error": maxerr,
                        "snr_db": snr,
                        "passed": True,
                    }
                )
            assert all(np.array_equal(outputs[0], x) for x in outputs[1:]), (
                language,
                "cold/warm/no-cache/recovered differ",
            )
            results.append(
                {
                    "language": language,
                    "check": "cold_warm_no_cache_recovered_bit_exact",
                    "passed": True,
                }
            )
            lib = ctypes.CDLL(
                str(
                    args.binary.resolve().parent
                    / (
                        "libptts.dylib"
                        if sys.platform == "darwin"
                        else "ptts.dll"
                        if sys.platform == "win32"
                        else "libptts.so"
                    )
                )
            )
            lib.ptt_create.argtypes = [ctypes.c_char_p] * 4 + [
                ctypes.c_float,
                ctypes.c_int,
                ctypes.c_int,
            ]
            lib.ptt_create.restype = ctypes.c_void_p
            lib.ptt_create_bundle.argtypes = [
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_int,
            ]
            lib.ptt_create_bundle.restype = ctypes.c_void_p
            lib.ptt_destroy.argtypes = [ctypes.c_void_p]
            lib.ptt_stream_start.argtypes = [
                ctypes.c_void_p,
                ctypes.c_char_p,
                ctypes.c_char_p,
            ]
            lib.ptt_stream_start.restype = ctypes.c_void_p
            ptr = ctypes.POINTER(ctypes.c_float)
            lib.ptt_stream_read.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ptr),
                ctypes.POINTER(ctypes.c_int),
            ]
            lib.ptt_stream_read.restype = ctypes.c_int
            lib.ptt_stream_end.argtypes = [ctypes.c_void_p]
            lib.ptt_free_audio.argtypes = [ptr]
            bundle = str((args.bundles / language).resolve()).encode()
            handle = lib.ptt_create(bundle, cachedir.encode(), None, b"fp32", 0.0, 1, 2)
            assert handle, "FFI creation failed"
            try:
                ffi = []
                for _ in range(2):
                    stream = lib.ptt_stream_start(
                        handle, text.encode(), str(voicepath).encode()
                    )
                    assert stream
                    chunks = []
                    try:
                        while True:
                            data = ptr()
                            count = ctypes.c_int()
                            status = lib.ptt_stream_read(
                                stream, ctypes.byref(data), ctypes.byref(count)
                            )
                            assert status >= 0
                            if status == 0:
                                break
                            chunks.append(
                                np.ctypeslib.as_array(data, shape=(count.value,)).copy()
                            )
                            lib.ptt_free_audio(data)
                    finally:
                        lib.ptt_stream_end(stream)
                    ffi.append(np.concatenate(chunks))
                assert np.array_equal(*ffi), "FFI memory-cache mismatch"
                assert np.array_equal(ffi[0], outputs[0]), "CLI/FFI mismatch"
            finally:
                lib.ptt_destroy(handle)
            handle = lib.ptt_create_bundle(bundle, cachedir.encode(), 2)
            assert handle, "Bundle-default FFI creation failed"
            lib.ptt_destroy(handle)
            results.append(
                {
                    "language": language,
                    "check": "ffi_cli_memory_cache_bit_exact",
                    "passed": True,
                }
            )
        print(language, "passed", flush=True)
        del model
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
