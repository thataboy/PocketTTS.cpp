#!/usr/bin/env python3
"""Validate INT8 speech with a local ASR model, native cache/streaming checks, and benchmarks."""

import argparse
import io
import json
import re
import socket
import subprocess
import sys
import tempfile
import time
import unicodedata
import urllib.request
import wave
from pathlib import Path

import numpy as np
from scipy.io import wavfile

CORPUS = {
    "english": [
        "Hello, world!",
        "Hello, world. This is a test of the new speech model.",
        "Please close the window before you leave. The weather is getting colder, and the children are still playing in the garden. Could you call them inside?",
        "The train arrived at the station early in the morning. A woman stepped onto the platform and looked around for her friend. They had planned to spend the day exploring the city together. After a quick breakfast, they walked beside the river and visited a small bookshop. The owner recommended a story about a young musician who travelled across the country. They bought the book, thanked the owner, and continued their walk. By the evening, they had discovered several quiet streets and a wonderful place to eat.",
    ],
    "french": [
        "Bonjour le monde !",
        "Bonjour le monde. Ceci est un test de synthèse vocale en français.",
        "J'espère que vous allez bien aujourd'hui. Nous avons préparé du café et des croissants pour le petit déjeuner. Voulez-vous vous asseoir près de la fenêtre ?",
        "Le train est arrivé à la gare tôt le matin. Une femme est descendue sur le quai et a cherché son ami. Ils avaient prévu de passer la journée à découvrir la ville ensemble. Après un petit déjeuner, ils ont marché au bord de la rivière et visité une petite librairie. Le propriétaire leur a recommandé une histoire sur un jeune musicien qui voyageait à travers le pays. Ils ont acheté le livre, remercié le propriétaire et continué leur promenade. Le soir, ils avaient découvert plusieurs rues tranquilles et un excellent restaurant.",
    ],
}


def words(text):
    return re.findall(
        r"[^\W_]+", unicodedata.normalize("NFC", text).lower().replace("’", "'")
    )


def edits(reference, actual):
    ref, hyp = words(reference), words(actual)
    row = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        nxt = [i]
        for j, b in enumerate(hyp, 1):
            nxt.append(min(row[j] + 1, nxt[-1] + 1, row[j - 1] + (a != b)))
        row = nxt
    return row[-1], len(ref)


def server_check(binary, bundle, voice, precision, text, output):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with (
        tempfile.TemporaryDirectory(prefix="pocket-http-") as cache,
        (output / f"server-{bundle.name}-{precision}.log").open("w") as log,
    ):
        server = subprocess.Popen(
            [
                binary,
                "--models-dir",
                str(bundle),
                "--precision",
                precision,
                "--temperature",
                "0",
                "--threads",
                "1",
                "1",
                "--port",
                str(port),
                "--max-engines",
                "1",
                "--keep-engines",
                "1",
                "--voices-dir",
                cache,
            ],
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 60
            while True:
                assert server.poll() is None, "Test server exited"
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/health", timeout=1
                    ) as response:
                        assert response.status == 200
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Test server did not become ready")
                    time.sleep(0.1)
            payload = json.dumps({"text": text, "voice": str(voice)}).encode()
            outputs = []
            for _ in range(2):
                request = urllib.request.Request(
                    f"http://127.0.0.1:{port}/synthesize",
                    payload,
                    {"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=90) as response:
                    outputs.append(response.read())
            assert outputs[0] == outputs[1], "HTTP cold/warm output differs"
            with wave.open(io.BytesIO(outputs[0])) as wav:
                assert wav.getframerate() == 24000 and wav.getnframes() > 0
                expected_pcm = wav.readframes(wav.getnframes())
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/stream",
                payload,
                {"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                assert response.read() == expected_pcm, (
                    "Streaming PCM differs from complete WAV"
                )
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    return {
        "language": bundle.name,
        "precision": precision,
        "http_cold_warm_bit_exact": True,
        "stream_matches_synthesize_pcm": True,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundles", type=Path, default=Path("models-v2"))
    ap.add_argument("--binary", type=Path, default=Path("ptts"))
    ap.add_argument("--asr-model", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path(".cache/int8-validation"))
    ap.add_argument(
        "--reuse-audio",
        action="store_true",
        help="Reuse existing WAVs when resuming the same validation run",
    )
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    binary = str(args.binary.resolve())
    records = []
    servers = []
    bench = []
    # Generate all audio first so ASR inference does not distort synthesis timing.
    for language, texts in CORPUS.items():
        bundle = (args.bundles / language).resolve()
        voice = (
            Path(".cache/milestone-validation") / (language + "-voice.wav")
        ).resolve()
        assert voice.exists(), f"Missing prepared voice: {voice}"
        with tempfile.TemporaryDirectory(prefix="pocket-int8-corpus-") as cache:
            for index, text in enumerate(texts):
                durations = []
                for precision in ("fp32", "int8"):
                    audio = args.output / f"{language}-{index}-{precision}.wav"
                    command = [
                        binary,
                        "--models-dir",
                        str(bundle),
                        "--precision",
                        precision,
                        "--temperature",
                        "0" if index == 0 else "0.3",
                        "--seed",
                        "42",
                        "--threads",
                        "1",
                        "1",
                        "--voices-dir",
                        cache,
                        text,
                        str(voice),
                        str(audio),
                    ]
                    if not args.reuse_audio or not audio.exists():
                        result = subprocess.run(
                            command,
                            text=True,
                            capture_output=True,
                            check=False,
                            timeout=240,
                        )
                        (audio.with_suffix(".log")).write_text(
                            result.stdout + result.stderr
                        )
                        assert result.returncode == 0, result.stderr
                    rate, samples = wavfile.read(audio)
                    assert rate == 24000 and samples.size and np.isfinite(samples).all()
                    duration = len(samples) / rate
                    durations.append(duration)
                    records.append(
                        {
                            "language": language,
                            "case": index,
                            "precision": precision,
                            "text": text,
                            "audio": str(audio),
                            "seconds": duration,
                            "peak": float(np.max(np.abs(samples))),
                        }
                    )
                assert 0.65 < durations[1] / durations[0] < 1.5, (
                    language,
                    index,
                    durations,
                )
        # Repeated memory-cache streams and peak RSS run in independent processes.
        for precision in ("fp32", "int8"):
            p = subprocess.run(
                [
                    sys.executable,
                    "scripts/benchmark_int8.py",
                    "--bundle",
                    str(bundle),
                    "--precision",
                    precision,
                    "--voice",
                    str(voice),
                    "--text",
                    texts[1],
                ],
                text=True,
                capture_output=True,
                check=True,
                timeout=180,
            )
            data = json.loads(p.stdout)
            data["language"] = language
            bench.append(data)
            servers.append(
                server_check(binary, bundle, voice, precision, texts[1], args.output)
            )
        print(language, "synthesis, memory-cache, and HTTP checks passed", flush=True)
    from faster_whisper import WhisperModel

    recognizer = WhisperModel(
        str(args.asr_model),
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
        num_workers=1,
        local_files_only=True,
    )
    for record in records:
        segments, _ = recognizer.transcribe(
            record["audio"],
            language="en" if record["language"] == "english" else "fr",
            beam_size=3,
            condition_on_previous_text=False,
        )
        transcript = " ".join(s.text.strip() for s in segments)
        errors, count = edits(record["text"], transcript)
        record.update(
            transcript=transcript,
            word_errors=errors,
            reference_words=count,
            wer=errors / count,
        )
        print(
            record["language"],
            record["case"],
            record["precision"],
            "WER",
            round(errors / count, 4),
            transcript,
            flush=True,
        )
    aggregate = {}
    for language in CORPUS:
        aggregate[language] = {}
        for precision in ("fp32", "int8"):
            items = [
                x
                for x in records
                if x["language"] == language and x["precision"] == precision
            ]
            aggregate[language][precision] = sum(x["word_errors"] for x in items) / sum(
                x["reference_words"] for x in items
            )
    report = {
        "asr_model": str(args.asr_model),
        "records": records,
        "aggregate_wer": aggregate,
        "benchmark": bench,
        "server_checks": servers,
    }
    (args.output / "results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    for language, values in aggregate.items():
        assert values["int8"] <= max(0.05, values["fp32"] + 0.03), (
            language,
            "INT8 WER regressed",
            values,
        )
    print(
        "Passed INT8 quality, streaming, cache, HTTP, and benchmark checks", flush=True
    )


if __name__ == "__main__":
    main()
