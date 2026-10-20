#!/usr/bin/env python3
"""CLI integration checks for named caches, timestamp refresh, and precision separation."""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=Path('ptts'))
    parser.add_argument('--bundle', type=Path, default=Path('models-v2-kv/english'))
    parser.add_argument('--voice', required=True, type=Path)
    args = parser.parse_args()
    binary = str(args.binary.resolve())
    source = args.bundle.resolve()
    with tempfile.TemporaryDirectory(prefix='voice-cache-', dir='.cache') as temp:
        root = Path(temp).resolve()
        bundle = root / 'bundle'
        bundle.mkdir()
        manifest = json.loads((source / 'manifest.json').read_text())
        for name in manifest['files']:
            (bundle / name).symlink_to(source / name)
        (bundle / 'manifest.json').write_text(json.dumps(manifest))
        voices = root / 'voices'
        voices.mkdir()
        voice = voices / 'narrator.wav'
        shutil.copy2(args.voice, voice)
        namespace = voices / '.cache/v2' / manifest['language']

        def run(precision='int8', no_cache=False):
            cmd = [binary, '--models-dir', str(bundle), '--voices-dir', str(voices),
                   '--precision', precision, '--temperature', '0', '--seed', '42',
                   '--threads', '1', '3', '--max-chunk', '4', '--verbose']
            if no_cache:
                cmd.append('--no-cache')
            output = root / 'output.wav'
            result = subprocess.run(cmd + ['Hello, this is a cache refresh test.', str(voice), str(output)],
                                    capture_output=True, check=True, timeout=120)
            result.audio = output.read_bytes()
            return result

        first = run()
        emb, kv = [namespace / 'int8' / ('narrator.' + ext) for ext in ['emb', 'kv']]
        assert emb.is_file() and kv.is_file()
        hit = run()
        assert b'Loaded cached embedding:' in hit.stderr and b'Loaded KV cache:' in hit.stderr, hit.stderr.decode()
        assert first.audio == hit.audio == run(no_cache=True).audio
        # Make caches older than the voice, using timestamps rather than sleeping.
        old = voice.stat().st_mtime - 10
        for path in [emb, kv]:
            os.utime(path, (old, old))
        refreshed = run()
        assert b'Loaded cached embedding:' not in refreshed.stderr and b'Loaded KV cache:' not in refreshed.stderr
        assert refreshed.audio == first.audio
        # A replaced bundle also invalidates cached conditioning by timestamp.
        old = (bundle / 'manifest.json').stat().st_mtime - 10
        for path in [emb, kv]:
            os.utime(path, (old, old))
        refreshed = run()
        assert b'Loaded cached embedding:' not in refreshed.stderr and b'Loaded KV cache:' not in refreshed.stderr
        assert refreshed.audio == first.audio
        if 'fp32' in manifest['precision']:
            run('fp32')
            assert (namespace / 'fp32/narrator.emb').is_file()
            assert (namespace / 'fp32/narrator.kv').is_file()
            assert emb.is_file() and kv.is_file()
        print('PASS: named files, cache hit audio, voice/model timestamp refresh, precision separation')


if __name__ == '__main__':
    main()
