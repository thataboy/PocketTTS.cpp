#!/usr/bin/env python3
"""Alternate two Mac CLI builds, checking audio identity and reporting timings."""
import argparse
import hashlib
import json
import re
import statistics
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--candidate', type=Path, default=Path('ptts'))
    parser.add_argument('--bundles', type=Path, default=Path('models-v2-kv'))
    parser.add_argument('--legacy', type=Path, default=Path('models'))
    parser.add_argument('--voice', required=True, type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--case', help='Run only this case name')
    parser.add_argument('--output', type=Path, default=Path('.cache/cli-regression.json'))
    args = parser.parse_args()
    english = 'The morning sun lit the quiet street. A narrator paused before telling the next part of the story.'
    french = 'Le soleil du matin éclairait la rue tranquille. Le narrateur a marqué une pause avant de poursuivre son histoire.'
    cases = [('v1-int8', args.legacy, 'int8', english),
             ('english-int8', args.bundles / 'english', 'int8', english),
             ('french-int8', args.bundles / 'french', 'int8', french),
             ('english-fp32', args.bundles / 'english', 'fp32', 'Hello, this is a short speech test.'),
             ('french-fp32', args.bundles / 'french', 'fp32', 'Bonjour, ceci est un court test de synthèse vocale.')]
    if args.case:
        cases = [c for c in cases if c[0] == args.case]
        if not cases:
            parser.error('Unknown case')
    report = {'runs': [], 'summary': {}}
    for name, models, precision, text in cases:
        for repeat in range(args.repeats):
            pair = []
            order = [('baseline', args.baseline), ('candidate', args.candidate)]
            if repeat % 2:
                order.reverse()
            for label, binary in order:
                command = [str(binary.resolve()), '--models-dir', str(models), '--precision', precision,
                           '--threads', '1', '3', '--temperature', '0', '--seed', '42',
                           '--first-chunk', '1', '--max-chunk', '4', '--no-cache', text, str(args.voice.resolve())]
                start = time.perf_counter()
                p = subprocess.run(command, capture_output=True, check=True, timeout=120)
                wall = time.perf_counter() - start
                log = p.stderr.decode()
                timing = re.search(r'([\d.]+)s audio in ([\d.]+)s \(RTFx: ([\d.]+)x\)', log)
                first = re.search(r'First chunk latency: ([\d.]+)ms', log)
                if not timing or not first:
                    raise RuntimeError(log)
                entry = dict(case=name, repeat=repeat, variant=label, wall_s=wall,
                             audio_s=float(timing[1]), generation_s=float(timing[2]),
                             rtfx=float(timing[3]), ttfa_ms=float(first[1]),
                             sha256=hashlib.sha256(p.stdout).hexdigest())
                pair.append(entry)
                report['runs'].append(entry)
            assert pair[0]['sha256'] == pair[1]['sha256'], f'Audio changed for {name}'
        summary = {}
        for label in ['baseline', 'candidate']:
            runs = [r for r in report['runs'] if r['case'] == name and r['variant'] == label]
            summary[label] = {key: statistics.median(r[key] for r in runs)
                              for key in ['wall_s', 'rtfx', 'ttfa_ms']}
        report['summary'][name] = summary
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(name, json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
