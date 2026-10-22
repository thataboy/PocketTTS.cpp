#!/usr/bin/env python3
"""Compare the C++ tokenizer with HF tokenizers and pinned pocket3 preparation.

The probe is a standalone C++ executable (no ONNX Runtime/SentencePiece needed):
  c++ -O3 -std=c++17 scripts/tokenizer_probe.cpp -o .cache/tokenizer-cpp/probe
"""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, default=Path('.cache/tokenizer-cpp/probe'))
    parser.add_argument('--bundles', type=Path, default=Path('models-v2-kv'))
    parser.add_argument('--source', type=Path, default=Path('../pocket3'))
    parser.add_argument('--output', type=Path, default=Path('.cache/tokenizer-cpp/parity.json'))
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    from pocket_tts.models.text_chunking import prepare_text_prompt, split_into_best_sentences
    from pocket_tts.modules.text_conditioner import JsonTokenizer

    fixtures = [
        '', ' ', '\n\t', 'Hello, world!', 'école à Noël; café, œuf et garçon.',
        'cafe\u0301 café', 'ßoup', 'ﬃeld', 'ǅuro', 'İstanbul', 'ςίγμα',
        '中文 日本語 한글 عربى हिन्दी 😀😺', '<s>Hello</s>', 'a<unk>b<pad>',
        '<0xE2><0x82><0xAC>', '▁ hello▁world', 'a\t b\r\nc', 'a  b   c    d',
        'It costs 3.14 euros; version 2.0 is ready. Next sentence!',
        'He was born in 1943 (his father, an immigrant, invented a knife).',
        'She was born in New York in 1943. Her father was a Hungarian Jewish immigrant.',
        'The poet’s father said, ‘She was born in 1943.’',
        '"Hi?", she said. Another sentence...', 'No punctuation',
        'Ending with a dash—', 'Ending with a dash — ”', "Final comma,')",
        'The number ².³ is unusual. The fraction ½.¼ is unusual.',
        'A sentence, ' * 80, 'a' * 1200, '.!?…;,:()[]“”’—',
    ]
    fixtures += [f'a{chr(c)}b' for c in [9, 10, 13, 28, 29, 30, 31, 0x85, 0xA0,
                                        0x1680, 0x2003, 0x2028, 0x2029, 0x202F, 0x3000]]
    # Exercise every nontrivial Unicode uppercase mapping in text preparation.
    casing = [chr(c) + ' test.' for c in range(0x110000)
              if not chr(c).isupper() and chr(c).upper() != chr(c)]
    rng = random.Random(42)
    alphabet = list(' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,;:!?()[]\t\r\n')
    alphabet += list('éçàùœßİıΣςπ中文日本語😀😺…“”’▁')
    random_text = [''.join(rng.choices(alphabet, k=rng.randint(1, 200))) for _ in range(2000)]
    summary = {}
    for language in ['english', 'french']:
        directory = args.bundles / language
        data = json.loads((directory / 'tokenizer.json').read_text())
        cfg = json.loads((directory / 'manifest.json').read_text())['config']
        tokenizer = Tokenizer.from_file(str(directory / 'tokenizer.json'))
        reference = JsonTokenizer(len(data['model']['vocab']), str((directory / 'tokenizer.json').resolve()))
        requests, expected = [], []

        def add(request, result):
            requests.append(request)
            expected.append(result)

        texts = fixtures + random_text + [v[0] for v in data['model']['vocab']]
        for text in texts:
            add({'action': 'encode', 'text': text}, tokenizer.encode(text).ids)
        id_lists = [[i] for i in range(len(data['model']['vocab']))]
        id_lists += [list(range(4000)), [0, 1, 2, 3], [99999, -1], [230, 132, 160], [4 + 0xFF]]
        id_lists += [[rng.randrange(4000) for _ in range(rng.randint(1, 80))] for _ in range(1000)]
        for ids in id_lists:
            # HF IDs are unsigned; the native decoder also ignores unknown IDs.
            add({'action': 'decode', 'ids': ids}, tokenizer.decode([i for i in ids if i >= 0], skip_special_tokens=True))
        process_cases = [(text, cfg) for text in fixtures + random_text[:200] + casing]
        for options in [dict(pad_with_spaces_for_short_inputs=True),
                        dict(append_terminal_punctuation=False, capitalize_first_letter=False),
                        dict(remove_semicolons=True), dict(max_tokens=7), dict(max_tokens=100),
                        dict(model_recommended_frames_after_eos=8)]:
            process_cases += [(text, dict(cfg, **options)) for text in fixtures]
        for text, config in process_cases:
            options = {k: config[k] for k in ['pad_with_spaces_for_short_inputs', 'remove_semicolons',
                'append_terminal_punctuation', 'capitalize_first_letter', 'replace_characters']}
            try:
                chunks = []
                for chunk in split_into_best_sentences(reference, text, config.get('max_tokens', 50), **options):
                    prepared, guess = prepare_text_prompt(chunk, **options)
                    chunks.append(dict(text=prepared, ids=reference.encode(prepared),
                                       eos_extra=config['model_recommended_frames_after_eos'] or guess + 2))
                value = {'chunks': chunks}
            except ValueError:
                value = {'error': True}
            add({'action': 'process', 'text': text, 'config': config}, value)
        p = subprocess.run([str(args.probe.resolve()), str(directory / 'tokenizer.json')],
                           input='\n'.join(json.dumps(r, ensure_ascii=False) for r in requests) + '\n',
                           capture_output=True, text=True, check=True)
        # U+0085/U+2028 can appear inside JSON strings; split only on LF.
        actual = [json.loads(line) for line in p.stdout.split('\n') if line]
        assert len(actual) == len(expected)
        failures = []
        for req, ref, got in zip(requests, expected, actual):
            if isinstance(ref, dict) and 'error' in ref:
                equal = isinstance(got, dict) and 'error' in got
            else:
                equal = ref == got
            if not equal:
                failures.append(dict(request=req, expected=ref, actual=got))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        (args.output.parent / (language + '-failures.json')).write_text(json.dumps(failures[:20], ensure_ascii=False, indent=2))
        summary[language] = {'checks': len(requests), 'failures': len(failures),
                             'encode': len(texts), 'decode': len(id_lists), 'process': len(process_cases)}
        print(language, summary[language], flush=True)
    args.output.write_text(json.dumps(summary, indent=2) + '\n')
    assert all(not r['failures'] for r in summary.values()), args.output


if __name__ == '__main__':
    main()
