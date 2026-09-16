#!/usr/bin/env python3
"""Validate a CSV with source,target and optional provenance columns."""
import argparse, csv, json, re
from pathlib import Path

OL_CHIKI = re.compile(r"[\u1C50-\u1C7F]")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--report', default=None)
    args = ap.parse_args()
    rows, seen, stats = [], set(), {'read': 0, 'kept': 0, 'duplicates': 0, 'empty': 0, 'no_olchiki': 0}
    with open(args.input, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            stats['read'] += 1
            source, target = (r.get('source') or '').strip(), (r.get('target') or '').strip()
            if not source or not target:
                stats['empty'] += 1; continue
            key = (source, target)
            if key in seen:
                stats['duplicates'] += 1; continue
            seen.add(key)
            if r.get('target_lang', 'sat_Olck') == 'sat_Olck' and not OL_CHIKI.search(target):
                stats['no_olchiki'] += 1; continue
            rows.append(r); stats['kept'] += 1
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ['source','target']
    with open(args.output, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    report = {'input': args.input, 'output': args.output, 'stats': stats}
    rp = args.report or str(Path(args.output).with_suffix('.report.json'))
    Path(rp).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
