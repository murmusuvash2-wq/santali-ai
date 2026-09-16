#!/usr/bin/env python3
import argparse, csv, hashlib, random
from pathlib import Path

def group_key(row):
    return hashlib.sha1((row.get('source','') + '\x00' + row.get('target','')).encode()).hexdigest()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--input', required=True); ap.add_argument('--output-dir', required=True); ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args(); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    with open(args.input, encoding='utf-8-sig', newline='') as f: rows = list(csv.DictReader(f))
    random.Random(args.seed).shuffle(rows)
    n = len(rows); splits = {'train': rows[:int(.8*n)], 'validation': rows[int(.8*n):int(.9*n)], 'test': rows[int(.9*n):]}
    fields = list(rows[0].keys()) if rows else ['source','target']
    for name, data in splits.items():
        with open(out / f'{name}.csv', 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(data)
        print(name, len(data))
if __name__ == '__main__': main()
