#!/usr/bin/env python3
import argparse, csv, unicodedata

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True); ap.add_argument('--output', required=True)
    args = ap.parse_args()
    with open(args.input, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ('source','target'):
            if r.get(k): r[k] = unicodedata.normalize('NFC', r[k]).strip()
    fields = list(rows[0].keys()) if rows else ['source','target']
    with open(args.output, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

if __name__ == '__main__': main()
