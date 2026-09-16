#!/usr/bin/env python3
"""Prepare approved CSV data for the Kaggle translation run."""
import argparse, subprocess, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--input', required=True); ap.add_argument('--output-dir', required=True)
    args = ap.parse_args(); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    normalized = out / 'parallel_normalized.csv'; validated = out / 'parallel_validated.csv'
    subprocess.check_call([sys.executable, 'scripts/normalize_olchiki.py', '--input', args.input, '--output', str(normalized)])
    subprocess.check_call([sys.executable, 'scripts/validate_parallel.py', '--input', str(normalized), '--output', str(validated)])
    subprocess.check_call([sys.executable, 'scripts/split_parallel.py', '--input', str(validated), '--output-dir', str(out)])
    print(f'Prepared data in {out}')
if __name__ == '__main__': main()
