#!/usr/bin/env python3
"""Prepare approved CSV data for the Kaggle translation run.

Path-safe version: works both locally and on Kaggle after cloning the repo.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def find_scripts_dir() -> Path:
    """Locate the scripts/ directory reliably."""
    # 1. Relative to this file (normal case when repo is cloned)
    candidate = Path(__file__).resolve().parent.parent / "scripts"
    if candidate.is_dir():
        return candidate

    # 2. Fallback: current working directory
    candidate = Path.cwd() / "scripts"
    if candidate.is_dir():
        return candidate

    raise FileNotFoundError(
        f"Could not find 'scripts/' folder.\n"
        f"  __file__ parent: {Path(__file__).resolve().parent}\n"
        f"  cwd: {Path.cwd()}\n"
        "Make sure you cloned the repo and are running from inside it, e.g.:\n"
        "  !git clone https://github.com/murmusuvash2-wq/santali-ai.git\n"
        "  %cd santali-ai"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input parallel CSV")
    ap.add_argument("--output-dir", required=True, help="Directory for processed CSVs")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    scripts = find_scripts_dir()
    print(f"Using scripts from: {scripts}")

    normalized = out / "parallel_normalized.csv"
    validated = out / "parallel_validated.csv"

    subprocess.check_call([
        sys.executable, str(scripts / "normalize_olchiki.py"),
        "--input", args.input,
        "--output", str(normalized),
    ])

    subprocess.check_call([
        sys.executable, str(scripts / "validate_parallel.py"),
        "--input", str(normalized),
        "--output", str(validated),
    ])

    subprocess.check_call([
        sys.executable, str(scripts / "split_parallel.py"),
        "--input", str(validated),
        "--output-dir", str(out),
    ])

    print(f"Prepared data in {out}")
    print("Created files:", sorted(p.name for p in out.glob("*.csv")))


if __name__ == "__main__":
    main()
