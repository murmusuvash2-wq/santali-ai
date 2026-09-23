#!/usr/bin/env python3
"""Prepare a traceable train/validation/test experiment from an approved CSV.

This wrapper intentionally stops before model training. It creates clean splits,
a source audit, and a single experiment manifest so every later run can point
to exact data hashes and the commit that produced them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Approved parallel CSV")
    parser.add_argument("--output-dir", required=True, help="Directory for split CSVs and reports")
    parser.add_argument("--manifest", required=True, help="Experiment manifest JSON path")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    split_manifest = output_dir / "split-manifest.json"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "build_seed_corpus.py"),
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--manifest",
            str(split_manifest),
            "--seed",
            str(args.seed),
        ],
        check=True,
    )
    split_data = json.loads(split_manifest.read_text(encoding="utf-8"))
    split_paths = {name: output_dir / f"{name}.csv" for name in ("train", "validation", "test")}
    experiment = {
        "experiment_schema": "santali-ai/experiment/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(input_path),
            "sha256": sha256(input_path),
            "language_pair": ["eng_Latn", "sat_Olck"],
            "target_script": "Ol_Chiki",
        },
        "split": {
            "seed": args.seed,
            "policy": split_data["policy"],
            "counts": split_data["split_counts"],
            "sha256": {name: sha256(path) for name, path in split_paths.items()},
        },
        "model": {
            "base": "ai4bharat/indictrans2-en-indic-dist-200M",
            "method": "LoRA",
            "status": "data-prepared-training-not-run",
        },
        "gates": {
            "rights_audit": "required",
            "unicode_script_audit": "required",
            "tokenizer_audit": "required",
            "numerical_preflight": "required",
            "native_speaker_review": "required",
            "retrieval_safety_review": "not_started",
        },
        "artifacts": {name: str(path) for name, path in split_paths.items()},
        "limitations": [
            "This manifest does not claim that a model has been trained.",
            "Native-speaker quality review and held-out benchmark evaluation remain required.",
        ],
    }
    manifest_path.write_text(json.dumps(experiment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(experiment, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
