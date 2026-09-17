#!/usr/bin/env python3
"""Download the approved Mod4 corpus and build a Kaggle-ready parallel.csv."""
from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/aiswarya9302/english-santali-datasetmod4/resolve/main"
FILES = ("train.csv", "valid.csv", "test.csv")


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, path)


def convert(raw_dir: Path, output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    seen: set[tuple[str, str]] = set()
    with output.open("w", encoding="utf-8", newline="") as dest:
        writer = csv.DictWriter(
            dest,
            fieldnames=["source", "target", "source_lang", "target_lang", "domain", "license", "verified"],
        )
        writer.writeheader()
        for name in FILES:
            with (raw_dir / name).open(encoding="utf-8-sig", newline="") as src:
                for row in csv.DictReader(src):
                    source = (row.get("src") or "").strip()
                    target = (row.get("tgt") or "").strip()
                    key = (source, target)
                    if not source or not target or key in seen:
                        continue
                    seen.add(key)
                    writer.writerow({
                        "source": source,
                        "target": target,
                        "source_lang": "eng_Latn",
                        "target_lang": "sat_Olck",
                        "domain": "mixed",
                        "license": "unknown",
                        "verified": "approved-source-review",
                    })
                    count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=".kaggle-dataset")
    args = parser.parse_args()
    root = Path(args.output_dir)
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        download(f"{BASE}/{name}?download=true", raw / name)
    count = convert(raw, root / "parallel.csv")
    metadata = {
        "title": "Santali AI Approved Parallel",
        "subtitle": "English to Santali Ol Chiki corpus for private model training.",
        "description": "Private training copy generated from the approved English-Santali Mod4 corpus. Source: https://huggingface.co/datasets/aiswarya9302/english-santali-datasetmod4. The source repository does not publish a clear license; this copy is kept private pending provenance confirmation.",
        "id": "ezqrio/approved-parallel",
        "licenses": [{"name": "unknown"}],
        "userSpecifiedSources": "Source dataset: aiswarya9302/english-santali-datasetmod4; source license is not stated on the dataset card.",
        "resources": [{"path": "parallel.csv", "description": "Deduplicated English to Santali Ol Chiki pairs.", "schema": {"fields": [
            {"name": "source", "description": "English source text", "type": "string"},
            {"name": "target", "description": "Santali Ol Chiki target text", "type": "string"},
            {"name": "source_lang", "description": "Source language code", "type": "string"},
            {"name": "target_lang", "description": "Target language code", "type": "string"},
            {"name": "domain", "description": "Corpus domain label", "type": "string"},
            {"name": "license", "description": "Source license status", "type": "string"},
            {"name": "verified", "description": "Review status", "type": "string"}
        ]}}]
    }
    (root / "dataset-metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": count, "output": str(root / "parallel.csv")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
