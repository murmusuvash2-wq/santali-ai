#!/usr/bin/env python3
"""Build a deterministic licensed Santali seed corpus from approved parallel data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

OL_CHIKI = re.compile(r"[\u1C50-\u1C7F]")


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value or "").strip().split())


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open(encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))

    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    rejected = Counter()
    for index, row in enumerate(raw_rows, start=1):
        source = normalize(row.get("source") or row.get("src") or row.get("english") or row.get("English") or "")
        target = normalize(row.get("target") or row.get("tgt") or row.get("santali") or row.get("Santali") or "")
        if not source or not target:
            rejected["empty"] += 1
            continue
        if not OL_CHIKI.search(target):
            rejected["target_not_olchiki"] += 1
            continue
        pair = (source.casefold(), target.casefold())
        if pair in seen:
            rejected["duplicate_pair"] += 1
            continue
        seen.add(pair)
        record = dict(row)
        record.update({
            "record_id": row.get("record_id") or f"seed-mmloso:{index}",
            "source": source,
            "target": target,
            "source_lang": row.get("source_lang") or "eng_Latn",
            "target_lang": row.get("target_lang") or "sat_Olck",
            "script": "Ol_Chiki",
            "split": "",
        })
        rows.append(record)

    random.Random(args.seed).shuffle(rows)
    n = len(rows)
    boundaries = {"train": int(n * 0.8), "validation": int(n * 0.9)}
    for index, row in enumerate(rows):
        row["split"] = "train" if index < boundaries["train"] else "validation" if index < boundaries["validation"] else "test"

    fields = list(rows[0].keys()) if rows else ["record_id", "source", "target", "split"]
    split_counts: dict[str, int] = {}
    split_hashes: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        output = output_dir / f"{split}.csv"
        split_rows = [row for row in rows if row["split"] == split]
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(split_rows)
        split_counts[split] = len(split_rows)
        split_hashes[split] = digest(output)

    manifest = {
        "input": str(input_path),
        "input_sha256": digest(input_path),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "policy": "approved provenance-preserving Ol Chiki seed; exact pair deduplication; no benchmark rows",
        "rows_read": len(raw_rows),
        "rows_accepted": len(rows),
        "rejected": dict(rejected),
        "split_counts": split_counts,
        "split_sha256": split_hashes,
        "target_script": "Ol_Chiki",
        "quality_review_required": True,
        "notes": [
            "This split is reproducible but not a substitute for native-speaker review.",
            "Keep benchmark/evaluation data outside this input and never merge it into training.",
        ],
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
