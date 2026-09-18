#!/usr/bin/env python3
"""Profile a parallel CSV before it can enter a training or benchmark split."""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

OL_CHIKI = re.compile(r"[\u1C50-\u1C7F]")
BENGALI = re.compile(r"[\u0980-\u09FF]")
DEVANAGARI = re.compile(r"[\u0900-\u097F]")
LATIN = re.compile(r"[A-Za-z]")

def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value or "").strip().split())

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    with Path(args.input).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    seen: set[tuple[str, str]] = set(); duplicate = 0; empty = 0
    target_scripts: Counter[str] = Counter(); source_scripts: Counter[str] = Counter(); lengths = []
    missing_provenance = 0
    for row in rows:
        source, target = normalize(row.get("source") or row.get("src") or ""), normalize(row.get("target") or row.get("tgt") or "")
        if not source or not target: empty += 1; continue
        key = (source, target)
        if key in seen: duplicate += 1
        seen.add(key)
        lengths.append({"source_chars": len(source), "target_chars": len(target), "ratio": len(target) / max(1, len(source))})
        def script(text: str) -> str:
            flags = []
            if OL_CHIKI.search(text): flags.append("Ol_Chiki")
            if BENGALI.search(text): flags.append("Bengali")
            if DEVANAGARI.search(text): flags.append("Devanagari")
            if LATIN.search(text): flags.append("Latin")
            return "+".join(flags) or "Other"
        source_scripts[script(source)] += 1; target_scripts[script(target)] += 1
        required = ("source_id", "license", "source_url", "verified")
        if any(not row.get(field) for field in required): missing_provenance += 1
    ratios = [item["ratio"] for item in lengths]
    report = {
        "input": str(args.input), "rows_read": len(rows), "empty_rows": empty,
        "duplicate_pairs": duplicate, "unique_nonempty_pairs": len(seen),
        "missing_provenance_rows": missing_provenance,
        "source_scripts": dict(source_scripts), "target_scripts": dict(target_scripts),
        "mean_source_chars": sum(item["source_chars"] for item in lengths) / max(1, len(lengths)),
        "mean_target_chars": sum(item["target_chars"] for item in lengths) / max(1, len(lengths)),
        "mean_target_source_ratio": sum(ratios) / max(1, len(ratios)),
        "quality_gate": duplicate == 0 and empty == 0 and missing_provenance == 0,
        "notes": ["Review outliers and language/script mismatches with native speakers before release.", "This report does not prove translation correctness or license validity."],
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
