#!/usr/bin/env python3
"""Import an exact MMLoSo release only after its archive hash is verified.

The script accepts a local CSV/TSV or an extracted directory. It never guesses
which remote Kaggle artifact is authoritative and refuses missing hashes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_table(root: Path) -> Path:
    candidates = sorted(root.rglob("*.csv")) + sorted(root.rglob("*.tsv"))
    if len(candidates) != 1:
        raise SystemExit(f"Expected exactly one MMLoSo table under {root}, found {len(candidates)}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Exact local MMLoSo CSV/TSV or directory")
    parser.add_argument("--source-sha256", required=True, help="SHA-256 recorded in the source ledger")
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default="configs/licensed_sources.yaml")
    args = parser.parse_args()

    source = next((item for item in yaml.safe_load(Path(args.manifest).read_text()).get("sources", []) if item.get("id") == "mmloso-santali-2025"), None)
    if not source:
        raise SystemExit("MMLoSo source is missing from the registry")
    if source.get("license") != "CC-BY-SA-4.0":
        raise SystemExit("MMLoSo license registry entry is not CC-BY-SA-4.0")
    path = Path(args.input)
    table = find_table(path) if path.is_dir() else path
    actual = sha256(table)
    if actual != args.source_sha256:
        raise SystemExit(f"Input hash mismatch: expected {args.source_sha256}, got {actual}")
    delimiter = "\t" if table.suffix.lower() == ".tsv" else ","
    rows: list[dict[str, str]] = []
    with table.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for index, row in enumerate(reader, start=1):
            source_text = (row.get("source") or row.get("src") or row.get("src_en") or row.get("english") or "").strip()
            target_text = (row.get("target") or row.get("tgt") or row.get("tgt_sat") or row.get("santali") or "").strip()
            if not source_text or not target_text:
                continue
            rows.append({
                "record_id": f"mmloso-santali-2025:{index}",
                "source_id": source["id"],
                "source": source_text,
                "target": target_text,
                "source_lang": "eng_Latn",
                "target_lang": "sat_Olck",
                "script": "Ol_Chiki",
                "domain": row.get("domain", "mixed"),
                "license": source["license"],
                "license_url": source["license_url"],
                "attribution": source["attribution"],
                "source_sha256": actual,
                "verified": "release-hash-verified",
            })
    if not rows:
        raise SystemExit("No usable source/target rows found")
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "output": str(out), "source_sha256": actual}, indent=2))

if __name__ == "__main__": main()
