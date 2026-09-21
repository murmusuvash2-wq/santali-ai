#!/usr/bin/env python3
"""Build a Kaggle parallel.csv from explicitly approved local artifacts.

The packager never downloads or guesses source rights. An operator must first
record an exact artifact path, license, permissions, and SHA-256 in the source
registry. Lexicons and speech manifests are not silently converted to pairs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import yaml

FIELDS = [
    "record_id", "source_id", "source", "target", "source_lang", "target_lang",
    "script", "transliteration_scheme", "domain", "license", "license_url",
    "attribution", "source_url", "verified", "code_switch", "split",
]
OL_CHIKI = re.compile(r"[\u1C50-\u1C7F]")

def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).strip().split())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/licensed_sources.yaml")
    parser.add_argument("--artifact-root", default=".")
    parser.add_argument("--output-dir", default=".kaggle-dataset")
    args = parser.parse_args()

    root = Path(args.artifact_root)
    out = Path(args.output_dir)
    registry: dict[str, Any] = yaml.safe_load(Path(args.manifest).read_text(encoding="utf-8"))
    candidates = [
        source for source in registry.get("sources", [])
        if source.get("status") == "approved"
        and source.get("kind") == "parallel_text"
        and source.get("training_use") == "allowed"
        and source.get("redistribution") == "allowed"
    ]
    if not candidates:
        raise SystemExit("No approved parallel_text source is registered; refusing to build Kaggle data.")

    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    used_sources: list[dict[str, Any]] = []
    for source in candidates:
        artifact = source.get("artifact_path")
        if not artifact:
            raise SystemExit(f"Approved source {source['id']} has no artifact_path.")
        path = root / str(artifact)
        if not path.exists():
            raise SystemExit(f"Approved artifact is missing: {path}")
        actual_hash = sha256(path)
        if actual_hash != source.get("artifact_sha256"):
            raise SystemExit(f"SHA-256 mismatch for {source['id']}: {actual_hash}")
        used_sources.append(source)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for index, raw in enumerate(reader, start=1):
                source_text = (raw.get("source") or raw.get("src") or raw.get("src_en") or raw.get("english") or raw.get("English") or "").strip()
                target_text = (raw.get("target") or raw.get("tgt") or raw.get("tgt_sat") or raw.get("santali") or raw.get("Santali") or "").strip()
                if not source_text or not target_text or not OL_CHIKI.search(target_text):
                    continue
                source_text, target_text = normalize(source_text), normalize(target_text)
                key = (source_text, target_text)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "record_id": f"{source['id']}:{index}",
                    "source_id": source["id"],
                    "source": source_text,
                    "target": target_text,
                    "source_lang": raw.get("source_lang") or raw.get("src_lang") or "eng_Latn",
                    "target_lang": raw.get("target_lang") or raw.get("tgt_lang") or "sat_Olck",
                    "script": ",".join(source.get("scripts", [])),
                    "transliteration_scheme": raw.get("transliteration_scheme", ""),
                    "domain": raw.get("domain", "mixed"),
                    "license": source["license"],
                    "license_url": source.get("license_url", ""),
                    "source_url": source.get("source_url", ""),
                    "attribution": source.get("attribution", ""),
                    "verified": "rights-approved-source",
                    "code_switch": raw.get("code_switch", "false"),
                    "split": raw.get("split", ""),
                })

    if not rows:
        raise SystemExit("Approved parallel artifacts contained zero usable source/target rows.")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "parallel.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "title": "Santali AI Licensed Parallel Corpus",
        "id": "janaiworkspace/approved-parallel",
        "licenses": [{"name": "CC-BY-SA-4.0"}],
        "description": "Only explicitly approved, hashed parallel sources are included.",
        "rows": len(rows),
        "sources": used_sources,
        "license_note": "See ATTRIBUTION.md and source metadata; do not detach rows from their provenance.",
    }
    (out / "dataset-metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "ATTRIBUTION.md").write_text("# Attribution\n\n" + "\n".join(f"- {s['name']}: {s['attribution']} ({s['license']})" for s in used_sources) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(out / "parallel.csv"), "sources": [s["id"] for s in used_sources]}, indent=2))


if __name__ == "__main__":
    main()
