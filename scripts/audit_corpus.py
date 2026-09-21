#!/usr/bin/env python3
"""Audit a source registry and emit a provenance manifest for Kaggle packaging.

This tool deliberately does not download data. Downloads are allowed only after
an operator records a verified license, exact artifact URL and SHA-256 hash.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REQUIRED_POLICY = (
    "require_license",
    "require_training_permission",
    "require_redistribution_permission",
    "require_source_url",
    "require_file_hash",
    "exclude_unknown_license",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/licensed_sources.yaml")
    parser.add_argument("--output-dir", default=".corpus-audit")
    parser.add_argument("--artifact-dir", default=None, help="Optional local artifact tree to hash")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    document: dict[str, Any] = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    policy = document.get("policy", {})
    missing_policy = [key for key in REQUIRED_POLICY if not policy.get(key)]
    if missing_policy:
        raise SystemExit(f"Missing required policy gates: {', '.join(missing_policy)}")

    sources = document.get("sources", [])
    artifact_root = Path(args.artifact_dir) if args.artifact_dir else None
    rows: list[dict[str, Any]] = []
    approved = 0
    excluded = 0
    for source in sources:
        license_name = str(source.get("license", "")).strip().lower()
        status = str(source.get("status", "")).strip()
        eligible = (
            status in {"approved", "approved_private"}
            and license_name not in {"", "unknown", "proprietary"}
            and source.get("training_use") == "allowed"
            and source.get("redistribution") in {"allowed", "allowed_private_only"}
            and bool(source.get("source_url"))
            and bool(source.get("artifact_sha256"))
        )
        artifact_path = ""
        artifact_hash = str(source.get("artifact_sha256", ""))
        if artifact_root and source.get("artifact_path"):
            local = artifact_root / str(source["artifact_path"])
            if local.exists():
                artifact_path = str(local)
                artifact_hash = sha256(local)
                if source.get("artifact_sha256") and artifact_hash != source["artifact_sha256"]:
                    eligible = False
        if eligible:
            approved += 1
        else:
            excluded += 1
        rows.append({
            "id": source.get("id", ""),
            "name": source.get("name", ""),
            "kind": source.get("kind", ""),
            "status": status,
            "eligible_for_kaggle_training": str(eligible).lower(),
            "license": source.get("license", ""),
            "license_url": source.get("license_url", ""),
            "source_url": source.get("source_url", ""),
            "attribution": source.get("attribution", ""),
            "artifact_path": artifact_path or source.get("artifact_path", ""),
            "artifact_sha256": artifact_hash,
            "use_class": source.get("use_class", ""),
            "training_use": source.get("training_use", ""),
            "redistribution": source.get("redistribution", ""),
            "languages": ",".join(source.get("languages", [])),
            "scripts": ",".join(source.get("scripts", [])),
            "notes": source.get("notes", ""),
        })

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["id", "name"]
    with (out / "sources.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "title": "Santali AI rights-aware corpus manifest",
        "version": document.get("version", 1),
        "policy": policy,
        "approved_source_count": approved,
        "excluded_source_count": excluded,
        "sources": rows,
        "training_note": "Only rows marked eligible_for_kaggle_training=true may enter the training dataset. Unknown, evaluation-only, or restricted sources remain excluded.",
    }
    (out / "dataset-metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "ATTRIBUTION.md").write_text(
        "# Attribution and provenance\n\n"
        "This file is generated from `configs/licensed_sources.yaml`. Before publishing a Kaggle dataset, replace each provisional source with an exact artifact path, release/version, SHA-256, license notice and attribution text. Sources that are unknown, evaluation-only, or restricted are intentionally excluded.\n\n"
        + "\n".join(f"- **{row['name']}** — {row['license']}; {row['attribution']}" for row in rows)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"approved": approved, "excluded": excluded, "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
