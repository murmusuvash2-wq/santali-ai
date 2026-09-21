#!/usr/bin/env python3
"""Validate the rights-aware source registry without downloading data."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ALLOWED_CLASSES = {"trainable", "trainable_after_audit", "manual_review", "eval_only", "restricted", "private_training_only"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/licensed_sources.yaml")
    args = parser.parse_args()
    document = yaml.safe_load(Path(args.manifest).read_text(encoding="utf-8"))
    policy = document.get("policy", {})
    required_policy = [
        "require_license", "require_training_permission", "require_source_url",
        "require_attribution", "exclude_unknown_license", "exclude_evaluation_only",
    ]
    errors: list[str] = [f"missing policy: {key}" for key in required_policy if not policy.get(key)]
    sources = document.get("sources", [])
    ids: set[str] = set()
    for index, source in enumerate(sources):
        sid = str(source.get("id", ""))
        prefix = f"sources[{index}]/{sid or 'missing-id'}"
        if not sid:
            errors.append(f"{prefix}: id is required")
        if sid in ids:
            errors.append(f"{prefix}: duplicate id")
        ids.add(sid)
        for field in ("name", "source_url", "license", "attribution", "status", "use_class", "training_use", "redistribution"):
            if not str(source.get(field, "")).strip():
                errors.append(f"{prefix}: missing {field}")
        use_class = source.get("use_class")
        if use_class not in ALLOWED_CLASSES:
            errors.append(f"{prefix}: unknown use_class={use_class!r}")
        if use_class == "eval_only" and source.get("training_use") not in {"prohibited", "evaluation_only"}:
            errors.append(f"{prefix}: eval_only source must be prohibited from training")
        if use_class in {"manual_review", "restricted"} and source.get("status") in {"approved", "approved_private"}:
            errors.append(f"{prefix}: review/restricted source cannot be approved")
        if source.get("license", "").lower() == "unknown" and use_class not in {"restricted", "eval_only"}:
            errors.append(f"{prefix}: unknown license must be restricted or eval_only")
        if source.get("status") in {"approved", "approved_private"} and not source.get("artifact_sha256"):
            errors.append(f"{prefix}: approved source requires artifact_sha256")

    if errors:
        print("Source registry validation failed:")
        print("\n".join(f"- {error}" for error in errors))
        raise SystemExit(1)
    print(f"Source registry valid: {len(sources)} sources; no downloads performed.")


if __name__ == "__main__":
    main()
