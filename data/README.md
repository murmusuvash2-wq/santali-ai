# Data directories

Do not commit downloaded datasets, private recordings, credentials, or model weights here.

- `raw/`: original approved downloads, kept outside Git when large.
- `interim/`: normalized and validated files.
- `processed/`: train/validation/test splits.
- `manifests/`: source, license, checksum, consent, and transformation metadata.

The multilingual record schema should preserve provenance and script identity:

```csv
record_id,source_id,source_url,source_version,artifact_sha256,license_spdx,license_url,attribution,language_primary,language_secondary,script,transliteration_scheme,direction,text_original,text_normalized,domain,code_switch,provenance_type,review_status,quality_score,pii_flag,duplicate_group_id,split
```

For audio, keep a controlled path or object reference rather than copying raw files into the repository or Kaggle by default. Add speaker hash, consent scope, transcript version, duration, segment timestamps, and re-hosting restriction fields.

Only source records marked `approved` by `configs/licensed_sources.yaml` and passing `scripts/audit_corpus.py` may enter a training package. Unknown-license, evaluation-only, restricted, or unreviewed sources remain excluded.

Keep restricted, private, or consented speech data outside public repositories and never upload it to a public Kaggle dataset.
