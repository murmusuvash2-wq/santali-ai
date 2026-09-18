# Licensed multilingual Santali corpus

This repository uses a **provenance-first** policy. Public visibility is not treated as permission to train, redistribute, or upload to Kaggle. Every source must be reviewed at file or release level before it becomes eligible.

## Scope

The target corpus will preserve Santali in Ol Chiki and Latin/Sanlish, together with clearly labelled Hindi and Bengali code-switching or parallel data. Hindi and Bengali are supporting languages; they must not replace Santali supervision. Evaluation data remains separate from training data.

## Eligibility gate

A source may enter the Kaggle training package only when all of the following are present: an exact source URL, release or commit identifier, a compatible license, explicit training-use permission, explicit redistribution or private-hosting permission, attribution text, an artifact SHA-256, language and script labels, and a review status of `approved`. Unknown-license, evaluation-only, restricted-audio, or unreviewed sources are excluded automatically.

The current Mod4 English–Santali source is intentionally marked `excluded_unknown_license`. Its 63,179-pair experiment must not be represented as a cleared corpus until the upstream license is confirmed.

## Initial source priorities

The lowest-risk candidates are verified CC0 or clearly licensed sources, such as a file-level-confirmed Santali word list and explicitly cleared components of AI4Bharat data. BPCC must be filtered to Santali rows and audited component-by-component; its corpus-wide headline size is not a Santali count. Google SMOL may be considered after exact-release inspection and CC-BY attribution. Common Voice Santali is useful for a controlled ASR pilot, but raw-audio re-hosting on Kaggle requires separate review because the release restrictions include re-hosting limitations.

FLORES/IndicGenBench and IN22 are evaluation resources and must stay out of training. The Sanlish thesis recordings referenced by the public model card are not publicly released and cannot be reconstructed or redistributed from the model.

## Row schema

Each text or parallel record should include `record_id`, `source_id`, `source_url`, `source_version`, `artifact_sha256`, `license_spdx`, `license_url`, `attribution`, `language_primary`, `language_secondary`, `script`, `transliteration_scheme`, `direction`, `text_original`, `text_normalized`, `domain`, `code_switch`, `provenance_type`, `review_status`, `quality_score`, `pii_flag`, `duplicate_group_id`, and `split`.

Speech manifests additionally need `audio_uri_or_controlled_path`, `duration_seconds`, `transcript_original`, `transcript_normalized`, `segment_start`, `segment_end`, `validated_flag`, `speaker_id_hash`, `consent_id`, `consent_scope`, `consent_date`, `rehosting_restriction`, and `retention_until`. Raw audio must not be copied into Kaggle until its rights and hosting terms are explicitly cleared.

## Recommended mixture

The initial target is 50% native Ol Chiki, 20% Hindi–Santali aligned or reviewed code-switching, 15% Bengali–Santali aligned or reviewed code-switching, 10% native-reviewed Latin/Sanlish, and 5% genuinely conversational code-switching. These are planning targets, not claims about current availability. Missing or uncleared buckets must remain pending rather than being filled with scraped or synthetic text.

## Release checklist

Before a Kaggle version is created, run `python scripts/audit_corpus.py`, attach the generated `sources.csv`, `dataset-metadata.json`, and `ATTRIBUTION.md`, verify every included artifact hash, preserve license notices, remove PII, deduplicate before splitting, split by source document and speaker, and keep a native-speaker audit sample outside the public repository. Synthetic examples must be labelled separately and must pass native review before inclusion.
