# Research data implementation

The researched sources are now represented in `configs/licensed_sources.yaml` as a rights-aware registry. The registry distinguishes `trainable`, `private_training_only`, `trainable_after_audit`, `manual_review`, `eval_only`, and `restricted` sources. It records the source URL, license or terms URL, attribution text, language/script, training permission, redistribution condition, and notes on provenance.

The current local audit reports **2 eligible private-training sources** and **12 excluded or review-gated sources**. The eligible set is intentionally small: the confirmed word-list artifact and the private MMLoSo baseline artifact. Candidate sources such as SantaliConnect, Google SMOL, Common Voice, IndicVoices, Rasa, BPCC, and the Santali ASR subset are registered but cannot enter Kaggle training until their exact release, provenance, permissions, checksums, and native-review requirements are satisfied.

FLORES, IndicGenBench, IN22, and other benchmark material are hard-coded as evaluation-only. The unknown-license Mod4 copy remains restricted and cannot be re-hosted or merged into an open corpus. Open educational intent and attribution are recorded, but they do not override upstream license, competition, speaker-consent, or redistribution terms.

CI now runs `scripts/validate_source_registry.py` and `scripts/audit_corpus.py`. No source downloader was added deliberately: a source may enter training only after an operator records an exact artifact and SHA-256 hash. This makes the next ingestion step reviewable and prevents accidental inclusion of restricted or benchmark data.

## Next ingestion order

The next candidates are the 506-row agriculture QA set, SantaliConnect, and the exact Google SMOL Santali files. Each will be added as a separate shard with a source manifest and native-review flag; no mixed aggregate will be labelled open until all component conditions are compatible.
