# Santali AI implementation plan

## Decision

The project will not chase a single large fine-tune. It will build a reproducible multilingual translation and conversation foundation. The first production-facing milestone is a licensed Ol Chiki translation baseline. Hindi, Bengali, Latin/Sanlish, speech, and conversation are separate tracks with separate data and evaluation gates.

## Phase 1: benchmark before training

Freeze evaluation tracks for English↔Ol Chiki. Add Hindi↔Ol Chiki, Bengali↔Ol Chiki, and Ol Chiki↔Latin only after rights-cleared data or native-reviewed fixtures exist. Run IndicTrans2 and NLLB-200 on the same rows. Record BLEU, chrF++, spBLEU, script validity, length ratio, and native-speaker scores. Never use benchmark rows in training.

The benchmark configuration is in `configs/benchmarks.yaml`. The deterministic evaluator is `scripts/evaluate_predictions.py`. Its fallback scores are diagnostics; publication numbers must use SacreBLEU and a documented tokenization profile.

## Phase 2: licensed corpus

Add exact-release MMLoSo data after verifying its CC BY-SA 4.0 terms, attribution, and permitted Kaggle hosting. Extract an exact Santali slice from BPCC only after component-level review. Keep the current unknown-license Mod4 source quarantined. Approve sources in `configs/licensed_sources.yaml` only after recording artifact path, immutable version, SHA-256, training permission, redistribution/private-hosting permission, and attribution.

Build `parallel.csv` with `scripts/prepare_licensed_kaggle_dataset.py`. The GitHub workflow refuses to upload data when no approved parallel source exists. This is intentional.

## Phase 3: fine-tune adapters

Train a small LoRA adapter for English↔Ol Chiki first. Preserve the base IndicTrans2 checkpoint. Add Hindi and Bengali adapters only after direction-specific data and test sets are ready. Add a Latin/Sanlish adapter only after a native-approved transliteration convention is defined. Keep synthetic teacher data below a controlled fraction and label every synthetic row.

## Phase 4: human review and error loop

Create a reviewer interface for adequacy, fluency, spelling, names, numbers, code-switching, and harmful outputs. Sample difficult examples using active learning instead of collecting arbitrary web text. Store only consented corrections and preserve a held-out community-reviewed test set.

## Phase 5: conversation and voice

Use a separate instruction model plus retrieval-augmented generation for Santali conversation. Use IndicTrans2 as the translation layer, not as the entire chatbot. Build speech-to-text and text-to-speech as separate components. Keep audio manifests, consent, speaker hashes, script, dialect, and re-hosting restrictions separate from text data.

## Release gate

A model version is not release-ready until the dataset card, source ledger, hashes, license notices, train/dev/test split, benchmark metrics, native review sample, known limitations, model card, and serving smoke test are complete. A GitHub workflow success means packaging and handoff passed; Kaggle output must separately prove that training completed and produced adapter files plus metrics.
