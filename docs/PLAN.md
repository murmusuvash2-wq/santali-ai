# Santali AI implementation plan

## 1. Scope

The first release is an English/Hindi ↔ Santali translation service and a Santali knowledge assistant. Voice and messaging integrations come after the text baseline is measured. This ordering reduces risk: translation data is easier to validate than speech data, and a retrieval system is safer for changing factual information than baking facts into model weights. All milestones follow the mandatory [Low-Resource Santali Engineering Standard](LOW_RESOURCE_ENGINEERING_STANDARD.md).

## 2. System architecture

The architecture is documented in [`../diagrams/architecture.mmd`](../diagrams/architecture.mmd).

The translation path uses IndicTrans2 with the language codes `hin_Deva`, `sat_Olck` and `eng_Latn`. The assistant path detects the input script, retrieves approved documents, generates an answer with an open-weight instruction model, and translates the answer into the requested output language. The voice path is optional and uses IndicConformer for ASR and IndicF5 or Indic Parler-TTS for speech synthesis.

## 3. Milestones

### M0 — Repository and data audit

Record each dataset's URL, license, access condition, language, script, size, provenance and known limitations. Validate NFC normalization, Ol Chiki coverage, mixed-script contamination, duplicates, and train/evaluation leakage. Never mix restricted or private data into the public training pipeline.

### M1 — Online-data baseline

Use approved public data to create a small Hindi–Santali and English–Santali baseline. Run the tokenizer audit and numerical finite-logit preflight before fine-tuning. Normalize Unicode, identify scripts, remove duplicates and split by source. Run the existing IndicTrans2 model before fine-tuning so improvement is measurable.

### M2 — Kaggle translation fine-tuning

Fine-tune the `ai4bharat/indictrans2-en-indic-dist-200M` IndicTrans2 checkpoint with a small learning rate. Start with a controlled subset only after tokenizer, encoder, decoder, `lm_head`, and initial-loss checks are finite. Save the tokenizer, adapter/checkpoint, configuration, dataset manifest and metrics as one versioned experiment.

### M3 — Native evaluation

Ask at least two Santali speakers to review a hidden test set. Score meaning preservation, grammar, naturalness, terminology, script purity and Ol Chiki correctness. Report chrF++ alongside BLEU/spBLEU and test mixed Santali–Hindi–Bengali–English inputs. Keep the test set private from the training notebook.

### M4 — RAG assistant

Add approved agriculture, education and public-service documents. Each answer should retain a source reference. Configure refusal and escalation behavior for medical emergencies, legal matters and unknown questions.

### M5 — Speech

Collect consented speech separately. Start with existing Santali ASR/TTS checkpoints. Fine-tune only after speaker-balanced train/dev/test splits and transcript quality checks exist.

## 4. Data target

The first experiment can run with 10,000–50,000 cleaned parallel pairs. The production target is a larger, expert-reviewed corpus with domain labels. For a chatbot, add at least 5,000 verified instruction/response examples before claiming Santali conversational quality. For ASR, target 20–50 labelled hours for domain adaptation and 100+ diverse hours for a robust public bot. For TTS, target 15–25 clean hours per voice.

## 5. Training strategy

Do not train a foundation model from scratch. Begin with supervised fine-tuning or LoRA/QLoRA. Use full fine-tuning only if the baseline is clearly limited and sufficient compute and data are available. Use RAG for changing facts and domain documents. Fine-tuning should teach language behavior, formatting and task style, not replace source-grounded retrieval.

## 6. Kaggle constraints

Kaggle is suitable for the first translation LoRA experiment. Keep the notebook restartable, cache models under `/kaggle/working`, save checkpoints to `/kaggle/working/output`, and export the final artifact through Kaggle Outputs. Do not place secrets or private recordings in a public notebook. Record GPU type, dataset versions, seed, package versions and runtime duration.

## 7. Definition of done

A release candidate must have reproducible preprocessing, a held-out test set, baseline and fine-tuned metrics, a model card, a dataset card, native-speaker review, a safety report, and a serving smoke test. A GitHub README claim must link to evidence rather than only a planned feature.
