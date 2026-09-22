# Post-Training Validation and Integration Plan

**Status:** Active after the first successful Kaggle training run  
**Primary direction:** English (`eng_Latn`) → Santali Ol Chiki (`sat_Olck`)  
**Repository:** `murmusuvash2-wq/santali-ai`

## Purpose

A successful GitHub Actions or Kaggle run proves that the training infrastructure completed. It does not, by itself, prove that the model translates Santali accurately. This plan defines the evidence required before another training run, bridge integration, or any public-facing release.

The current trained artifact must therefore be validated in the following order: tokenizer audit, held-out translation evaluation, comparison against the base model and NLLB-200, approved data integration, training decision, and only then internal Gemma–IndicTrans bridge testing.

## Phase 1: Tokenizer audit

The first task is to inspect the tokenizer that was used by the trained model. The audit must use representative English, Santali Ol Chiki, Hindi, Bengali, and mixed-script examples from the same preprocessing path used during training.

The audit will report Ol Chiki character coverage, unknown-token occurrences, tokens per word, characters per token, English-to-Santali token expansion, special language-tag handling, long-sequence truncation, and script behavior. It must also include agriculture, education, conversational, names, numbers, punctuation, and long-sentence examples.

The following conditions are warning signals rather than universal hard thresholds: excessive Ol Chiki fragmentation, any unexpected unknown tokens, unusually high Santali token expansion relative to English, frequent truncation, incorrect handling of `eng_Latn` and `sat_Olck`, or generated text that falls back to Devanagari, Bengali, or Roman script when Ol Chiki is expected.

If fragmentation is severe, the tokenizer/model compatibility must be fixed before more fine-tuning. Additional epochs cannot reliably compensate for a broken or unsuitable tokenization path.

## Phase 2: Held-out translation evaluation

After the tokenizer passes, evaluate the original IndicTrans2 base model and the trained LoRA model on an untouched test split. Report BLEU or spBLEU and chrF++, together with output length ratio, empty-output count, duplicate-output count, and Ol Chiki script purity.

A small manually reviewed sample should be included for adequacy, fluency, terminology, script correctness, and harmful or misleading translation behavior. Native Santali reviewers should be used whenever possible. Test examples must not be copied into training data.

The evaluation must answer two questions: whether fine-tuning improved on the base model, and whether the output is genuinely Santali in Ol Chiki rather than merely well-formed text.

## Phase 3: NLLB-200 benchmark

NLLB-200 should be evaluated as a controlled baseline, not accepted or rejected based only on model size or marketing claims. Use the same test examples, target script, preprocessing assumptions, metrics, and review criteria for IndicTrans2 base, IndicTrans2 LoRA, and NLLB-200.

The comparison record must include model identifier, language codes, checkpoint/license, inference settings, BLEU or spBLEU, chrF++, script purity, failure cases, and reviewer notes. NLLB-200's `sat_Olck` coverage makes it a valid comparison candidate, but measured quality on this project corpus determines the decision.

## Phase 4: Approved data integration

Only datasets that pass the rights-aware source registry may enter training. Candidate P0 sources, including SantaliConnect and Agriculture QA, must be reviewed for provenance, license, attribution, consent, transformations, and redistribution restrictions before ingestion.

Each accepted source must pass schema validation, Unicode and Ol Chiki validation, language/script routing, duplicate removal, alignment checks, contamination checks, and train/validation/test leakage checks. Data versions and source citations must be recorded with every training run.

Hindi and Bengali examples may be used to improve mixed-language detection and routing, but their volume must be controlled so the model does not learn to replace Santali output with Devanagari, Bengali, or Roman text.

## Phase 5: Training decision

Use the validation evidence to choose one of three outcomes:

1. **Tokenizer and fine-tuning pass:** keep the current LoRA candidate and proceed to internal bridge testing.
2. **Tokenizer passes but quality improvement is weak:** revise data mixture, learning rate, LoRA configuration, or training schedule, then run a controlled follow-up experiment.
3. **Tokenizer fails:** fix tokenizer/model compatibility and regenerate the training artifact before changing training hyperparameters.

Every follow-up run must preserve the previous checkpoint, dataset version, configuration, and evaluation results so improvements are reproducible rather than anecdotal.

## Phase 6: Internal Gemma–IndicTrans bridge

Bridge integration begins only after offline translation gates pass. The bridge remains internal and non-public during this phase, consistent with `src/santali_ai/bridge.py`.

For Santali input, the intended path is:

```text
Santali Ol Chiki input
  → IndicTrans: Santali to internal English
  → Gemma reasoning
  → IndicTrans: English answer to Santali Ol Chiki
```

For English input, the intended path is:

```text
English input
  → Gemma reasoning
  → IndicTrans: English answer to Santali Ol Chiki
```

The language/script router must support Santali, Hindi, Bengali, English, and unknown/mixed input. Low-confidence translation or reasoning must produce a review or clarification path rather than an unqualified answer. Public release remains disabled until evaluation, safety, provenance, and native-speaker review gates pass.

## Release gates

No model is considered production-ready until it has:

- a tokenizer audit with no unresolved critical failure;
- held-out BLEU or spBLEU and chrF++ results;
- base-versus-LoRA evidence;
- a documented NLLB-200 comparison;
- native-speaker adequacy and fluency review;
- script-purity and mixed-language routing checks;
- dataset source, license, attribution, and transformation records;
- safety tests for medical, agriculture, privacy, hate, and unknown questions; and
- a model card describing limitations and intended use.

The first successful run is therefore a **validated training artifact candidate**, not an automatic public release.

## Current next action

Run the tokenizer audit against the downloaded artifact from the latest successful Kaggle run. Store the audit output with the run artifacts, then proceed to held-out evaluation only if no critical tokenizer issue is found.

## References

- [IndicTrans2 official repository](https://github.com/AI4Bharat/IndicTrans2)
- [AI4Bharat IndicTrans2 model page](https://ai4bharat.iitm.ac.in/areas/model/NMT/IndicTrans2/)
- [NLLB-200 distilled model card](https://huggingface.co/facebook/nllb-200-distilled-600M)
- [Project data policy](DATA_POLICY.md)
- [Project evaluation documentation](EVALUATION.md)
- [Internal bridge contract](../src/santali_ai/bridge.py)
