# Low-Resource Santali Engineering Standard

**Status:** Mandatory project policy  
**Applies to:** English, Hindi, Bengali, Santali, Ol Chiki, translation, conversational, and speech work

## Why this standard exists

Santali is a low-resource language and Ol Chiki is not a high-resource script with abundant pretrained coverage. A successful cloud job, a low training loss, or a valid language code is not enough evidence that a system understands or generates Santali correctly. Every experiment must therefore make script coverage, data provenance, tokenizer behavior, numerical stability, mixed-language robustness, and native-speaker quality visible before claiming progress.

This document is the default engineering contract. Contributors should not need to rediscover or re-explain these requirements in every issue, pull request, or training run.

## Existing verified data comes first

The project must not assume that a low-resource language has no usable data merely because the model learns poorly from the first experiment. Existing verified English–Santali and Santali Ol Chiki data is the starting asset. The immediate problem may be representation, tokenizer fragmentation, task design, split quality, curriculum, or optimization rather than data absence.

Do not wait for a perfect corpus before learning from the available verified data. First improve how the same data is presented: clean target-side Ol Chiki examples, controlled denoising and word-reconstruction variants, explicit language/script tags, and carefully weighted translation directions. Keep every derived variant traceable to its original row. Synthetic Hindi/Bengali expansion is auxiliary and must not replace verified Santali targets or dilute the primary corpus.

“Verified” means accepted by the source and quality gates; it does not mean every row is guaranteed error-free. Small residual errors should be measured and corrected where possible, not used as a reason to discard the whole low-resource corpus.

## Mandatory pre-training gates

No fine-tuning run may start until all of the following are checked and recorded:

1. **Data and rights:** every source has provenance, license, attribution, permission, transformation, and redistribution status in the rights-aware registry. Restricted or unclear-license data remains private and is never silently mixed into a releasable corpus.
2. **Unicode and script:** text is NFC-normalized; expected Ol Chiki characters are present; accidental Bengali, Devanagari, Roman, or replacement-character contamination is reported; empty and duplicate pairs are removed.
3. **Language routing:** Santali, Hindi, Bengali, English, and unknown/mixed examples are counted separately. Mixing must not teach the model to replace Santali output with another script.
4. **Leakage:** duplicate and near-duplicate pairs do not cross train, validation, and test splits. Evaluation examples remain hidden from training preparation.
5. **Tokenizer:** audit unknown tokens, Ol Chiki fragmentation, tokens per word, token expansion relative to English, long-sequence truncation, special language tags, and mixed-script examples. A tokenizer-health proxy score must be reported separately from translation quality.
6. **Numerical preflight:** model weights, encoder hidden states, decoder hidden states, output projection logits, and initial loss must be finite. Any NaN or Inf is a hard stop; do not hide it with clipping, masking, or more epochs.
7. **Reproducibility:** record model identifier, language codes, dataset version/hash, tokenizer, package versions, seed, GPU type, precision, LoRA configuration, and commit SHA.

## Mandatory post-training gates

A completed job is only a candidate artifact. Approval requires:

- held-out BLEU or spBLEU and chrF++;
- output length and empty/duplicate-output checks;
- Ol Chiki script purity and Unicode validation;
- base-model versus fine-tuned comparison;
- NLLB-200 comparison on the same examples when used as a baseline;
- mixed Santali–Hindi–Bengali–English stress tests;
- native Santali adequacy, fluency, terminology, and script review;
- safety checks for agriculture, medical, legal, privacy, hate, and unknown questions; and
- a model card, dataset card, attribution record, and limitations statement.

## Required failure behavior

Pipelines must fail closed. A Kaggle kernel marked `COMPLETE` is not considered successful if its output contains a traceback, runtime error, non-finite preflight result, missing metrics, or missing expected artifacts. GitHub Actions must inspect the downloaded output and report failure rather than publishing a false green run.

The Gemma–IndicTrans bridge remains internal until the translation and safety gates pass. Low-confidence translation or reasoning must route to clarification or review rather than producing an unqualified answer.

## Required experiment record

Every training or evaluation run should answer these questions in its report:

- Which model and tokenizer were used?
- Which exact language/script codes were used?
- How many examples and which sources entered each split?
- What was the Ol Chiki and mixed-script coverage?
- Did the tokenizer pass, require review, or fail?
- Did every numerical preflight stage remain finite?
- Did fine-tuning improve over the base model?
- How was output quality reviewed by Santali speakers?
- What data, model, and software licenses apply?
- What limitations prevent public or production use?

## Project-specific rule

The project must not claim that a model is trained, robust, production-ready, or conversationally capable solely because GitHub Actions, Kaggle, or a dashboard reports success. Claims must link to the corresponding logs, metrics, source records, and review evidence.

## Related documents

- [Post-Training Validation and Integration Plan](POST_TRAINING_VALIDATION_PLAN.md)
- [Data policy](DATA_POLICY.md)
- [Evaluation documentation](EVALUATION.md)
- [Internal Gemma–IndicTrans bridge](INTERNAL_GEMMA_INDIC_BRIDGE.md)
