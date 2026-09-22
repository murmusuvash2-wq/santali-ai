# Contributing to Santali AI

Before changing data preparation, tokenization, model loading, training, evaluation, or bridge behavior, read the [Low-Resource Santali Engineering Standard](docs/LOW_RESOURCE_ENGINEERING_STANDARD.md). It is mandatory project policy, not optional advice.

## Required contribution behavior

Every pull request that changes a model, dataset, tokenizer, or training workflow must explain the affected pre-training and post-training gates. Do not describe a Kaggle or GitHub job as successful unless the kernel output, numerical preflight, metrics, and expected artifacts were checked.

Dataset contributions must include the source URL, language and script, provenance, license, attribution, permission status, transformation steps, and redistribution restrictions. Unclear or restricted sources must remain private and must pass the rights-aware source registry before entering training.

Model contributions must preserve NFC normalization, Ol Chiki validation, language/script routing, leakage checks, tokenizer auditing, finite-logit preflight, reproducible configuration, and held-out evaluation. Hindi and Bengali examples may support mixed-language detection, but they must not displace Santali targets or hide script contamination.

Quality claims should include chrF++ and BLEU or spBLEU where appropriate, script-purity checks, failure cases, and native Santali review. A tokenizer-health score must not be presented as a translation-quality score.

## Pull request checklist

- [ ] I read the low-resource engineering standard.
- [ ] Data provenance, license, attribution, and permissions are recorded.
- [ ] Ol Chiki Unicode and mixed-script validation are preserved or updated.
- [ ] Train/validation/test leakage checks are preserved or updated.
- [ ] Tokenizer and numerical preflight requirements are preserved or updated.
- [ ] The change has reproducible configuration and a clear commit/run reference.
- [ ] Quality claims are supported by held-out metrics and, where applicable, native-speaker review.
- [ ] I have not exposed restricted data, secrets, or private model inputs.
