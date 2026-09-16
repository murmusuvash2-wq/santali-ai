# Data policy and online-data assessment

## Can online data be used now?

Yes. Public online resources are enough to build a **baseline translation model and a reproducible Kaggle experiment**. They are not enough to claim production-quality Santali understanding without review.

## Candidate resources

| Resource | Use | Decision |
|---|---|---|
| AI4Bharat BPCC | Parallel translation data | Candidate training source after filtering and license review |
| AI4Bharat IndicTrans2 artifacts | Baseline model and scripts | Primary translation starting point |
| FLORES+ `sat_Olck` | Held-out evaluation | Evaluation only; do not train on it |
| AI4Bharat Rasa | Santali TTS speech | Candidate CC-BY-4.0 research data; preserve attribution |
| Common Voice Santali | ASR | Verify release, transcript quality and current terms before use |
| Santali Wikipedia/public-domain text | Language modelling and RAG | Use only with source and license metadata |
| Public GitHub/Hugging Face datasets | Discovery and augmentation | Accept only after provenance and quality audit |

## Required metadata

Every row must carry a dataset ID, original URL, license, source language, target language, script, creation method, quality score and reviewer status. For audio, also store speaker ID, consent status, region, recording conditions and transcript version.

## Data rules

- Do not scrape copyrighted books, news or YouTube audio for training without permission.
- Do not upload private voice recordings to Kaggle.
- Do not treat machine-translated text as gold data.
- Keep Ol Chiki, Devanagari and Romanized Santali as explicit script labels.
- Deduplicate before splitting.
- Split by source document and speaker to prevent leakage.
- Keep FLORES+ hidden from training.
- Keep a native-speaker audit sample outside the public repository.

## Quality checks

1. Unicode normalization and valid Ol Chiki code points.
2. Language identification on both sides.
3. Length-ratio and alignment checks.
4. Duplicate and near-duplicate detection.
5. Script consistency.
6. Random native-speaker review.
7. License and provenance review.

## Dataset card template

```text
Name:
Version:
Source URL:
License:
Commercial use:
Redistribution:
Language/script:
Rows or hours:
Collection method:
Machine-translated content:
Human verification:
PII removed:
Known limitations:
```
