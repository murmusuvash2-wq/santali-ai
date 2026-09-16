# Santali AI

Open, data-first Santali AI stack for **Ol Chiki text**, Hindi/Santali translation, conversational assistance, and later voice interaction.

> **Current decision:** We will not train a foundation model from scratch. We will start from open Indian-language models, validate online datasets, fine-tune on Kaggle, and keep a native-speaker evaluation gate before deployment.

## Project status

**Phase:** Planning and data validation  
**Training:** Kaggle GPU  
**First milestone:** Hindi ↔ Santali text translation plus a verified Santali knowledge assistant  
**Current data strategy:** Public online data can start the project, but it is not enough by itself for production quality. We must audit licenses, remove machine-generated noise, normalize Ol Chiki, and add native-speaker review.

## What we are building

```text
Phase 1  Hindi ↔ Santali translation API
Phase 2  Santali chatbot with retrieval-augmented generation (RAG)
Phase 3  ASR and TTS voice pipeline
Phase 4  WhatsApp/Telegram and offline Android deployment
```

## Recommended models

| Capability | Starting model | Decision |
|---|---|---|
| Translation | `ai4bharat/indictrans2-indic-indic-dist-320M` | Primary model; supports `sat_Olck` Santali and `hin_Deva` Hindi |
| Chat reasoning | Qwen3-4B/8B Instruct with QLoRA | Use for response style and instruction following, not as the factual database |
| Santali ASR | AI4Bharat IndicConformer Santali checkpoint | Start with inference; fine-tune only after collecting consented labelled speech |
| Santali TTS | AI4Bharat IndicF5 or Indic Parler-TTS | Evaluate existing voices first; adapt only with licensed voice data |

## Online data: can we start now?

**Yes, for a baseline and first Kaggle experiment.** The following resources are usable candidates:

1. **AI4Bharat BPCC:** large multilingual parallel-corpus collection for Indic translation. Use only the relevant language pairs after filtering and license review.
2. **AI4Bharat IndicTrans2 artifacts:** model, tokenizer, scripts, training and evaluation resources.
3. **FLORES+:** Santali `sat_Olck` evaluation material. Keep it as a held-out test set; do not train on it.
4. **AI4Bharat Rasa:** Santali speech data for TTS research, listed as CC-BY-4.0. Follow attribution and access conditions.
5. **Common Voice Santali:** candidate ASR data. Verify the current release, locale, clip count, transcript quality and CC0 terms before downloading.
6. **Public Santali text:** Wikipedia and other openly licensed/public-domain sources, after provenance and script checks.

Online data is **not sufficient alone** for a reliable public bot. Machine-translated pairs, duplicate web text, mixed scripts, dialect variation and incorrect Ol Chiki spellings must be measured. Production requires native-speaker review and consented field speech.

## Repository layout

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── Makefile
├── requirements.txt
├── .env.example
├── configs/
│   ├── data_sources.yaml
│   └── translation_lora.yaml
├── data/
│   ├── README.md
│   ├── raw/.gitkeep
│   ├── interim/.gitkeep
│   ├── processed/.gitkeep
│   └── manifests/.gitkeep
├── docs/
│   ├── PLAN.md
│   ├── DATA_POLICY.md
│   ├── KAGGLE.md
│   └── EVALUATION.md
├── diagrams/
│   └── architecture.mmd
├── scripts/
│   ├── validate_parallel.py
│   ├── normalize_olchiki.py
│   └── split_parallel.py
├── training/
│   ├── kaggle_translation_lora.py
│   └── kaggle_prepare_data.py
└── src/santali_ai/
    └── __init__.py
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/validate_parallel.py --input data/raw/parallel.csv --output data/interim/parallel_validated.csv
python scripts/split_parallel.py --input data/interim/parallel_validated.csv --output-dir data/processed
```

## Kaggle workflow

1. Create a Kaggle notebook with GPU enabled.
2. Add the approved public datasets as Kaggle inputs or download them using the Hugging Face `datasets` library.
3. Run `training/kaggle_prepare_data.py`.
4. Inspect the validation report before training.
5. Run `training/kaggle_translation_lora.py`.
6. Export the adapter and tokenizer as a Kaggle output.
7. Evaluate on the held-out FLORES+/native-review set.
8. Do not upload private, consented or restricted data to a public Kaggle dataset.

Kaggle is appropriate for the first translation experiment. For a larger full fine-tune, ASR, or TTS job, use a controlled GPU environment with persistent storage.

## Quality gates

A model cannot be marked production-ready until it passes all of these gates:

- No train/test leakage by source sentence or speaker.
- At least two native Santali reviewers inspect a held-out sample.
- Ol Chiki Unicode and script checks pass.
- BLEU/chrF are reported together with human adequacy and fluency scores.
- Safety tests cover medical, agriculture, hate, privacy and unknown questions.
- Every dataset has a source, license, consent/provenance and transformation record.

## References

[1]: https://github.com/ai4bharat/IndicTrans2 "AI4Bharat IndicTrans2 repository"
[2]: https://ai4bharat.iitm.ac.in/areas/model/ASR/IndicConformer/ "AI4Bharat IndicConformer ASR"
[3]: https://huggingface.co/datasets/ai4bharat/BPCC "AI4Bharat BPCC dataset"
[4]: https://huggingface.co/datasets/openlanguagedata/flores_plus "FLORES+ evaluation dataset"
[5]: https://huggingface.co/datasets/ai4bharat/Rasa "AI4Bharat Rasa TTS dataset"
[6]: https://github.com/common-voice/common-voice "Mozilla Common Voice project"
[7]: https://github.com/QwenLM/Qwen3 "Qwen3 official repository"

This project is research software. Model outputs are not medical, legal or emergency advice.
