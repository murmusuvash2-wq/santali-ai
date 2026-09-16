# Kaggle training guide

## Notebook setup

1. Create a Kaggle Notebook with GPU enabled.
2. Add this repository as a dataset, or clone it in the notebook.
3. Add approved public data as Kaggle inputs.
4. Install dependencies from `requirements.txt`.
5. Run preprocessing and inspect the report before training.

```bash
pip install -q -r requirements.txt
python training/kaggle_prepare_data.py \
  --input /kaggle/input/approved-parallel/parallel.csv \
  --output-dir /kaggle/working/santali_data
python training/kaggle_translation_lora.py \
  --data-dir /kaggle/working/santali_data \
  --output-dir /kaggle/working/santali-indictrans2-adapter
```

## Kaggle input expectations

The initial CSV must contain:

```text
source,target,source_lang,target_lang,domain,license,verified
```

Use `source_lang=hin_Deva` and `target_lang=sat_Olck` for the first run. Keep reverse-direction rows explicitly labelled rather than silently swapping them.

## Reproducibility

Record the dataset version, model revision, package versions, GPU type, random seed, maximum sequence length, batch size, gradient accumulation, learning rate, number of epochs and final metrics. Save `manifest.json`, `metrics.json` and the adapter with the notebook output.

## What Kaggle is good for

Kaggle is suitable for a first 320M translation experiment and small LoRA/adapter runs. It may be unsuitable for long-running ASR/TTS training, large full-model fine-tuning or private-data work because sessions are temporary and public sharing can expose data.
