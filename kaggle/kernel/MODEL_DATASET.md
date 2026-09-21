# Optional: attach IndicTrans2 as a local Kaggle dataset

When Kaggle DNS/Internet fails, HuggingFace download breaks.
Upload the model once as a Kaggle dataset and the kernel will use it automatically.

## 1. Download model (on your PC / Colab with good internet)

```bash
pip install -U huggingface_hub
huggingface-cli download ai4bharat/indictrans2-en-indic-dist-200M --local-dir ./indictrans2-en-indic-200m
```

Folder must contain at least:
- `config.json`
- `*.safetensors` or `pytorch_model.bin`
- tokenizer files

## 2. Create Kaggle dataset

- Name suggestion: `indictrans2-en-indic-200m` (owner: `janaiworkspace`)
- Upload the folder contents
- Keep private or public

## 3. Attach to the kernel

On https://www.kaggle.com/code/janaiworkspace/santali-ai-translation

1. **Add Input** → your dataset `indictrans2-en-indic-200m`
2. Re-run

## 4. What the script does

Load order:

1. `MODEL_PATH` environment variable (if set)
2. `/kaggle/input/indictrans2-en-indic-200m`
3. `/kaggle/input/indictrans2-indic-indic-dist-320m`
4. `/kaggle/input/ai4bharat-indictrans2-en-indic-200m`
5. `/kaggle/input/indictrans2`
6. Any other `/kaggle/input/**/config.json` with weights
7. **Else** HuggingFace: `ai4bharat/indictrans2-en-indic-dist-200M`

Log line you want to see:

```
Using LOCAL model: /kaggle/input/indictrans2-en-indic-200m
```

or

```
No local model found ... will load from HuggingFace
```
