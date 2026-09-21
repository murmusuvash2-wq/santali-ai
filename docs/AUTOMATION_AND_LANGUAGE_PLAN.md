# Santali AI automation and language-routing plan

## Weekly schedule

The GitHub Actions training workflow runs every **Monday at 20:00 IST** (`14:30 UTC`). The workflow first audits licenses and checksums, then prepares the private approved-parallel Kaggle dataset, downloads the gated IndicTrans2 En-Indic checkpoint through `HF_TOKEN`, verifies the private model dataset, and starts the Kaggle GPU kernel. Manual dispatch remains available for urgent experiments.

## Truthful training numbers

The dashboard reads the public GitHub Actions API and shows the latest workflow ID, total training workflow count, completed and failed counts, workflow progress, and the fixed next-run schedule. It deliberately does not invent GPU epoch percentages: those values will appear only after Kaggle publishes a machine-readable metrics artifact.

## Mixed-language routing

The first implementation is an explainable script-first router in `scripts/detect_language_mix.py`. Ol Chiki is a strong Santali signal. Devanagari and Bengali script are emitted as review-required candidates because Santali can be written in those scripts and script identity alone cannot prove Hindi or Bengali. Latin text is treated as a possible Romanized/code-mixed input.

The next model phase should combine IndicLID, character n-grams trained on native-labelled Santali/Hindi/Bengali examples, vocabulary evidence, and token-level code-mix labels. Training data must keep pure Santali, Hindi, Bengali, Devanagari Santali, Bengali-script Santali, Romanized Santali, and mixed examples as separate provenance-aware shards.

## Current Kaggle runtime fix

The latest log confirmed that both the approved-parallel dataset and the En-Indic model dataset mounted correctly. The remaining failure was a Transformers compatibility issue: IndicTrans2 calls `_tie_or_clone_weights`, which newer Transformers builds no longer expose on the custom model class. The local loader now injects a compatibility implementation before loading the model.
