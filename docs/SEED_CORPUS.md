# Ol Chiki seed corpus

The first reproducible seed is built from the verified MMLoSo 2025 English–Santali artifact. The raw and processed CSV files remain ignored from Git because they are dataset artifacts; the build script and manifest schema are versioned.

## Current build

- Input rows: **20,000**
- Accepted pairs: **19,869**
- Exact duplicate pairs removed: **130**
- Target rows rejected because they did not contain Ol Chiki: **1**
- Train: **15,895**
- Validation: **1,987**
- Test: **1,987**
- Split seed: `42`
- Source: MMLoSo 2025 English–Santali, CC BY-SA 4.0 as recorded in `configs/licensed_sources.yaml`

The training profile currently has zero empty rows, zero duplicate pairs, and complete provenance fields. Most targets are Ol Chiki, with a small number containing Latin or Devanagari alongside Ol Chiki; these require native-speaker review before release.

## Rebuild

```bash
python scripts/build_seed_corpus.py \
  --input data/interim/mmloso-2025/parallel.csv \
  --output-dir data/processed/seed-2025 \
  --manifest data/manifests/seed-2025.json

python scripts/profile_parallel.py \
  --input data/processed/seed-2025/train.csv \
  --output /tmp/seed-train-profile.json
```

Or use Make:

```bash
make build-seed-corpus \
  INPUT=data/interim/mmloso-2025/parallel.csv \
  OUTPUT_DIR=data/processed/seed-2025 \
  MANIFEST=data/manifests/seed-2025.json
```

## Important limitations

This is a **licensed data preparation milestone**, not a claim of translation quality. The deterministic split does not replace native Santali review. FLORES-IN and other benchmark rows remain outside the input and must never be merged into training. Keep source partitions separate when adding Tatoeba, GitHub dictionaries, Latin transliteration, Hindi, or Bengali data so that a rights or quality problem can be removed safely.

Before Kaggle upload, run `scripts/audit_corpus.py`, inspect the manifest checksums, and generate the required attribution text. Do not publish the processed files until the MMLoSo attribution and ShareAlike notices are included.
