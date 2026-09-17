# Approved dataset source review

## Selected source

The approved source is `https://huggingface.co/datasets/aiswarya9302/english-santali-datasetmod4`. The dataset page reports 63,179 rows split into approximately 50,500 train, 6,320 validation, and 6,320 test rows. Its fields are `src`, `tgt`, `src_len`, and `tgt_len`; the target examples are Santali in Ol Chiki script. The repository exposes `train.csv`, `valid.csv`, and `test.csv` files.

## Provenance limitation

The dataset page currently has no dataset card and does not state a clear license. The automated Kaggle copy therefore uses Kaggle license `unknown`, is private, and preserves the source URL and provenance note. It must not be made public or redistributed until the original data license is confirmed.

## Official references

- Source dataset: https://huggingface.co/datasets/aiswarya9302/english-santali-datasetmod4
- Source file tree: https://huggingface.co/datasets/aiswarya9302/english-santali-datasetmod4/tree/main
- Kaggle dataset metadata specification: https://github.com/Kaggle/kaggle-api/blob/main/docs/datasets_metadata.md
- Kaggle dataset CLI create/version guide: https://github.com/Kaggle/kaggle-api/blob/main/docs/datasets.md
- FLORES+ is evaluation-only and should not be used as the training corpus: https://huggingface.co/datasets/openlanguagedata/flores_plus
