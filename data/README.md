# Data directories

Do not commit downloaded datasets or model weights here.

- `raw/`: original approved downloads, kept outside Git when large.
- `interim/`: normalized and validated files.
- `processed/`: train/validation/test splits.
- `manifests/`: source, license, checksum and transformation metadata.

The expected first CSV schema is:

```csv
source,target,source_lang,target_lang,domain,license,verified
```

Keep restricted, private or consented data outside the public repository and never upload it to a public Kaggle dataset.
