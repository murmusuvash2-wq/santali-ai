# GitHub → Kaggle automatic trigger

## Short answer

GitHub cannot trigger Kaggle through a native built-in repository webhook in this project. It **can trigger Kaggle through GitHub Actions and the official Kaggle CLI/API**.

This repository includes `.github/workflows/kaggle-train.yml`. A push to `main` that changes training, scripts, configs or the Kaggle runner will call `kaggle kernels push`; a manual `workflow_dispatch` run is also available in the GitHub Actions tab.

The official Kaggle CLI documents that `kaggle kernels push` uploads the kernel code and metadata and runs the kernel. It also supports an accelerator and timeout. The kernel status and output can be queried after the run.

## One-time setup

1. In Kaggle, create or authorize the private kernel whose ID is in `kaggle/kernel/kernel-metadata.json`:

   ```text
   ezqrio/santali-ai-translation
   ```

2. In GitHub repository settings, open **Secrets and variables → Actions**.
3. Add these repository secrets:

   ```text
   KAGGLE_USERNAME     your Kaggle username
   KAGGLE_API_TOKEN    your Kaggle API token
   ```

4. Attach an approved Kaggle dataset to the kernel. The runner expects:

   ```text
   /kaggle/input/approved-parallel/parallel.csv
   ```

5. Push to `main`, or open **Actions → Kaggle training → Run workflow**.

## Security rules

Never put a Kaggle token in the repository, workflow YAML, notebook, commit message or public dataset. Use GitHub Actions Secrets. Keep the Kaggle kernel private while the data is under review. Do not send private or consented speech data to a public Kaggle kernel.

## What this automation does

```text
GitHub push/manual run
        ↓
GitHub Actions runner
        ↓
Kaggle CLI authentication from GitHub Secrets
        ↓
kaggle kernels push
        ↓
Kaggle GPU kernel executes kaggle/kernel/train.py
        ↓
Adapter + metrics are saved to Kaggle output
```

## Important limitations

- GitHub Actions only starts the Kaggle run; it does not bypass Kaggle quotas, GPU availability, account permissions or session limits.
- A Kaggle input dataset must already be attached or referenced in kernel metadata.
- The current workflow does not automatically download model outputs back into GitHub. Add a separate, authenticated download step only after deciding where model artifacts should be stored.
- Kaggle metadata field names and accelerator availability can change. If Kaggle rejects a metadata field, run `kaggle kernels init` with the installed CLI version and update the JSON accordingly.
