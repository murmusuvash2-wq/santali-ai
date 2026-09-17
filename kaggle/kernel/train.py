"""Kaggle entrypoint for the uploaded script kernel.

Hardened version: prints directory layout and uses absolute paths so
FileNotFoundError is easier to diagnose and less likely.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

def main() -> None:
    # On Kaggle script kernels the uploaded files land in the working directory.
    repo = Path.cwd()
    print("=" * 60)
    print("Kaggle kernel starting")
    print(f"cwd          : {repo}")
    print(f"__file__     : {__file__ if '__file__' in globals() else 'N/A'}")
    print("Directory listing:")
    for p in sorted(repo.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(repo)}")
    print("=" * 60)

    # Install requirements if present
    requirements = repo / "requirements-kaggle.txt"
    if requirements.exists():
        print(f"Installing {requirements} ...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)
        ])
    else:
        print("No requirements-kaggle.txt found, skipping pip install")

    # Locate the training scripts (prefer next to this file)
    train_dir = repo / "training"
    if not train_dir.is_dir():
        # Fallback: sometimes Kaggle puts things under /kaggle/working
        train_dir = Path("/kaggle/working/training")

    prepare_script = train_dir / "kaggle_prepare_data.py"
    lora_script = train_dir / "kaggle_translation_lora.py"

    if not prepare_script.exists():
        raise FileNotFoundError(
            f"Cannot find {prepare_script}\n"
            f"cwd contents: {[p.name for p in repo.iterdir()]}\n"
            "Make sure kaggle/kernel/training/ was included when the kernel was pushed."
        )
    if not lora_script.exists():
        raise FileNotFoundError(f"Cannot find {lora_script}")

    input_csv = os.environ.get(
        "INPUT_CSV", "/kaggle/input/approved-parallel/parallel.csv"
    )
    if not Path(input_csv).exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_csv}\n"
            "Attach the 'approved-parallel' dataset to the kernel."
        )

    out = Path("/kaggle/working/santali-output")
    out.mkdir(parents=True, exist_ok=True)
    data_dir = out / "data"
    adapter_dir = out / "adapter"

    print(f"Running prepare: {prepare_script}")
    subprocess.check_call([
        sys.executable,
        str(prepare_script),
        "--input", input_csv,
        "--output-dir", str(data_dir),
    ])

    print(f"Running LoRA training: {lora_script}")
    subprocess.check_call([
        sys.executable,
        str(lora_script),
        "--data-dir", str(data_dir),
        "--output-dir", str(adapter_dir),
    ])

    print("Training artifacts:", out)
    print("Done.")

if __name__ == "__main__":
    main()
