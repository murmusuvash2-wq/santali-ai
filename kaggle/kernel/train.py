"""Kaggle entrypoint — fully self-contained.

Kaggle script kernels execute only the code_file (this file) from
/kaggle/src/script.py. Supporting files in training/ or scripts/ are
NOT available at runtime. Everything needed is inlined here.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
INPUT_CSV = os.environ.get(
    "INPUT_CSV", "/kaggle/input/approved-parallel/parallel.csv"
)
OUT = Path("/kaggle/working/santali-output")
DATA_DIR = OUT / "data"
ADAPTER_DIR = OUT / "adapter"
MODEL_ID = "ai4bharat/indictrans2-indic-indic-dist-320M"
SRC_LANG = "hin_Deva"
TGT_LANG = "sat_Olck"
OL_CHIKI = re.compile(r"[\u1C50-\u1C7F]")
EPOCHS = float(os.environ.get("EPOCHS", "3"))


def log(msg: str) -> None:
    print(msg, flush=True)


def debug_layout() -> None:
    log("=" * 60)
    log("Kaggle kernel starting (self-contained)")
    log(f"cwd      : {Path.cwd()}")
    log(f"__file__  : {globals().get('__file__', 'N/A')}")
    for root in [Path.cwd(), Path("/kaggle/src"), Path("/kaggle/working"), Path("/kaggle/input")]:
        if root.exists():
            files = sorted(p for p in root.rglob("*") if p.is_file())[:40]
            log(f"{root} ({len(list(root.rglob('*')))} entries):")
            for p in files:
                try:
                    log(f"  {p}")
                except Exception:
                    pass
    log("=" * 60)


def install_deps() -> None:
    pkgs = [
        "transformers>=4.40",
        "datasets>=2.18",
        "accelerate>=0.28",
        "peft>=0.10",
        "sentencepiece>=0.2",
        "pandas>=2.0",
        "sacrebleu>=2.4",
        "evaluate>=0.4",
    ]
    log("Installing training dependencies...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *pkgs],
    )


# ---------------------------------------------------------------------------
# Data prep (inlined from scripts/)
# ---------------------------------------------------------------------------
def normalize_rows(rows: list[dict]) -> list[dict]:
    for r in rows:
        for k in ("source", "target"):
            if r.get(k):
                r[k] = unicodedata.normalize("NFC", r[k]).strip()
    return rows


def validate_rows(rows: list[dict]) -> list[dict]:
    kept, seen = [], set()
    stats = {"read": 0, "kept": 0, "duplicates": 0, "empty": 0, "no_olchiki": 0}
    for r in rows:
        stats["read"] += 1
        source = (r.get("source") or "").strip()
        target = (r.get("target") or "").strip()
        if not source or not target:
            stats["empty"] += 1
            continue
        key = (source, target)
        if key in seen:
            stats["duplicates"] += 1
            continue
        seen.add(key)
        # If target looks like Ol Chiki language, require at least one Ol Chiki char
        tlang = r.get("target_lang", "sat_Olck")
        if tlang == "sat_Olck" and not OL_CHIKI.search(target):
            stats["no_olchiki"] += 1
            continue
        kept.append(r)
        stats["kept"] += 1
    log(f"validate: {json.dumps(stats)}")
    return kept


def split_rows(rows: list[dict], seed: int = 42) -> dict[str, list[dict]]:
    random.Random(seed).shuffle(rows)
    n = len(rows)
    if n == 0:
        return {"train": [], "validation": [], "test": []}
    return {
        "train": rows[: int(0.8 * n)],
        "validation": rows[int(0.8 * n) : int(0.9 * n)],
        "test": rows[int(0.9 * n) :],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["source", "target"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def prepare_data() -> Path:
    csv_path = Path(INPUT_CSV)
    if not csv_path.exists():
        # try common alternate locations
        alts = list(Path("/kaggle/input").rglob("parallel.csv"))
        if alts:
            csv_path = alts[0]
            log(f"Using alternate input: {csv_path}")
        else:
            raise FileNotFoundError(
                f"Input CSV not found: {INPUT_CSV}\n"
                f"/kaggle/input contents: {list(Path('/kaggle/input').rglob('*'))[:30]}\n"
                "Attach dataset ezqrio/approved-parallel to the kernel."
            )

    log(f"Reading {csv_path}")
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    log(f"Raw rows: {len(rows)}")

    # Map common column names to source/target
    if rows and ("source" not in rows[0] or "target" not in rows[0]):
        for r in rows:
            if "src" in r and "source" not in r:
                r["source"] = r.get("src", "")
            if "tgt" in r and "target" not in r:
                r["target"] = r.get("tgt", "")
            if "en" in r and "source" not in r:
                r["source"] = r.get("en", "")
            if "sat" in r and "target" not in r:
                r["target"] = r.get("sat", "")

    rows = normalize_rows(rows)
    rows = validate_rows(rows)
    if not rows:
        raise RuntimeError("No valid parallel pairs after validation.")

    splits = split_rows(rows)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        write_csv(DATA_DIR / f"{name}.csv", data)
        log(f"  {name}: {len(data)}")
    return DATA_DIR


# ---------------------------------------------------------------------------
# LoRA training (inlined)
# ---------------------------------------------------------------------------
def train_lora(data_dir: Path) -> None:
    import pandas as pd
    from datasets import Dataset, DatasetDict
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )
    from peft import LoraConfig, TaskType, get_peft_model

    def load_csv(path: Path) -> Dataset:
        df = pd.read_csv(path).fillna("")
        if not {"source", "target"}.issubset(df.columns):
            raise ValueError(f"Need source/target columns, got {list(df.columns)}")
        return Dataset.from_pandas(df[["source", "target"]], preserve_index=False)

    for split in ("train", "validation", "test"):
        p = data_dir / f"{split}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")

    ds = DatasetDict({s: load_csv(data_dir / f"{s}.csv") for s in ("train", "validation", "test")})
    log(f"Dataset sizes: {{k: len(v) for k, v in ds.items()}}")

    log(f"Loading model {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, trust_remote_code=True)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    def encode(batch):
        sources = [f"{SRC_LANG} {TGT_LANG} {t}" for t in batch["source"]]
        model_inputs = tokenizer(sources, max_length=128, truncation=True, padding=False)
        labels = tokenizer(
            text_target=batch["target"], max_length=128, truncation=True, padding=False
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = ds.map(
        encode, batched=True, remove_columns=ds["train"].column_names, desc="Tokenizing"
    )
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    args = Seq2SeqTrainingArguments(
        output_dir=str(ADAPTER_DIR),
        learning_rate=5e-5,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        eval_strategy="steps",
        eval_steps=250,
        save_steps=250,
        logging_steps=25,
        predict_with_generate=True,
        fp16=True,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=collator,
    )

    log("Starting training...")
    trainer.train()
    trainer.save_model(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))

    log("Evaluating on test set...")
    metrics = trainer.evaluate(tokenized["test"])
    metrics_path = ADAPTER_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(json.dumps(metrics, indent=2))
    log(f"Artifacts saved to {ADAPTER_DIR}")


def main() -> None:
    debug_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    install_deps()
    data_dir = prepare_data()
    train_lora(data_dir)
    log("Done.")


if __name__ == "__main__":
    main()
