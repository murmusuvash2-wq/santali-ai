#!/usr/bin/env python3
"""Minimal Kaggle LoRA scaffold for IndicTrans2.

Run after reviewing the prepared CSVs.
Note: For best results, prefer the official IndicTransToolkit preprocessor
in a future iteration. This script is a simple starting point.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

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

DEFAULT_MODEL = "ai4bharat/indictrans2-en-indic-dist-200M"
DEFAULT_SRC = "eng_Latn"
DEFAULT_TGT = "sat_Olck"


def load_csv(path: Path) -> Dataset:
    df = pd.read_csv(path).fillna("")
    if not {"source", "target"}.issubset(df.columns):
        raise ValueError(f"CSV must contain 'source' and 'target' columns. Found: {list(df.columns)}")
    return Dataset.from_pandas(df[["source", "target"]], preserve_index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Directory containing train/validation/test.csv")
    ap.add_argument("--output-dir", required=True, help="Where to save the LoRA adapter")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--src-lang", default=DEFAULT_SRC)
    ap.add_argument("--tgt-lang", default=DEFAULT_TGT)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    for split in ("train", "validation", "test"):
        if not (data_dir / f"{split}.csv").exists():
            raise FileNotFoundError(f"Missing {split}.csv in {data_dir}")

    print(f"Loading data from {data_dir}")
    ds = DatasetDict({
        s: load_csv(data_dir / f"{s}.csv") for s in ("train", "validation", "test")
    })
    print({k: len(v) for k, v in ds.items()})

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, trust_remote_code=True)

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    src_lang = args.src_lang
    tgt_lang = args.tgt_lang

    def encode(batch):
        # Simple prefix format (works as a baseline).
        # For production, switch to official IndicTransToolkit preprocessing.
        sources = [f"{src_lang} {tgt_lang} {text}" for text in batch["source"]]
        model_inputs = tokenizer(
            sources,
            max_length=128,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            text_target=batch["target"],
            max_length=128,
            truncation=True,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = ds.map(
        encode,
        batched=True,
        remove_columns=ds["train"].column_names,
        desc="Tokenizing",
    )

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out),
        learning_rate=5e-5,
        num_train_epochs=args.epochs,
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
        load_best_model_at_end=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=collator,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {out}")
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))

    print("Evaluating on test set...")
    metrics = trainer.evaluate(tokenized["test"])
    metrics_path = out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Done. Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
