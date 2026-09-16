#!/usr/bin/env python3
"""Minimal Kaggle LoRA scaffold for IndicTrans2.

Run after reviewing the prepared CSVs. Verify the current IndicTransToolkit API
against the pinned IndicTrans2 revision before a long training job.
"""
import argparse, json
from pathlib import Path
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import LoraConfig, TaskType, get_peft_model

MODEL = 'ai4bharat/indictrans2-indic-indic-dist-320M'
SRC, TGT = 'hin_Deva', 'sat_Olck'

def load_csv(path):
    df = pd.read_csv(path).fillna('')
    return Dataset.from_pandas(df[['source','target']], preserve_index=False)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data-dir', required=True); ap.add_argument('--output-dir', required=True); ap.add_argument('--epochs', type=float, default=3)
    args = ap.parse_args(); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    ds = DatasetDict({s: load_csv(Path(args.data_dir) / f'{s}.csv') for s in ('train','validation','test')})
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, trust_remote_code=True)
    peft = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM, r=16, lora_alpha=32, lora_dropout=0.05, target_modules=['q_proj','k_proj','v_proj','out_proj'])
    model = get_peft_model(model, peft)
    def encode(batch):
        src = [f'{SRC} {TGT} {x}' for x in batch['source']]
        tok = tokenizer(src, max_length=128, truncation=True)
        with tokenizer.as_target_tokenizer(): tok['labels'] = tokenizer(batch['target'], max_length=128, truncation=True)['input_ids']
        return tok
    tokenized = ds.map(encode, batched=True, remove_columns=ds['train'].column_names)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    train_args = Seq2SeqTrainingArguments(output_dir=str(out), learning_rate=5e-5, num_train_epochs=args.epochs, per_device_train_batch_size=2, per_device_eval_batch_size=2, gradient_accumulation_steps=8, evaluation_strategy='steps', eval_steps=250, save_steps=250, logging_steps=25, predict_with_generate=True, fp16=True, report_to='none', save_total_limit=2)
    trainer = Seq2SeqTrainer(model=model, args=train_args, train_dataset=tokenized['train'], eval_dataset=tokenized['validation'], tokenizer=tokenizer, data_collator=collator)
    trainer.train(); trainer.save_model(str(out)); tokenizer.save_pretrained(str(out))
    metrics = trainer.evaluate(tokenized['test']); (out/'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))
if __name__ == '__main__': main()
