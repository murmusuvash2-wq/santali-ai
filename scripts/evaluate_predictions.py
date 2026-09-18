#!/usr/bin/env python3
"""Evaluate aligned reference/prediction files without touching training data.

Input files are CSV/TSV with one source, reference, and prediction column, or
plain text with one sentence per line. Outputs a versioned JSON metric ledger.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Iterable

OL_CHIKI = re.compile(r"[\u1C50-\u1C7F]")
LATIN = re.compile(r"[A-Za-z]")

def norm(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value or "").strip().split())

def read_column(path: Path, column: str | None) -> list[str]:
    if path.suffix.lower() in {".txt", ".text"}:
        return [norm(line) for line in path.read_text(encoding="utf-8").splitlines()]
    delimiter = "\t" if path.suffix.lower() in {".tsv"} else ","
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle, delimiter=delimiter)
        name = column or (rows.fieldnames or [None])[0]
        if not name:
            raise ValueError(f"No column found in {path}")
        return [norm(row.get(name, "")) for row in rows]

def exact_match(refs: Iterable[str], preds: Iterable[str]) -> float:
    refs, preds = list(refs), list(preds)
    return sum(r == p for r, p in zip(refs, preds)) / max(1, len(refs))

def char_f1(reference: str, prediction: str) -> float:
    ref = list(reference.replace(" ", "")); pred = list(prediction.replace(" ", ""))
    if not ref and not pred: return 1.0
    if not ref or not pred: return 0.0
    ref_counts: dict[str, int] = {}; pred_counts: dict[str, int] = {}
    for char in ref: ref_counts[char] = ref_counts.get(char, 0) + 1
    for char in pred: pred_counts[char] = pred_counts.get(char, 0) + 1
    overlap = sum(min(count, pred_counts.get(char, 0)) for char, count in ref_counts.items())
    precision = overlap / len(pred); recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

def bleu_fallback(refs: list[str], preds: list[str]) -> float:
    scores = []
    for ref, pred in zip(refs, preds):
        ref_tokens, pred_tokens = ref.split(), pred.split()
        if not pred_tokens: scores.append(0.0); continue
        overlap = sum(1 for token in pred_tokens if token in ref_tokens)
        brevity = min(1.0, math.exp(1 - len(ref_tokens) / max(1, len(pred_tokens))))
        scores.append(100 * brevity * overlap / len(pred_tokens))
    return sum(scores) / max(1, len(scores))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--reference-column", default=None)
    parser.add_argument("--prediction-column", default=None)
    parser.add_argument("--track", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    refs = read_column(Path(args.references), args.reference_column)
    preds = read_column(Path(args.predictions), args.prediction_column)
    if len(refs) != len(preds):
        raise SystemExit(f"Alignment mismatch: references={len(refs)} predictions={len(preds)}")
    if not refs:
        raise SystemExit("No aligned rows found")
    chrf = sum(char_f1(r, p) for r, p in zip(refs, preds)) / len(refs) * 100
    result = {
        "track": args.track,
        "rows": len(refs),
        "exact_match": exact_match(refs, preds),
        "bleu_fallback": bleu_fallback(refs, preds),
        "chrf_fallback": chrf,
        "reference_olchiki_rate": sum(bool(OL_CHIKI.search(r)) for r in refs) / len(refs),
        "prediction_olchiki_rate": sum(bool(OL_CHIKI.search(p)) for p in preds) / len(preds),
        "prediction_latin_rate": sum(bool(LATIN.search(p)) for p in preds) / len(preds),
        "mean_reference_chars": sum(len(r) for r in refs) / len(refs),
        "mean_prediction_chars": sum(len(p) for p in preds) / len(preds),
        "note": "Fallback metrics are deterministic diagnostics. Use SacreBLEU chrF++/spBLEU for publication metrics.",
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__": main()
