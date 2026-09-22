#!/usr/bin/env python3
"""Audit IndicTrans2 tokenization for Santali, English, and mixed-script text.

This produces tokenizer-health evidence only; it is not a translation-quality score.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import types
import unicodedata
from pathlib import Path
from statistics import mean, median

OL_CHIKI = re.compile(r"[\u1C50-\u1C7F]")
DEVANAGARI = re.compile(r"[\u0900-\u097F]")
BENGALI = re.compile(r"[\u0980-\u09FF]")


def install_transformers_onnx_shim() -> None:
    """Support IndicTrans2 custom code on Transformers builds without onnx."""
    try:
        import transformers.onnx  # type: ignore[attr-defined]
        return
    except Exception:
        pass
    module = types.ModuleType("transformers.onnx")
    module.__path__ = []

    class OnnxConfig:
        default_fixed_batch = 2
        default_fixed_sequence = 8

    class OnnxSeq2SeqConfigWithPast(OnnxConfig):
        use_past = False

        def fill_with_past_key_values_(self, inputs, direction="inputs"):
            return inputs

    module.OnnxConfig = OnnxConfig
    module.OnnxSeq2SeqConfigWithPast = OnnxSeq2SeqConfigWithPast
    sys.modules["transformers.onnx"] = module
    utils = types.ModuleType("transformers.onnx.utils")

    def compute_effective_axis_dimension(dimension, fixed_dimension, num_token_to_add=0):
        if dimension is None or dimension < 0:
            return fixed_dimension + num_token_to_add
        return dimension + num_token_to_add

    utils.compute_effective_axis_dimension = compute_effective_axis_dimension
    sys.modules["transformers.onnx.utils"] = utils


def nonspace_chars(text: str) -> int:
    return sum(1 for c in text if not c.isspace())


def word_count(text: str) -> int:
    return max(1, len(re.findall(r"\S+", text)))


def script_counts(text: str) -> dict[str, int]:
    return {
        "ol_chiki": len(OL_CHIKI.findall(text)),
        "devanagari": len(DEVANAGARI.findall(text)),
        "bengali": len(BENGALI.findall(text)),
        "latin": sum(1 for c in text if "LATIN" in unicodedata.name(c, "")),
    }


def read_pairs(path: Path, limit: int) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"No rows found in {path}")
    source_key = next((k for k in ("English", "english", "source", "src") if k in rows[0]), None)
    target_key = next((k for k in ("Santali", "santali", "target", "tgt") if k in rows[0]), None)
    if not source_key or not target_key:
        raise SystemExit(f"Could not find English/Santali columns in {path}; columns={list(rows[0])}")
    pairs = []
    for row in rows[:limit]:
        source, target = (row.get(source_key, "") or "").strip(), (row.get(target_key, "") or "").strip()
        if source and target:
            pairs.append({"english": source, "santali": target})
    return pairs


def measure(tokenizer, texts: list[str], label: str, max_length: int) -> dict:
    records = []
    unk_id = tokenizer.unk_token_id
    for text in texts:
        raw = tokenizer(text, add_special_tokens=True, truncation=False, return_attention_mask=False)
        ids = raw["input_ids"]
        truncated = len(ids) > max_length
        limited = tokenizer(text, add_special_tokens=True, truncation=True, max_length=max_length, return_attention_mask=False)
        limited_ids = limited["input_ids"]
        unknown = sum(1 for token_id in ids if unk_id is not None and token_id == unk_id)
        chars = nonspace_chars(text)
        tokens = len(ids)
        records.append({
            "label": label,
            "chars": chars,
            "words": word_count(text),
            "tokens": tokens,
            "tokens_per_word": tokens / word_count(text),
            "tokens_per_char": tokens / max(1, chars),
            "unknown_tokens": unknown,
            "unknown_rate": unknown / max(1, tokens),
            "truncated": truncated,
            "kept_tokens": len(limited_ids),
            "scripts": script_counts(text),
        })
    return summarize(records, label)


def summarize(records: list[dict], label: str) -> dict:
    def avg(key: str) -> float:
        return round(mean(float(r[key]) for r in records), 4)
    def med(key: str) -> float:
        return round(median(float(r[key]) for r in records), 4)
    return {
        "label": label,
        "samples": len(records),
        "mean_tokens_per_word": avg("tokens_per_word"),
        "median_tokens_per_word": med("tokens_per_word"),
        "mean_tokens_per_char": avg("tokens_per_char"),
        "median_tokens_per_char": med("tokens_per_char"),
        "unknown_token_rate": round(sum(r["unknown_tokens"] for r in records) / max(1, sum(r["tokens"] for r in records)), 6),
        "samples_with_unknown": sum(r["unknown_tokens"] > 0 for r in records),
        "truncation_rate": round(sum(r["truncated"] for r in records) / max(1, len(records)), 6),
        "max_tokens": max(r["tokens"] for r in records),
        "p95_tokens": sorted(r["tokens"] for r in records)[min(len(records) - 1, math.ceil(len(records) * 0.95) - 1)],
    }


def health_score(stats: list[dict]) -> dict:
    by_label = {s["label"]: s for s in stats}
    sat = by_label["santali"]
    # This is a transparent proxy score, not BLEU/chrF and not a quality claim.
    unknown_component = max(0.0, 1.0 - sat["unknown_token_rate"] * 100.0)
    trunc_component = max(0.0, 1.0 - sat["truncation_rate"])
    fragmentation_component = min(1.0, 2.0 / max(0.01, sat["mean_tokens_per_char"]))
    expansion = sat["mean_tokens_per_word"] / max(0.01, by_label["english"]["mean_tokens_per_word"])
    expansion_component = min(1.0, 1.0 / max(1.0, expansion / 2.5))
    score = 100.0 * (0.35 * unknown_component + 0.25 * trunc_component + 0.25 * fragmentation_component + 0.15 * expansion_component)
    level = "PASS" if score >= 80 and sat["unknown_token_rate"] == 0 and sat["truncation_rate"] <= 0.02 else "REVIEW" if score >= 60 else "FAIL"
    return {
        "score": round(score, 2),
        "level": level,
        "components": {
            "unknown_tokens": round(unknown_component * 100, 2),
            "truncation": round(trunc_component * 100, 2),
            "fragmentation": round(fragmentation_component * 100, 2),
            "relative_expansion": round(expansion_component * 100, 2),
        },
        "note": "Tokenizer-health proxy only; translation quality requires held-out chrF++/BLEU and human review.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ai4bharat/indictrans2-en-indic-dist-200M")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/tokenizer-audit.json"))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    install_transformers_onnx_shim()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    pairs = read_pairs(args.input, args.limit)
    english = [p["english"] for p in pairs]
    santali = [p["santali"] for p in pairs]
    mixed = [f'{p["santali"]} {p["english"]}' for p in pairs[: min(100, len(pairs))]]
    stats = [
        measure(tokenizer, english, "english", args.max_length),
        measure(tokenizer, santali, "santali", args.max_length),
        measure(tokenizer, mixed, "mixed_santali_english", args.max_length),
    ]
    result = {
        "model": args.model,
        "input": str(args.input),
        "max_length": args.max_length,
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": len(tokenizer),
        "unk_token": tokenizer.unk_token,
        "special_tokens": tokenizer.special_tokens_map,
        "stats": stats,
        "tokenizer_health": health_score(stats),
        "sample_script_counts": {"santali": script_counts(santali[0]), "mixed": script_counts(mixed[0])},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
