#!/usr/bin/env python3
"""Explainable baseline language/script router for mixed Santali input.

This is intentionally conservative: it detects script evidence first and emits
candidate languages instead of pretending that script alone proves language.
It is a bootstrap for the later IndicLID + character-ngram ensemble.
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter

TOKEN_RE = re.compile(r"\S+", re.UNICODE)

# High-signal hints only; these are not a replacement for native-speaker labels.
HINTS = {
    "sat": {"ᱦᱚᱲ", "ᱥᱟᱱᱛᱟᱲᱤ", "ᱡᱟᱦᱟᱸ", "ᱢᱤᱫ", "ᱱᱟᱜᱟᱢ"},
    "hin": {"है", "और", "का", "के", "की", "में", "से", "यह", "एक"},
    "ben": {"এবং", "এই", "করে", "জন্য", "থেকে", "এক", "বাংলা"},
}


def char_script(ch: str) -> str | None:
    code = ord(ch)
    if 0x1C50 <= code <= 0x1C7F:
        return "Olck"
    if 0x0900 <= code <= 0x097F:
        return "Deva"
    if 0x0980 <= code <= 0x09FF:
        return "Beng"
    if ch.isascii() and ch.isalpha():
        return "Latn"
    return None


def detect(text: str) -> dict:
    counts: Counter[str] = Counter()
    for ch in text:
        script = char_script(ch)
        if script:
            counts[script] += 1

    tokens = TOKEN_RE.findall(text)
    hint_scores = Counter()
    token_rows = []
    for token in tokens:
        normalized = unicodedata.normalize("NFC", token.strip(".,!?;:()[]{}\"'"))
        matches = [lang for lang, words in HINTS.items() if normalized in words]
        for lang in matches:
            hint_scores[lang] += 1
        token_rows.append({"text": token, "hint_languages": matches})

    scripts = [script for script, count in counts.items() if count]
    if "Olck" in scripts:
        primary = "sat"
    elif "Deva" in scripts:
        primary = "hin_or_sat_Deva"
    elif "Beng" in scripts:
        primary = "ben_or_sat_Beng"
    elif "Latn" in scripts:
        primary = "latn_mixed"
    else:
        primary = "unknown"

    candidates = []
    for lang, score in hint_scores.most_common():
        candidates.append({"language": lang, "evidence": score})
    if "Olck" in scripts and not any(item["language"] == "sat" for item in candidates):
        candidates.insert(0, {"language": "sat", "evidence": "Olck_script"})

    code_mixed = len(scripts) > 1 or len(hint_scores) > 1
    return {
        "language": "sat_mixed" if code_mixed and ("sat" in hint_scores or "Olck" in scripts) else primary,
        "script": "+".join(scripts) if scripts else "unknown",
        "is_code_mixed": code_mixed,
        "secondary_languages": [item["language"] for item in candidates[1:]],
        "candidates": candidates,
        "confidence": "high" if "Olck" in scripts else "review_required",
        "needs_review": primary in {"hin_or_sat_Deva", "ben_or_sat_Beng", "latn_mixed"},
        "tokens": token_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?", help="Text to inspect; stdin is used when omitted")
    args = parser.parse_args()
    text = args.text if args.text is not None else input().strip()
    print(json.dumps(detect(text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
