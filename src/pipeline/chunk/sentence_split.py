"""Sentence splitting.

Tries nltk's punkt first; if unavailable, falls back to a regex splitter.
Returns a list of stripped sentences.
"""
from __future__ import annotations

import re

_FALLBACK = re.compile(r"(?<=[\.\?\!])\s+(?=[A-Z0-9\"'(])|\n{2,}")


def split_sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    try:
        import nltk
        try:
            from nltk.tokenize import sent_tokenize
            return [s.strip() for s in sent_tokenize(text) if s.strip()]
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            from nltk.tokenize import sent_tokenize
            return [s.strip() for s in sent_tokenize(text) if s.strip()]
    except Exception:
        parts = _FALLBACK.split(text)
        return [p.strip() for p in parts if p and p.strip()]
