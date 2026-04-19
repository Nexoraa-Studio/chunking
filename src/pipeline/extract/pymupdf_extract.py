"""Stage 1 alternative: PyMuPDF (pymupdf / fitz) extractor.

No ML layout model — much faster than docling. Heading detection uses
font-size clustering:
  1. Pass 1: scan every text span, bucket by font size (weighted by char count)
  2. body_size = most-frequent font size
  3. heading sizes = distinct sizes > body*1.05 with enough content
  4. Pass 2: each block whose max span size matches a heading bucket AND is
     short enough to be a heading becomes a SectionHeaderItem; block with a
     leading bullet/roman/enumerated marker becomes a ListItem; else TextItem.

Writes the same output artifacts as docling_extract.py so downstream stages
(structural, coarse_dp) don't need to change:
  data/interim/docling_elements.jsonl
  data/interim/docling.md
  data/interim/docling_stats.json  (with {"extractor": "pymupdf", ...})
"""
from __future__ import annotations

import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.pipeline.utils import paths
from src.pipeline.utils.io import write_json, write_jsonl

LIST_PREFIX_RE = re.compile(
    r"^\s*(?:[•·\-*◦▪▫‣⁃]|\d+[\.\)]|\([a-z0-9]+\)|[ivx]+\.)\s+[A-Za-z0-9]"
)


def _collect_body_and_heading_sizes(doc) -> tuple[float, list[float]]:
    """Char-weighted histogram of span sizes -> (body_size, heading_sizes).

    Threshold for heading candidates is intentionally low (>= 5 chars
    per size bucket) because real-section headings can be short
    ('2 Methods' = 9 chars, '3.1 Data' = 8 chars) and would otherwise
    be filtered out alongside genuine font noise (say, a single
    subscript glyph)."""
    sc: Counter = Counter()
    for page_num in range(len(doc)):
        pd = doc[page_num].get_text("dict", sort=True)
        for block in pd.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = round(span.get("size", 0), 1)
                    if size > 0:
                        sc[size] += len(span.get("text", ""))
    if not sc:
        raise RuntimeError("PyMuPDF found no text — is this a scanned PDF?")
    body = max(sc.items(), key=lambda kv: kv[1])[0]
    heading_sizes = sorted(
        [s for s, c in sc.items() if s > body * 1.05 and c >= 5],
        reverse=True,
    )[:4]
    return body, heading_sizes


def _iter_blocks(doc):
    """Yield (page_num_1based, block_text, max_size, has_bold) over the doc."""
    for page_num in range(len(doc)):
        pd = doc[page_num].get_text("dict", sort=True)
        for block in pd.get("blocks", []):
            if block.get("type") != 0:
                continue
            parts: list[str] = []
            sizes: list[float] = []
            any_bold = False
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    t = span.get("text", "")
                    if t:
                        parts.append(t)
                    sizes.append(round(span.get("size", 0), 1))
                    if span.get("flags", 0) & 16:  # bit 4 = bold
                        any_bold = True
                parts.append(" ")  # line break -> space
            text = "".join(parts).strip()
            text = re.sub(r"\s+", " ", text)
            if not text:
                continue
            yield (page_num + 1, text, max(sizes) if sizes else 0.0, any_bold)


def main() -> None:
    import pymupdf  # type: ignore

    paths.INTERIM.mkdir(parents=True, exist_ok=True)
    pdf = paths.current_pdf()
    if pdf is None:
        raise FileNotFoundError(f"No PDF in {paths.RAW}")

    print(f"[pymupdf] extracting {pdf.name} ...", flush=True)
    t0 = time.time()
    doc = pymupdf.open(str(pdf))

    body_size, heading_sizes = _collect_body_and_heading_sizes(doc)
    print(f"[pymupdf] body_size={body_size}  heading_sizes={heading_sizes}", flush=True)

    # Heading level lookup: largest size -> level 1, next -> 2, ...
    size_to_level = {s: i + 1 for i, s in enumerate(heading_sizes)}

    records: list[dict] = []
    idx = 0
    for page_num, text, max_size, bold in _iter_blocks(doc):
        level = size_to_level.get(max_size, 0)
        # Fallback heading signal: a short bold block at body size is often a
        # section marker (academic papers with uniform font sizes).
        is_bold_short = bold and len(text) <= 120 and max_size >= body_size
        if level == 0 and is_bold_short:
            level = max(2, len(heading_sizes)) + 1  # treat as lowest-level heading

        # Classification
        if level >= 1 and len(text) <= 250:
            typ = "SectionHeaderItem"
        elif LIST_PREFIX_RE.match(text):
            typ = "ListItem"
        else:
            typ = "TextItem"

        rec = {
            "idx": idx,
            "page": page_num,
            "type": typ,
            "level": 1,  # docling-compat: a single flat level; structural.py
                         # re-infers hierarchy from the number prefix anyway.
            "parent_heading": None,
            "text": text,
        }
        if typ == "SectionHeaderItem":
            rec["heading_level"] = level or 1

        records.append(rec)
        idx += 1

    elapsed = time.time() - t0

    el_path = paths.INTERIM / "docling_elements.jsonl"
    write_jsonl(el_path, records)
    md = "\n\n".join(r["text"] for r in records)
    (paths.INTERIM / "docling.md").write_text(md, encoding="utf-8")

    type_counts: dict[str, int] = {}
    for r in records:
        type_counts[r["type"]] = type_counts.get(r["type"], 0) + 1

    stats = {
        "extractor": "pymupdf",
        "elapsed_seconds": round(elapsed, 2),
        "n_elements": len(records),
        "n_pages_seen": len(doc),
        "body_font_size": body_size,
        "heading_sizes": heading_sizes,
        "markdown_chars": len(md),
        "type_counts": type_counts,
    }
    write_json(paths.INTERIM / "docling_stats.json", stats)
    doc.close()

    print(f"[pymupdf] done in {elapsed:.1f}s  "
          f"elements={len(records)}  pages={stats['n_pages_seen']}  "
          f"types={type_counts}", flush=True)


if __name__ == "__main__":
    main()
