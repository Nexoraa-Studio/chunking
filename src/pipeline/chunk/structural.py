"""Stage 2: Structural chunking — organize docling elements by section.

Key behaviors
-------------
1. Infer heading depth from the numbering prefix in the heading text
   ("1." -> L1, "1.1" -> L2, "2.3.1" -> L3). Docling reports all headings
   at heading_level=1 so we cannot trust it; we parse the numbers instead.
2. Drop table-of-contents tables: any TableItem whose rendered markdown is
   >50% dots/whitespace is a dotted-leader TOC and is discarded.
3. Group every non-heading element under the nearest preceding heading.
4. Emit two artifacts:
   - data/chunks/structural.jsonl  (flat list, input to semantic stage)
   - data/interim/document.json    (hierarchical tree, for inspection/debug)

Chunk record (structural.jsonl):
  {id, strategy, number, heading, heading_level, heading_path,
   pages, element_idxs, n_chars, text, contains_table, n_sentences_est}
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.pipeline.utils import paths
from src.pipeline.utils.io import read_jsonl, write_json, write_jsonl

NUMBER_PREFIX = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+")

# Post-split: heading-like prefix inside a larger blob that docling missed.
# Academic PDFs often render headings inline with their body paragraph
# ("1.1. Beyond Dyadic Interactions: ...title... . Body text starts here.").
# We anchor at start-of-line, read number + title up to the first period+space
# which typically marks the heading→body boundary. Title is capped at 120 chars
# to reject multi-sentence false positives.
MISSED_HEADING_RE = re.compile(
    r"(?m)^\s*(\d+(?:\.\d+){0,3})\.?\s+([A-Z][A-Za-z0-9][^\.\n|]{1,120}?)\.\s"
)


def _infer_level(heading_text: str) -> tuple[int | None, str | None]:
    """Return (level, number) inferred from heading text, or (None, None)."""
    m = NUMBER_PREFIX.match(heading_text or "")
    if not m:
        return None, None
    number = m.group(1)
    level = number.count(".") + 1
    return level, number


def _post_split_missed_headings(chunks: list[dict], min_chunk_chars: int = 1024) -> list[dict]:
    """Re-segment any non-table chunk >= `min_chunk_chars` by detecting missed
    numbered-heading lines inside its text. This compensates for docling
    occasionally mis-classifying section headers as body text (most visible on
    academic papers where '2 Methods', '3 Results' share styling with prose).

    Splitting only happens when at least one plausible heading is found inside.
    The new chunks inherit their parent's page list and section metadata, with
    `heading_path` extended by the newly-discovered heading.
    """
    out: list[dict] = []
    for c in chunks:
        if c.get("contains_table") or c["n_chars"] < min_chunk_chars:
            out.append(c)
            continue
        text = c["text"] or ""
        matches = list(MISSED_HEADING_RE.finditer(text))
        if not matches:
            out.append(c)
            continue

        base_path = c.get("heading_path") or c.get("heading")
        base_level = c.get("heading_level") or 1
        pieces: list[tuple[str | None, str]] = []

        # Preamble before first match (if meaningful)
        if matches[0].start() > 0:
            pre = text[:matches[0].start()].strip()
            if pre:
                pieces.append((None, pre))

        for i, m in enumerate(matches):
            num = m.group(1)
            title = m.group(2).strip()
            heading = f"{num} {title}"
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[m.start():end].strip()
            if body:
                pieces.append((heading, body))

        if len(pieces) == 1 and pieces[0][0] is None:
            out.append(c)
            continue

        for (h, body) in pieces:
            new_c = dict(c)
            new_c["n_chars"] = len(body)
            new_c["text"] = body
            if h:
                new_c["heading"] = h
                new_c["heading_path"] = (f"{base_path} > {h}" if base_path else h)
                new_c["heading_level"] = base_level + 1
                new_c["number"] = h.split(" ", 1)[0]
                new_c["flush_reason"] = "post_split_missed_heading"
            else:
                new_c["flush_reason"] = "post_split_preamble"
            out.append(new_c)

    for i, c in enumerate(out):
        c["id"] = i
    return out


def _is_toc_table(text: str) -> bool:
    if not text:
        return False
    stripped = text.strip()
    # proportion of '.' characters in body
    dots = stripped.count(".")
    alnum = sum(c.isalnum() for c in stripped)
    return dots > 0 and dots > alnum


def _heading_path(stack: list[dict]) -> str:
    return " > ".join(h["text"] for h in stack if h["text"] != "__root__")


def build(elements: list[dict]) -> tuple[list[dict], dict]:
    """Returns (flat_chunks, hierarchical_document_tree)."""
    root = {"id": 0, "number": None, "text": "__root__", "level": 0,
            "children": [], "paragraphs": [], "lists": [], "tables": [],
            "pages": set(), "n_chars_self": 0}
    stack: list[dict] = [root]
    next_section_id = 1
    chunks: list[dict] = []

    # per-chunk accumulator (one chunk == contiguous non-heading content under a section)
    buf_text: list[str] = []
    buf_pages: set[int] = set()
    buf_idxs: list[int] = []
    current_section = root

    def flush(reason: str) -> None:
        nonlocal buf_text, buf_pages, buf_idxs
        if not buf_text:
            return
        body = "\n\n".join(buf_text).strip()
        if not body:
            buf_text, buf_pages, buf_idxs = [], set(), []
            return
        chunks.append({
            "id": len(chunks),
            "strategy": "structural",
            "section_id": current_section["id"],
            "number": current_section["number"],
            "heading": current_section["text"] if current_section["text"] != "__root__" else None,
            "heading_level": current_section["level"],
            "heading_path": _heading_path(_path_to(root, current_section)),
            "pages": sorted(buf_pages),
            "element_idxs": buf_idxs,
            "n_chars": len(body),
            "text": body,
            "contains_table": False,
            "flush_reason": reason,
        })
        buf_text, buf_pages, buf_idxs = [], set(), []

    # helper to get path to a node
    def _path_to(root_node: dict, target: dict) -> list[dict]:
        if root_node is target:
            return [root_node]
        for c in root_node["children"]:
            p = _path_to(c, target)
            if p:
                return [root_node] + p
        return []

    for el in elements:
        t = el["type"]
        text = (el.get("text") or "").strip()
        page = el.get("page")

        if t == "SectionHeaderItem":
            flush("new_heading")
            inferred_level, number = _infer_level(text)
            # fallback: if we can't infer, treat as sibling of current
            lvl = inferred_level if inferred_level is not None else max(1, stack[-1]["level"] + 1)
            # pop until parent has strictly smaller level
            while len(stack) > 1 and stack[-1]["level"] >= lvl:
                stack.pop()
            new_node = {
                "id": next_section_id,
                "number": number,
                "text": text,
                "level": lvl,
                "children": [],
                "paragraphs": [],
                "lists": [],
                "tables": [],
                "pages": {page} if page is not None else set(),
                "n_chars_self": 0,
            }
            next_section_id += 1
            stack[-1]["children"].append(new_node)
            stack.append(new_node)
            current_section = new_node
            continue

        if t == "TableItem":
            if _is_toc_table(text):
                continue  # drop TOC noise
            flush("before_table")
            chunks.append({
                "id": len(chunks),
                "strategy": "structural",
                "section_id": current_section["id"],
                "number": current_section["number"],
                "heading": current_section["text"] if current_section["text"] != "__root__" else None,
                "heading_level": current_section["level"],
                "heading_path": _heading_path(_path_to(root, current_section)),
                "pages": [page] if page is not None else [],
                "element_idxs": [el["idx"]],
                "n_chars": len(text),
                "text": text,
                "contains_table": True,
                "n_rows": el.get("n_rows"),
                "n_cols": el.get("n_cols"),
                "flush_reason": "table_atomic",
            })
            current_section["tables"].append({"idx": el["idx"], "page": page,
                                              "n_rows": el.get("n_rows"),
                                              "n_cols": el.get("n_cols"),
                                              "text": text})
            if page is not None:
                current_section["pages"].add(page)
            current_section["n_chars_self"] += len(text)
            continue

        if not text:
            continue

        # text / list item belongs to current section's buffer
        buf_text.append(text)
        buf_idxs.append(el["idx"])
        if page is not None:
            buf_pages.add(page)
            current_section["pages"].add(page)
        current_section["n_chars_self"] += len(text)
        if t == "ListItem":
            current_section["lists"].append({"idx": el["idx"], "page": page, "text": text})
        else:
            current_section["paragraphs"].append({"idx": el["idx"], "page": page, "text": text})

    flush("end")

    # finalize tree: sets -> sorted lists, compute n_chars_total
    def _finalize(node: dict) -> int:
        total = node["n_chars_self"]
        for c in node["children"]:
            total += _finalize(c)
        node["n_chars_total"] = total
        pages = set(node["pages"])
        for c in node["children"]:
            pages.update(c["pages"])
        node["pages"] = sorted(p for p in pages if p is not None)
        return total

    _finalize(root)

    def _count_sections(node: dict) -> int:
        return 1 + sum(_count_sections(c) for c in node["children"])

    doc = {
        "title": root["children"][0]["text"] if root["children"] else None,
        "n_pages": len(root["pages"]),
        "n_sections": _count_sections(root) - 1,
        "root": root,
    }
    return chunks, doc


def main() -> None:
    elements = list(read_jsonl(paths.INTERIM / "docling_elements.jsonl"))
    chunks, doc = build(elements)
    n_before = len(chunks)
    chunks = _post_split_missed_headings(chunks, min_chunk_chars=10_240)
    n_after = len(chunks)
    if n_after != n_before:
        print(f"[structural] post-split missed headings: "
              f"{n_before} -> {n_after} chunks", flush=True)

    write_jsonl(paths.CHUNKS / "structural.jsonl", chunks)
    write_json(paths.INTERIM / "document.json", doc)

    sizes = [c["n_chars"] for c in chunks]
    long_threshold = 1200
    stats = {
        "n_chunks": len(chunks),
        "n_table_chunks": sum(1 for c in chunks if c["contains_table"]),
        "n_long_chunks_over_1200c": sum(1 for s in sizes if s > long_threshold),
        "chars_total": sum(sizes),
        "chars_mean": round(sum(sizes) / max(1, len(sizes)), 1),
        "chars_min": min(sizes) if sizes else 0,
        "chars_max": max(sizes) if sizes else 0,
        "n_sections": doc["n_sections"],
        "document_tree_total_chars": doc["root"]["n_chars_total"],
    }
    write_json(paths.METRICS / "structural_stats.json", stats)
    print(f"[structural] {len(chunks)} chunks  sections={doc['n_sections']}  "
          f"mean={stats['chars_mean']}c  max={stats['chars_max']}c  "
          f"long(>{long_threshold})={stats['n_long_chunks_over_1200c']}  "
          f"tables={stats['n_table_chunks']}")


if __name__ == "__main__":
    main()
