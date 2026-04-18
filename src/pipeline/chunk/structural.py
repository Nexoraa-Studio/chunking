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


def _infer_level(heading_text: str) -> tuple[int | None, str | None]:
    """Return (level, number) inferred from heading text, or (None, None)."""
    m = NUMBER_PREFIX.match(heading_text or "")
    if not m:
        return None, None
    number = m.group(1)
    level = number.count(".") + 1
    return level, number


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
