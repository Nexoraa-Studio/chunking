"""Stage 1a: Extract PDF with docling.

Produces three artifacts in data/interim/:
  - docling.md          : full markdown dump (with tables as pipe-tables)
  - docling_elements.jsonl : one record per document element (text/heading/table/list/figure)
  - docling_stats.json  : coverage + timing

Each element record has: {idx, page, type, level, text, parent_heading}
This is the canonical structured representation that later stages chunk over.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.pipeline.utils import paths
from src.pipeline.utils.io import write_json, write_jsonl


def _element_records(doc) -> list[dict]:
    """Walk docling document and emit a flat sequence of element dicts.

    docling's DoclingDocument exposes .iterate_items() yielding (item, level).
    Item subclasses include TextItem, SectionHeaderItem, TableItem, ListItem, PictureItem.
    """
    records: list[dict] = []
    current_heading: str | None = None
    current_heading_level: int = 0

    for idx, (item, level) in enumerate(doc.iterate_items()):
        cls = item.__class__.__name__
        page = None
        prov = getattr(item, "prov", None)
        if prov:
            try:
                page = prov[0].page_no
            except Exception:
                page = None

        rec: dict = {
            "idx": idx,
            "page": page,
            "type": cls,
            "level": level,
            "parent_heading": current_heading,
        }

        if cls == "SectionHeaderItem":
            text = getattr(item, "text", "") or ""
            rec["text"] = text
            rec["heading_level"] = getattr(item, "level", level)
            current_heading = text
            current_heading_level = rec["heading_level"]
        elif cls in ("TextItem", "ListItem"):
            rec["text"] = getattr(item, "text", "") or ""
        elif cls == "TableItem":
            try:
                rec["text"] = item.export_to_markdown()
            except Exception:
                rec["text"] = ""
            try:
                rec["n_rows"] = item.data.num_rows
                rec["n_cols"] = item.data.num_cols
            except Exception:
                pass
        elif cls == "PictureItem":
            rec["text"] = getattr(item, "caption_text", "") or ""
        else:
            rec["text"] = getattr(item, "text", "") or ""

        records.append(rec)
    return records


def _build_fast_converter(num_threads: int, do_tables: bool):
    """Fast CPU pipeline for text-layer PDFs.

    OCR off, table structure on (but FAST mode), no page images, reduced scale,
    and all threads wired to the accelerator. ~3-5x faster than defaults on
    text-layer PDFs with acceptable fidelity for downstream semantic chunking.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        AcceleratorDevice,
        AcceleratorOptions,
        PdfPipelineOptions,
        TableFormerMode,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.do_table_structure = do_tables
    if do_tables:
        opts.table_structure_options.mode = TableFormerMode.FAST
        opts.table_structure_options.do_cell_matching = True
    opts.generate_page_images = False
    opts.generate_picture_images = False
    opts.images_scale = 1.0
    opts.accelerator_options = AcceleratorOptions(
        num_threads=num_threads,
        device=AcceleratorDevice.CPU,
    )

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def main() -> None:
    import os

    num_threads = int(os.environ.get("DOCLING_THREADS", os.cpu_count() or 4))
    do_tables = os.environ.get("DOCLING_TABLES", "1") not in ("0", "false", "False")

    paths.INTERIM.mkdir(parents=True, exist_ok=True)

    pdf_path = paths.current_pdf()
    if pdf_path is None:
        raise FileNotFoundError(f"No PDF found in {paths.RAW} — upload one first.")

    print(f"[docling] converting {pdf_path.name} "
          f"(threads={num_threads}, ocr=off, tables={'FAST' if do_tables else 'off'}) ...",
          flush=True)
    t0 = time.time()
    converter = _build_fast_converter(num_threads, do_tables)
    result = converter.convert(str(pdf_path))
    doc = result.document
    elapsed = time.time() - t0

    md_path = paths.INTERIM / "docling.md"
    md_path.write_text(doc.export_to_markdown(), encoding="utf-8")

    records = _element_records(doc)
    el_path = paths.INTERIM / "docling_elements.jsonl"
    n = write_jsonl(el_path, records)

    type_counts: dict[str, int] = {}
    total_chars = 0
    for r in records:
        type_counts[r["type"]] = type_counts.get(r["type"], 0) + 1
        total_chars += len(r.get("text", "") or "")

    pages = {r["page"] for r in records if r["page"] is not None}
    stats = {
        "elapsed_seconds": round(elapsed, 2),
        "n_elements": n,
        "n_pages_seen": len(pages),
        "markdown_chars": len(md_path.read_text(encoding="utf-8")),
        "element_text_chars": total_chars,
        "type_counts": type_counts,
    }
    write_json(paths.INTERIM / "docling_stats.json", stats)

    print(f"[docling] done in {elapsed:.1f}s")
    print(f"[docling] elements: {n}  pages: {len(pages)}")
    print(f"[docling] type counts: {type_counts}")
    print(f"[docling] wrote {md_path.name}, {el_path.name}, docling_stats.json")


if __name__ == "__main__":
    main()
