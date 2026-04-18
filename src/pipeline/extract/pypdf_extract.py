"""Stage 1b: Plain-text baseline extraction with pypdf.

Used only as a control to compare coverage vs docling.
Writes data/interim/pypdf.txt and data/interim/pypdf_stats.json.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.pipeline.utils import paths
from src.pipeline.utils.io import write_json


def main() -> None:
    from pypdf import PdfReader

    paths.INTERIM.mkdir(parents=True, exist_ok=True)

    pdf_path = paths.current_pdf()
    if pdf_path is None:
        raise FileNotFoundError(f"No PDF found in {paths.RAW}")
    t0 = time.time()
    reader = PdfReader(str(pdf_path))
    pages_text: list[str] = []
    for page in reader.pages:
        pages_text.append(page.extract_text() or "")
    elapsed = time.time() - t0

    full = "\n\n".join(pages_text)
    out_path = paths.INTERIM / "pypdf.txt"
    out_path.write_text(full, encoding="utf-8")

    stats = {
        "elapsed_seconds": round(elapsed, 2),
        "n_pages": len(pages_text),
        "total_chars": len(full),
        "mean_chars_per_page": round(len(full) / max(1, len(pages_text)), 1),
    }
    write_json(paths.INTERIM / "pypdf_stats.json", stats)
    print(f"[pypdf] {len(pages_text)} pages, {len(full):,} chars, {elapsed:.2f}s")


if __name__ == "__main__":
    main()
