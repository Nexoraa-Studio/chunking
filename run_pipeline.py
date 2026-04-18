#!/usr/bin/env python3
"""End-to-end pipeline orchestrator.

Reads the PDF currently in data/raw/, clears downstream artifacts, then runs:
    docling extract  ->  structural chunking  ->  coarse DP chunking

Called by the web UI's /api/run endpoint and available as a CLI:
    python3 run_pipeline.py
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline.utils import paths


def clear_artifacts() -> None:
    """Remove anything downstream of data/raw/ so a fresh PDF gets a clean run."""
    for d in (paths.INTERIM, paths.CHUNKS, paths.EMBEDDINGS, paths.FIGURES, paths.METRICS):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)


def run() -> int:
    pdf = paths.current_pdf()
    if pdf is None:
        print(f"[run] no PDF in {paths.RAW} — upload one first", flush=True)
        return 2

    print(f"[run] processing {pdf.name} ({pdf.stat().st_size:,} bytes)", flush=True)
    clear_artifacts()
    print("[run] cleared prior artifacts", flush=True)

    from src.pipeline.extract.docling_extract import main as run_extract
    from src.pipeline.chunk.structural import main as run_structural
    from src.pipeline.chunk.coarse_dp import main as run_coarse

    stages = [("extract", run_extract), ("structural", run_structural), ("coarse-DP", run_coarse)]
    for name, fn in stages:
        t0 = time.time()
        print(f"[run] === {name} ===", flush=True)
        try:
            fn()
        except Exception as e:
            print(f"[run] {name} FAILED: {type(e).__name__}: {e}", flush=True)
            return 1
        print(f"[run] {name} done in {time.time() - t0:.1f}s", flush=True)

    print("[run] pipeline complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(run())
