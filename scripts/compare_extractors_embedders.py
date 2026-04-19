#!/usr/bin/env python3
"""Compare extractors (docling vs pymupdf) x embedders (MiniLM vs Arctic).

Runs the full pipeline for every combination over every PDF in test/,
captures timing + chunk stats, and prints a markdown table.

Run from project root:   python3 scripts/compare_extractors_embedders.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.utils import paths

TEST_DIR = ROOT / "test"

EXTRACTORS = ["docling", "pymupdf"]
EMBEDDERS = [
    ("miniLM", "sentence-transformers/all-MiniLM-L6-v2"),
    ("arctic-s", "Snowflake/snowflake-arctic-embed-s"),
]


def run_pipeline(pdf: Path, extractor: str, embed_model: str) -> dict:
    # Reset data/raw/ to contain only this PDF.
    for p in paths.RAW.glob("*.pdf"):
        p.unlink()
    shutil.copy(pdf, paths.RAW / pdf.name)

    env = os.environ.copy()
    env["EXTRACTOR"] = extractor
    env["EMBED_MODEL"] = embed_model

    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-u", str(ROOT / "run_pipeline.py")],
        cwd=str(ROOT), env=env,
        capture_output=True, text=True, timeout=600,
    )
    total = time.time() - t0

    # Extract key numbers from logs + stat files.
    extract_s = _log_scrape(proc.stdout, r"\[run\] extract done in ([0-9.]+)")
    structural_s = _log_scrape(proc.stdout, r"\[run\] structural done in ([0-9.]+)")
    coarse_s = _log_scrape(proc.stdout, r"\[run\] coarse-DP done in ([0-9.]+)")
    embed_s = _log_scrape(proc.stdout, r"embed=([0-9.]+)s")
    dp_ms = _log_scrape(proc.stdout, r"dp=([0-9.]+)ms")

    ok = "pipeline complete" in proc.stdout
    result = {
        "pdf": pdf.name,
        "extractor": extractor,
        "embed_model": embed_model.split("/")[-1],
        "ok": ok,
        "total_s": round(total, 1),
        "extract_s": extract_s,
        "structural_s": structural_s,
        "coarse_s": coarse_s,
        "embed_s": embed_s,
        "dp_ms": dp_ms,
    }

    # Chunk counts / sizes.
    try:
        st = json.loads((paths.METRICS / "structural_stats.json").read_text())
        result["n_structural"] = st.get("n_chunks")
        result["struct_mean_c"] = st.get("chars_mean")
        result["struct_max_c"] = st.get("chars_max")
    except Exception:
        pass
    try:
        st = json.loads((paths.METRICS / "coarse_stats.json").read_text())
        result["n_coarse"] = st.get("n_chunks_output")
        result["coarse_mean_c"] = st.get("chars_mean")
        result["coarse_max_c"] = st.get("chars_max")
    except Exception:
        pass
    try:
        st = json.loads((paths.INTERIM / "docling_stats.json").read_text())
        result["extract_elements"] = st.get("n_elements")
        result["extract_tag"] = st.get("extractor", "docling")
    except Exception:
        pass
    if not ok:
        result["stderr_tail"] = proc.stderr[-400:] if proc.stderr else proc.stdout[-400:]
    return result


def _log_scrape(text: str, pattern: str):
    import re
    m = re.search(pattern, text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def main():
    pdfs = sorted(TEST_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"no PDFs in {TEST_DIR}")
        return 1
    print(f"{len(pdfs)} PDFs x {len(EXTRACTORS)} extractors x {len(EMBEDDERS)} embedders = "
          f"{len(pdfs) * len(EXTRACTORS) * len(EMBEDDERS)} runs")

    rows: list[dict] = []
    i = 0
    total_runs = len(pdfs) * len(EXTRACTORS) * len(EMBEDDERS)
    for pdf in pdfs:
        for ex in EXTRACTORS:
            for ename, emodel in EMBEDDERS:
                i += 1
                print(f"[{i}/{total_runs}] {pdf.name[:40]:<40} ex={ex:<8} em={ename}", flush=True)
                r = run_pipeline(pdf, ex, emodel)
                rows.append(r)
                short = (f"  total={r['total_s']}s  extract={r.get('extract_s')}s  "
                         f"n_struct={r.get('n_structural')}  n_coarse={r.get('n_coarse')}  "
                         f"ok={r['ok']}")
                print(short, flush=True)

    out_json = ROOT / "outputs" / "metrics" / "extractor_embedder_comparison.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2))
    print(f"\nsaved: {out_json}")

    # Markdown summary
    print("\n\n## Comparison matrix")
    header = ("| PDF | extractor | embedder | total | extract | embed | n_struct "
              "| n_coarse | coarse_mean | coarse_max | ok |")
    sep    = "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|"
    print(header)
    print(sep)
    for r in rows:
        pdf_short = r["pdf"].rsplit(".", 1)[0][:28]
        print(f"| {pdf_short} | {r['extractor']} | {r['embed_model']} "
              f"| {r.get('total_s')}s | {r.get('extract_s')}s | {r.get('embed_s')}s "
              f"| {r.get('n_structural')} | {r.get('n_coarse')} "
              f"| {r.get('coarse_mean_c')} | {r.get('coarse_max_c')} | {'✓' if r['ok'] else '✗'} |")


if __name__ == "__main__":
    sys.exit(main())
