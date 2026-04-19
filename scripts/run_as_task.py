#!/usr/bin/env python3
"""Fargate run-task entrypoint for the serverless deployment.

Env vars (required):
  JOB_ID  -- unique job identifier (usually an S3 prefix)
  BUCKET  -- S3 bucket holding jobs/<JOB_ID>/input.pdf on entry,
             jobs/<JOB_ID>/chunks.zip and jobs/<JOB_ID>/status.json on exit
  REGION  -- AWS region (default ap-south-1)

Flow:
  1. Write status.json {"status":"running","started_at":..., "pdf":...}
  2. Download input.pdf from S3 -> data/raw/input.pdf
  3. run_pipeline.main() (pymupdf extract -> structural -> coarse-DP)
  4. Pack the coarse chunks into chunks.zip (same format as /api/download)
  5. Upload chunks.zip + final status.json to S3
  6. Exit 0 (Fargate task terminates)

Any unhandled exception is logged and written into status.json as
{"status":"error","message":...} so the Lambda API can surface it.
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
import time
import traceback
import zipfile
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _status_object(bucket: str, job_id: str):
    return f"jobs/{job_id}/status.json"


def _put_status(s3, bucket: str, job_id: str, state: dict) -> None:
    body = json.dumps(state, default=str).encode("utf-8")
    s3.put_object(
        Bucket=bucket, Key=_status_object(bucket, job_id),
        Body=body, ContentType="application/json",
        CacheControl="no-store",
    )


def _build_zip_from_coarse(coarse_jsonl: Path) -> tuple[bytes, int]:
    buf = io.BytesIO()
    n = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for line in coarse_jsonl.open("r", encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            hdr_txt = r.get("heading_first") or "chunk"
            safe = re.sub(r"[^A-Za-z0-9]+", "-", hdr_txt).strip("-")[:50] or "chunk"
            fname = f"chunk_{r.get('id', n):02d}_{safe}.txt"
            hdr = (f"# chunk {r.get('id')} | pages {r.get('pages')} | "
                   f"structural_ids {r.get('structural_ids')} | "
                   f"mean_internal_cosine {r.get('mean_internal_cosine')}\n"
                   f"# heading: {hdr_txt}\n\n")
            zf.writestr(fname, hdr + (r.get("text") or ""))
            n += 1
    return buf.getvalue(), n


def main() -> int:
    job_id = os.environ["JOB_ID"]
    bucket = os.environ["BUCKET"]
    region = os.environ.get("REGION", "ap-south-1")

    print(f"[task] job={job_id} bucket={bucket}", flush=True)
    s3 = boto3.client("s3", region_name=region)

    state: dict = {"job_id": job_id, "status": "starting", "started_at": time.time()}
    _put_status(s3, bucket, job_id, state)

    try:
        from src.pipeline.utils import paths

        # Clean slate
        if paths.RAW.exists():
            for p in paths.RAW.glob("*.pdf"):
                p.unlink()
        paths.RAW.mkdir(parents=True, exist_ok=True)

        # Download input.pdf from S3
        in_key = f"jobs/{job_id}/input.pdf"
        # Preserve the original filename if it was stored alongside as meta.json
        meta_key = f"jobs/{job_id}/meta.json"
        original_name = "input.pdf"
        try:
            obj = s3.get_object(Bucket=bucket, Key=meta_key)
            meta = json.loads(obj["Body"].read())
            original_name = meta.get("original_name", "input.pdf")
        except Exception:
            pass

        local_pdf = paths.RAW / original_name
        print(f"[task] downloading s3://{bucket}/{in_key} -> {local_pdf.name}", flush=True)
        s3.download_file(bucket, in_key, str(local_pdf))

        state.update({"status": "running", "pdf": original_name,
                      "pdf_size": local_pdf.stat().st_size})
        _put_status(s3, bucket, job_id, state)

        # Run the pipeline in-process (same code as run_pipeline.py)
        from src.pipeline.extract.pymupdf_extract import main as run_extract
        from src.pipeline.chunk.structural import main as run_structural
        from src.pipeline.chunk.coarse_dp import main as run_coarse

        import shutil
        for d in (paths.INTERIM, paths.CHUNKS, paths.EMBEDDINGS):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        print("[task] === extract ===", flush=True)
        run_extract()
        print("[task] === structural ===", flush=True)
        run_structural()
        print("[task] === coarse-DP ===", flush=True)
        run_coarse()
        elapsed = time.time() - t0
        print(f"[task] pipeline done in {elapsed:.1f}s", flush=True)

        # Pack chunks.zip
        coarse_path = paths.CHUNKS / "semantic_coarse.jsonl"
        if not coarse_path.exists():
            raise RuntimeError("semantic_coarse.jsonl not produced")
        zip_bytes, n_chunks = _build_zip_from_coarse(coarse_path)
        out_key = f"jobs/{job_id}/chunks.zip"
        print(f"[task] uploading {len(zip_bytes)} bytes to s3://{bucket}/{out_key}",
              flush=True)
        s3.put_object(Bucket=bucket, Key=out_key, Body=zip_bytes,
                      ContentType="application/zip", CacheControl="no-store")

        # Also upload intermediate artifacts so the dashboard UI can browse them.
        def _upload(local: Path, key: str, ctype: str) -> None:
            if not local.exists():
                return
            s3.upload_file(str(local), bucket, key,
                           ExtraArgs={"ContentType": ctype,
                                      "CacheControl": "no-store"})
            print(f"[task] +s3://{bucket}/{key} ({local.stat().st_size}B)", flush=True)

        _upload(paths.CHUNKS / "structural.jsonl",
                f"jobs/{job_id}/structural.jsonl", "application/x-ndjson")
        _upload(paths.CHUNKS / "semantic_coarse.jsonl",
                f"jobs/{job_id}/coarse.jsonl", "application/x-ndjson")
        _upload(paths.INTERIM / "docling_elements.jsonl",
                f"jobs/{job_id}/elements.jsonl", "application/x-ndjson")

        if paths.METRICS.exists():
            for mfile in sorted(paths.METRICS.glob("*.json")):
                _upload(mfile, f"jobs/{job_id}/metrics/{mfile.name}",
                        "application/json")

        state.update({
            "status": "done",
            "finished_at": time.time(),
            "elapsed_s": round(elapsed, 2),
            "n_chunks": n_chunks,
            "zip_key": out_key,
            "zip_size": len(zip_bytes),
        })
        _put_status(s3, bucket, job_id, state)
        print(f"[task] done: {n_chunks} chunks in {elapsed:.1f}s", flush=True)
        return 0
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[task] ERROR: {e}\n{tb}", flush=True)
        state.update({"status": "error", "message": f"{type(e).__name__}: {e}",
                      "traceback": tb, "finished_at": time.time()})
        try:
            _put_status(s3, bucket, job_id, state)
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
