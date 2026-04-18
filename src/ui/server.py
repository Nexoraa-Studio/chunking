"""SOP chunking pipeline dashboard and upload/run/download endpoints.

Pure-stdlib HTTP server serving:
  GET  /                     -> static/index.html
  GET  /api/state            -> per-stage status
  GET  /api/metrics          -> all outputs/metrics/*.json
  GET  /api/status           -> current pipeline run state (idle/running/done/error)
  GET  /api/log              -> SSE stream tailing logs/current.log
  GET  /api/chunks?src=…     -> paginated chunk viewer
  GET  /api/file?path=…      -> whitelisted artifact text preview
  GET  /figures/<name>       -> figure image bytes
  GET  /api/download         -> ZIP of coarse-DP chunks (txt per chunk)
  POST /api/upload           -> multipart PDF upload -> data/raw/ (replaces prior)
  POST /api/run              -> trigger pipeline in background thread

Auth: if env var API_KEY is set, all /api/* endpoints require header
`X-API-Key: <value>` (matching value) OR `?key=<value>` in the query.
If API_KEY is unset, the server runs open (local-dev mode).
"""
from __future__ import annotations

import cgi
import io
import json
import mimetypes
import os
import re
import shutil
import sys
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
CHUNKS_DIR = DATA / "chunks"
OUTPUTS = ROOT / "outputs"
LOGS = ROOT / "logs"
STATIC = Path(__file__).resolve().parent / "static"
LOG_FILE = LOGS / "current.log"

API_KEY = os.environ.get("API_KEY", "").strip()

# ---- pipeline run state (shared across request threads) ----
_run_lock = threading.Lock()
_run_state: dict = {
    "status": "idle",      # idle | running | done | error
    "message": "",
    "started_at": None,
    "finished_at": None,
    "pdf": None,
}

def _current_pdf_rel() -> list[str]:
    """Return the first PDF in data/raw/ as a relative-path list (for STAGES)."""
    if not RAW.exists():
        return []
    pdfs = sorted(RAW.glob("*.pdf"))
    return [str(pdfs[0].relative_to(ROOT))] if pdfs else []


STAGES = [
    {
        "id": "pdf",
        "label": "PDF",
        "description": "Source document (SOP) uploaded via /api/upload.",
        "artifacts_fn": _current_pdf_rel,
    },
    {
        "id": "extract",
        "label": "Docling extract",
        "description": "Layout-aware extraction: headings, paragraphs, lists, tables. OCR off, fast-mode tables.",
        "artifacts": [
            "data/interim/docling.md",
            "data/interim/docling_elements.jsonl",
            "data/interim/docling_stats.json",
        ],
    },
    {
        "id": "structural",
        "label": "Structural chunks",
        "description": "Group elements by heading. Numbering (1.1, 2.3) infers level. TOC tables dropped. Tables kept atomic.",
        "artifacts": [
            "data/chunks/structural.jsonl",
            "data/interim/document.json",
            "outputs/metrics/structural_stats.json",
        ],
    },
    {
        "id": "coarse",
        "label": "Coarse DP",
        "description": "Exact dynamic programming over structural units. cost = (1 − cohesion) + α·length_prior + β·heading_crossings. ~20 LLM-sized chunks.",
        "artifacts": [
            "data/chunks/semantic_coarse.jsonl",
            "data/interim/coarse_trace.json",
            "outputs/metrics/coarse_stats.json",
        ],
    },
    {
        "id": "eval",
        "label": "Evaluation",
        "description": "Chunk size, intra-cohesion, inter-separation, synthetic-query retrieval (R@1/3/5).",
        "artifacts": ["outputs/metrics/evaluation.json"],
    },
    {
        "id": "viz",
        "label": "Figures",
        "description": "Chunk size histograms, adjacent-distance curves, UMAP projections, sentence trajectory.",
        "artifacts": [
            "outputs/figures/chunk_sizes.png",
            "outputs/figures/adjacent_distance.png",
            "outputs/figures/umap_chunks.png",
            "outputs/figures/umap_trajectory.png",
        ],
    },
]

ALLOWED_FILE_PREFIXES = ("data/interim/", "data/chunks/", "outputs/metrics/", "logs/")

CHUNK_SOURCES = {
    "elements": "data/interim/docling_elements.jsonl",
    "structural": "data/chunks/structural.jsonl",
    "coarse": "data/chunks/semantic_coarse.jsonl",
}


def _read_jsonl(path: Path, offset: int, limit: int) -> tuple[list[dict], int]:
    """Streaming read with offset/limit. Returns (records, total_count)."""
    records: list[dict] = []
    total = 0
    if not path.exists():
        return records, 0
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            total += 1
            if i >= offset and len(records) < limit:
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
    return records, total


def _chunk_summary(rec: dict, src: str) -> dict:
    """Compact row for the list view."""
    text = rec.get("text", "") or ""
    preview = text.replace("\n", " ")[:160]
    row = {
        "id": rec.get("id", rec.get("idx")),
        "n_chars": rec.get("n_chars", len(text)),
        "preview": preview,
    }
    if src == "elements":
        row["type"] = rec.get("type")
        row["page"] = rec.get("page")
        row["heading_level"] = rec.get("heading_level")
    else:
        row["heading"] = rec.get("heading")
        row["heading_path"] = rec.get("heading_path")
        row["pages"] = rec.get("pages")
        row["contains_table"] = rec.get("contains_table", False)
    return row


def _stage_state(stage: dict) -> dict:
    found, missing = [], []
    latest_mtime = 0.0
    total_bytes = 0
    # stages may declare `artifacts` (static list) or `artifacts_fn` (callable
    # returning current list — lets the PDF stage reflect whatever was uploaded).
    artifacts = stage.get("artifacts")
    if artifacts is None and callable(stage.get("artifacts_fn")):
        artifacts = stage["artifacts_fn"]()
    artifacts = artifacts or []
    for rel in artifacts:
        p = ROOT / rel
        if p.exists():
            st = p.stat()
            found.append({"path": rel, "size": st.st_size, "mtime": st.st_mtime})
            latest_mtime = max(latest_mtime, st.st_mtime)
            total_bytes += st.st_size
        else:
            missing.append(rel)
    if not found:
        status = "pending"
    elif not missing:
        status = "done"
    else:
        status = "partial"
    return {
        "id": stage["id"],
        "label": stage["label"],
        "description": stage.get("description", ""),
        "status": status,
        "found": found,
        "missing": missing,
        "latest_mtime": latest_mtime,
        "total_bytes": total_bytes,
    }


def _collect_metrics() -> dict:
    out = {}
    mdir = OUTPUTS / "metrics"
    if mdir.exists():
        for p in sorted(mdir.glob("*.json")):
            try:
                out[p.name] = json.loads(p.read_text())
            except Exception as e:
                out[p.name] = {"_error": str(e)}
    return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # silence access log
        return

    def do_HEAD(self) -> None:  # noqa: N802
        # Reuse GET path but suppress body by temporarily swapping wfile.
        import io
        original = self.wfile
        self.wfile = io.BytesIO()
        try:
            self.do_GET()
        finally:
            buf = self.wfile
            self.wfile = original
            # Forward only headers (already written to buf); discard body.
            head = buf.getvalue().split(b"\r\n\r\n", 1)[0] + b"\r\n\r\n"
            try:
                original.write(head)
            except Exception:
                pass

    def _send_json(self, obj, code: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, code: int = 200, ctype: str = "text/plain; charset=utf-8") -> None:
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, data: bytes, ctype: str, headers: dict | None = None) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _check_auth(self, parsed) -> bool:
        """Return True if request is authorized. Index and static assets are
        always public; /api/* and /figures/* require the key when set."""
        if not API_KEY:
            return True
        path = parsed.path
        if path in ("/", "/index.html") or not (path.startswith("/api") or path.startswith("/figures")):
            return True
        provided = self.headers.get("X-API-Key", "")
        if not provided:
            qs = parse_qs(parsed.query)
            provided = qs.get("key", [""])[0]
        return provided == API_KEY

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if not self._check_auth(parsed):
            self._send_json({"error": "unauthorized"}, 401)
            return

        if path == "/" or path == "/index.html":
            idx = STATIC / "index.html"
            self._send_text(idx.read_text(encoding="utf-8"), ctype="text/html; charset=utf-8")
            return

        if path == "/api/state":
            stages = [_stage_state(s) for s in STAGES]
            pdfs = sorted(RAW.glob("*.pdf")) if RAW.exists() else []
            self._send_json({
                "stages": stages,
                "pdf": pdfs[0].name if pdfs else None,
                "pdf_size": pdfs[0].stat().st_size if pdfs else 0,
                "log_exists": LOG_FILE.exists(),
                "log_size": LOG_FILE.stat().st_size if LOG_FILE.exists() else 0,
                "auth_required": bool(API_KEY),
                "now": time.time(),
            })
            return

        if path == "/api/status":
            with _run_lock:
                self._send_json(dict(_run_state))
            return

        if path == "/api/metrics":
            self._send_json(_collect_metrics())
            return

        if path == "/api/download":
            self._handle_download()
            return

        if path == "/api/file":
            qs = parse_qs(parsed.query)
            rel = qs.get("path", [""])[0]
            if not any(rel.startswith(pfx) for pfx in ALLOWED_FILE_PREFIXES):
                self._send_text("forbidden", code=403)
                return
            p = ROOT / rel
            if not p.exists() or not p.is_file():
                self._send_text("not found", code=404)
                return
            max_bytes = int(qs.get("max", ["200000"])[0])
            data = p.read_bytes()[:max_bytes]
            try:
                text = data.decode("utf-8", errors="replace")
            except Exception:
                text = "<binary>"
            self._send_text(text)
            return

        if path.startswith("/figures/"):
            name = path[len("/figures/"):]
            p = OUTPUTS / "figures" / name
            if not p.exists():
                self._send_text("not found", code=404)
                return
            ctype, _ = mimetypes.guess_type(str(p))
            self._send_bytes(p.read_bytes(), ctype or "application/octet-stream")
            return

        if path == "/api/chunks":
            qs = parse_qs(parsed.query)
            src = qs.get("src", ["elements"])[0]
            if src not in CHUNK_SOURCES:
                self._send_json({"error": f"unknown src; valid: {list(CHUNK_SOURCES)}"}, code=400)
                return
            file_path = ROOT / CHUNK_SOURCES[src]
            offset = int(qs.get("offset", ["0"])[0])
            limit = min(int(qs.get("limit", ["50"])[0]), 500)
            full_id = qs.get("id", [None])[0]
            records, total = _read_jsonl(file_path, offset, limit if full_id is None else 10 ** 9)
            if full_id is not None:
                # return the single matching record with full text
                try:
                    want = int(full_id)
                except ValueError:
                    want = full_id
                match = None
                for r in records:
                    rid = r.get("id", r.get("idx"))
                    if rid == want:
                        match = r
                        break
                self._send_json({"src": src, "record": match})
                return
            self._send_json({
                "src": src,
                "file": CHUNK_SOURCES[src],
                "exists": file_path.exists(),
                "total": total,
                "offset": offset,
                "limit": limit,
                "rows": [_chunk_summary(r, src) for r in records],
            })
            return

        if path == "/api/log":
            self._stream_log()
            return

        self._send_text("not found", code=404)

    def _stream_log(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        LOGS.mkdir(parents=True, exist_ok=True)
        if not LOG_FILE.exists():
            LOG_FILE.touch()

        pos = 0
        last_ping = time.time()
        try:
            while True:
                size = LOG_FILE.stat().st_size
                if size < pos:
                    pos = 0  # file truncated/rotated
                if size > pos:
                    with LOG_FILE.open("rb") as f:
                        f.seek(pos)
                        chunk = f.read(size - pos).decode("utf-8", errors="replace")
                        pos = size
                    for line in chunk.splitlines():
                        payload = json.dumps({"line": line, "t": time.time()})
                        self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                elif time.time() - last_ping > 15:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    last_ping = time.time()
                time.sleep(0.4)
        except (BrokenPipeError, ConnectionResetError):
            return


    # ---- POST routing ----
    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if not self._check_auth(parsed):
            self._send_json({"error": "unauthorized"}, 401)
            return
        path = parsed.path
        if path == "/api/upload":
            self._handle_upload()
            return
        if path == "/api/run":
            self._handle_run()
            return
        self._send_text("not found", 404)

    # ---- upload a new PDF (replaces previous, clears downstream artifacts) ----
    def _handle_upload(self) -> None:
        ctype_hdr = self.headers.get("Content-Type", "")
        ctype, pdict = cgi.parse_header(ctype_hdr)
        if ctype != "multipart/form-data":
            self._send_json({"error": "expected multipart/form-data"}, 400)
            return
        if "boundary" in pdict:
            pdict["boundary"] = pdict["boundary"].encode()
        pdict["CONTENT-LENGTH"] = int(self.headers.get("Content-Length", 0))
        fs = cgi.FieldStorage(
            fp=self.rfile, headers=self.headers,
            environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": ctype_hdr},
            keep_blank_values=True,
        )
        if "file" not in fs:
            self._send_json({"error": "no 'file' field in form"}, 400)
            return
        file_item = fs["file"]
        if not getattr(file_item, "file", None) or not file_item.filename:
            self._send_json({"error": "empty file"}, 400)
            return
        filename = os.path.basename(file_item.filename)
        if not filename.lower().endswith(".pdf"):
            self._send_json({"error": "only .pdf accepted"}, 400)
            return
        RAW.mkdir(parents=True, exist_ok=True)
        # replace any prior PDF
        for old in RAW.glob("*.pdf"):
            old.unlink()
        target = RAW / filename
        with target.open("wb") as out:
            shutil.copyfileobj(file_item.file, out)
        size = target.stat().st_size
        self._send_json({"ok": True, "filename": filename, "size": size})

    # ---- trigger the pipeline in a background thread ----
    def _handle_run(self) -> None:
        with _run_lock:
            if _run_state["status"] == "running":
                self._send_json({"error": "pipeline already running",
                                 "state": dict(_run_state)}, 409)
                return
            pdfs = sorted(RAW.glob("*.pdf")) if RAW.exists() else []
            if not pdfs:
                self._send_json({"error": "no PDF uploaded yet"}, 400)
                return
            _run_state.update({"status": "running",
                               "message": "starting",
                               "started_at": time.time(),
                               "finished_at": None,
                               "pdf": pdfs[0].name})

        def worker():
            import subprocess
            LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with LOG_FILE.open("a", encoding="utf-8") as logf:
                logf.write(f"\n[run] triggered at {time.ctime()} pdf={pdfs[0].name}\n")
                logf.flush()
                proc = subprocess.Popen(
                    [sys.executable, "-u", str(ROOT / "run_pipeline.py")],
                    stdout=logf, stderr=subprocess.STDOUT, cwd=str(ROOT),
                )
                rc = proc.wait()
            with _run_lock:
                _run_state.update({
                    "status": "done" if rc == 0 else "error",
                    "message": f"exit {rc}",
                    "finished_at": time.time(),
                })

        threading.Thread(target=worker, daemon=True).start()
        self._send_json({"ok": True, "state": dict(_run_state)})

    # ---- download coarse chunks as a ZIP of txt files ----
    def _handle_download(self) -> None:
        coarse = CHUNKS_DIR / "semantic_coarse.jsonl"
        if not coarse.exists():
            self._send_json({"error": "no coarse chunks yet — run the pipeline"}, 404)
            return
        buf = io.BytesIO()
        count = 0
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for line in coarse.open("r", encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                hdr_txt = r.get("heading_first") or "chunk"
                safe = re.sub(r"[^A-Za-z0-9]+", "-", hdr_txt).strip("-")[:50] or "chunk"
                name = f"chunk_{r.get('id', count):02d}_{safe}.txt"
                header = (f"# chunk {r.get('id')} | pages {r.get('pages')} | "
                          f"structural ids {r.get('structural_ids')} | "
                          f"mean_internal_cosine {r.get('mean_internal_cosine')}\n"
                          f"# heading: {hdr_txt}\n\n")
                zf.writestr(name, header + (r.get("text") or ""))
                count += 1
        data = buf.getvalue()
        pdfs = sorted(RAW.glob("*.pdf")) if RAW.exists() else []
        base = (pdfs[0].stem if pdfs else "sop").replace(" ", "_")
        zipname = f"{base}_coarse_chunks.zip"
        self._send_bytes(
            data, "application/zip",
            headers={"Content-Disposition": f'attachment; filename="{zipname}"'},
        )


def main(host: str = "0.0.0.0", port: int = 8765) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer((host, port), Handler)
    auth_note = "API key enforced (X-API-Key header)" if API_KEY else "no auth (local-dev)"
    print(f"[dashboard] http://{host}:{port}  ({auth_note})  log={LOG_FILE}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    port = int(os.environ.get("DASH_PORT", 8765))
    main(port=port)
