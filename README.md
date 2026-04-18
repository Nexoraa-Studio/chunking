# SOP chunking pipeline

Semantic chunking of structured PDF documents (SOPs, process guides, compliance
manuals) into **LLM-sized, reading-order-preserving chunks** suitable for
retrieval and downstream LLM reasoning. Pure Python, CPU only, no LangChain.

## What it does

Upload a PDF → get back ~15–25 coherent chunks, each a self-contained unit you
can hand to an LLM as a sub-task. Each chunk carries its section heading path
and page numbers.

## Pipeline

```
PDF  →  docling extract  →  structural chunks  →  coarse DP chunks  →  evaluate / visualize
```

1. **Docling extract** — layout-aware extraction. Fast-mode tables, OCR off
   (text-layer PDFs only). Emits paragraphs, list items, tables as JSONL + a
   markdown dump. ~20 s for a 20-page doc on CPU.
2. **Structural chunking** — groups elements under their heading. Heading
   levels are inferred from numbering prefixes (`1.`, `1.1`, `2.3.1`).
   Table-of-contents tables are dropped; real tables kept atomic.
3. **Coarse DP chunking** — *the headline method*. Exact dynamic programming
   over the ordered structural units:

   ```
   cost(S) = (1 − mean_cosine_sim_within(S))                        # cohesion
           + α · ((chars(S) − target_chars) / target_chars)²        # length prior
           + β · count_major_heading_crossings_inside(S)            # soft heading wall
   ```

   `target_k` auto-calibrates from `total_chars / ~2000` clamped to `[12, 30]`.
   O(n²) DP runs in ~2 ms at n=54 on CPU.

4. **Evaluation** — size distribution, intra-chunk coherence, inter-adjacent
   similarity, synthetic-query retrieval (Recall@1/3/5) using MiniLM embeddings
   over chunk centroids.
5. **Visualization** — chunk-size histograms, adjacent-distance curves,
   UMAP 2-D projections, sentence trajectory colored by chunk membership.

### Why DP over agglomerative / UMAP-HDBSCAN

- **Order-preserving** by construction (no sentence-3-merged-with-sentence-17).
- **Global optimum** — not greedy like agglomerative.
- **Auto-calibrating** via length prior instead of percentile thresholds.
- **Honest about doc structure** — UMAP+HDBSCAN was evaluated and found inert
  on short (~5–15 sentence) sections: HDBSCAN collapsed every long section to
  a single cluster. The DP cohesion term plays the same role without the
  fragility.

See `src/pipeline/chunk/coarse_dp.py` for the algorithm.

## Evaluation results on a sample 23-page SOP

Ground truth: one query per structural heading (54 queries); a chunk is
relevant if it covers the heading's section.

| strategy | n | mean_c | intra | inter-adj | R@1 | R@3 | **R@5** |
|---|---:|---:|---:|---:|---:|---:|---:|
| structural (one-per-heading) | 54 | 750 | 0.445 | 0.559 | 0.44 | 0.67 | 0.69 |
| **coarse-DP** | **21** | **1933** | 0.395 | 0.658 | 0.44 | **0.72** | **0.87** |

Coarse-DP wins R@5 by 18 points because each chunk spans a whole topic — the
correct chunk reliably appears in the retrieved top-5.

## Project layout

```
SOP_18042027_v2/
├── src/
│   ├── pipeline/
│   │   ├── extract/       # docling_extract.py, pypdf_extract.py
│   │   ├── chunk/         # structural.py, coarse_dp.py, sentence_split.py
│   │   ├── embed/         # embedder.py  (MiniLM wrapper + on-disk cache)
│   │   ├── eval/          # metrics.py, viz.py
│   │   └── utils/         # paths, io helpers
│   └── ui/
│       ├── server.py      # stdlib HTTP dashboard + upload/run/download API
│       └── static/
│           └── index.html # flow chart, tabs, chunk viewer, SSE live log
├── run_pipeline.py        # end-to-end orchestrator
├── requirements.txt
├── Dockerfile             # CPU-only image on port 17842
└── README.md
```

## Running locally (no Docker)

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# CLI pipeline over a PDF in data/raw/
python run_pipeline.py

# Web UI (upload / run / download)
DASH_PORT=17842 python -m src.ui.server
# open http://localhost:17842
```

The UI exposes:

- `POST /api/upload` — multipart PDF upload (replaces prior)
- `POST /api/run` — kick off the pipeline in a background thread
- `GET  /api/status` — poll run state
- `GET  /api/download` — ZIP of coarse-DP chunks as `chunk_NN_<heading>.txt`
- `GET  /api/state` — per-stage artifact status
- `GET  /api/chunks?src=structural|coarse|elements` — paginated chunk viewer
- `GET  /api/log` — server-sent events stream of `logs/current.log`

## Running with Docker

```bash
docker build -t sop-chunker .
docker run -p 17842:17842 \
  -e API_KEY='some-long-random-string' \
  -v "$(pwd)/data:/app/data" \
  sop-chunker
```

Omit `API_KEY` for local-open mode. Set it in any public deploy.

## Tech choices (pinned by intent)

| | Choice |
|---|---|
| Language | Python 3.11 (tested on 3.9 too) |
| PDF structure | [docling](https://github.com/docling-project/docling) — OCR off, fast tables |
| Sentence split | nltk punkt |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-D, L2-normalized) |
| Indexing | (planned) FAISS flat + HNSW |
| Viz | matplotlib + umap-learn |
| Web UI | pure Python stdlib `http.server` (no Flask, no LangChain) |

## Ports and env vars

| var | default | meaning |
|---|---|---|
| `DASH_PORT` | `8765` | server port; `17842` recommended for Docker / public deploy |
| `API_KEY` | *(unset)* | if set, all `/api/*` and `/figures/*` require `X-API-Key` header or `?key=` |
| `DOCLING_THREADS` | CPU count | threads for docling's layout/table models |
| `DOCLING_TABLES` | `1` | set to `0` to disable table structure extraction |

## Roadmap

- FAISS index over coarse chunks for fast retrieval
- Optional PELT / `ruptures` comparator (change-point detection)
- AWS App Runner deploy template
