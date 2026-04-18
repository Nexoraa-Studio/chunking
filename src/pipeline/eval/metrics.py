"""Stage 5: Evaluate + compare chunking strategies.

Per strategy (structural, semantic-fine, coarse-DP):
  - size stats: count, mean, sigma, p10, median, p90, min, max
  - intra-chunk coherence: mean pairwise sentence cosine within each chunk,
    averaged over chunks
  - inter-chunk separation: mean cosine between adjacent chunk centroids
    (lower is better — more distinct adjacent chunks)
  - retrieval test: synthetic queries from headings; measure Recall@1/@3/@5
    using FAISS-flat cosine similarity over chunk centroids.

Queries are built from the structural headings (each heading = one synthetic
question). Ground truth is the chunk whose `element_idxs` / `structural_ids`
include the heading's owning section, or the chunk whose heading_path starts
with the query heading.

Writes outputs/metrics/evaluation.json.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.pipeline.chunk.sentence_split import split_sentences
from src.pipeline.embed.embedder import Embedder
from src.pipeline.utils import paths
from src.pipeline.utils.io import read_jsonl, write_json

STRATEGIES = [
    ("structural", "data/chunks/structural.jsonl"),
    ("coarse-DP", "data/chunks/semantic_coarse.jsonl"),
]


def _load(rel: str) -> list[dict]:
    p = paths.ROOT / rel
    return list(read_jsonl(p)) if p.exists() else []


def _chunk_centroid(text: str, embedder: Embedder) -> np.ndarray:
    sents = split_sentences(text) or [text]
    v = embedder.encode(sents).mean(axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _size_stats(sizes: np.ndarray) -> dict:
    if sizes.size == 0:
        return {}
    return {
        "count": int(sizes.size),
        "mean": float(sizes.mean()),
        "std": float(sizes.std()),
        "min": int(sizes.min()),
        "p10": float(np.percentile(sizes, 10)),
        "median": float(np.median(sizes)),
        "p90": float(np.percentile(sizes, 90)),
        "max": int(sizes.max()),
    }


def _intra_coherence(chunks: list[dict], embedder: Embedder) -> float:
    """Mean pairwise sentence cosine similarity inside each chunk, averaged
    over chunks. Higher = more topically tight."""
    scores = []
    for c in chunks:
        sents = split_sentences(c["text"])
        if len(sents) < 2:
            continue
        embs = embedder.encode(sents)
        s = embs.sum(axis=0)
        k = len(sents)
        mean_sim = (float(np.dot(s, s)) - k) / (k * (k - 1))
        scores.append(mean_sim)
    return float(np.mean(scores)) if scores else 0.0


def _inter_adjacent_separation(chunks: list[dict], embedder: Embedder) -> float:
    """Mean cosine similarity between adjacent chunk centroids.
    Lower = adjacent chunks are more distinct (good)."""
    if len(chunks) < 2:
        return 0.0
    cents = np.stack([_chunk_centroid(c["text"], embedder) for c in chunks])
    sims = [float(np.dot(cents[i], cents[i + 1])) for i in range(len(cents) - 1)]
    return float(np.mean(sims))


def _build_queries_and_gt(structural: list[dict]) -> tuple[list[str], list[int]]:
    """Use each structural chunk's heading (non-null) as a query; ground truth
    is that chunk's own text. Returns (queries, gt_structural_ids)."""
    queries, gts = [], []
    for c in structural:
        h = c.get("heading")
        if not h or len(h) < 4:
            continue
        queries.append(h)
        gts.append(c["id"])
    return queries, gts


def _retrieval_eval(queries: list[str], gt_structural_ids: list[int],
                    structural: list[dict], strategy_chunks: list[dict],
                    embedder: Embedder, ks: tuple[int, ...] = (1, 3, 5)) -> dict:
    """For each query: retrieve top-k chunks from `strategy_chunks` by cosine.
    A chunk is relevant if it covers (partially or fully) the ground-truth
    structural chunk's text.
    """
    if not strategy_chunks or not queries:
        return {f"recall@{k}": 0.0 for k in ks} | {"n_queries": 0}

    # centroids of strategy chunks
    cents = np.stack([_chunk_centroid(c["text"], embedder) for c in strategy_chunks])

    # relevance map: strategy_chunk_idx -> set of structural_ids covered
    covers: list[set[int]] = []
    for c in strategy_chunks:
        if "structural_ids" in c:  # coarse
            covers.append(set(c["structural_ids"]))
        elif "parent_structural_id" in c:  # semantic-fine
            covers.append({c["parent_structural_id"]})
        else:  # structural itself
            covers.append({c["id"]})

    q_embs = embedder.encode(queries)
    hits = {k: 0 for k in ks}
    per_query = []

    for q_emb, gt_sid, qtext in zip(q_embs, gt_structural_ids, queries):
        sims = cents @ q_emb  # cosine (both L2 normalized)
        order = np.argsort(-sims)  # descending
        # ranks at which a relevant chunk appears
        ranks = [rank for rank, idx in enumerate(order)
                 if gt_sid in covers[int(idx)]]
        first_rel = ranks[0] if ranks else None
        for k in ks:
            if first_rel is not None and first_rel < k:
                hits[k] += 1
        per_query.append({"q": qtext[:60], "gt_sid": gt_sid,
                          "first_rel_rank": first_rel})

    n = len(queries)
    return {f"recall@{k}": hits[k] / n for k in ks} | \
           {"n_queries": n, "per_query_sample": per_query[:5]}


def main() -> None:
    paths.METRICS.mkdir(parents=True, exist_ok=True)
    embedder = Embedder(cache_dir=paths.EMBEDDINGS)
    embedder.encode(["warmup."])

    structural = _load("data/chunks/structural.jsonl")
    queries, gts = _build_queries_and_gt(structural)
    print(f"[eval] {len(queries)} synthetic queries from structural headings",
          flush=True)

    report = {"strategies": {}, "n_queries": len(queries)}
    for name, rel in STRATEGIES:
        chunks = _load(rel)
        if not chunks:
            report["strategies"][name] = {"_error": f"missing {rel}"}
            print(f"[eval] {name}: missing, skipping", flush=True)
            continue

        t0 = time.time()
        sizes = np.array([c["n_chars"] for c in chunks])
        size_s = _size_stats(sizes)
        intra = _intra_coherence(chunks, embedder)
        inter = _inter_adjacent_separation(chunks, embedder)
        retrieval = _retrieval_eval(queries, gts, structural, chunks, embedder)
        elapsed = time.time() - t0

        report["strategies"][name] = {
            "size_stats": size_s,
            "intra_chunk_coherence_mean_sim": round(intra, 4),
            "inter_adjacent_similarity_mean": round(inter, 4),
            "retrieval": {k: round(v, 3) if isinstance(v, float) else v
                          for k, v in retrieval.items()},
            "elapsed_s": round(elapsed, 2),
        }
        print(f"[eval] {name:<14s} n={size_s['count']:3d} mean={size_s['mean']:.0f}c "
              f"intra={intra:.3f} inter_adj={inter:.3f} "
              f"R@1={retrieval['recall@1']:.2f} R@3={retrieval['recall@3']:.2f} "
              f"R@5={retrieval['recall@5']:.2f}  ({elapsed:.1f}s)", flush=True)

    write_json(paths.METRICS / "evaluation.json", report)
    print("[eval] wrote evaluation.json", flush=True)


if __name__ == "__main__":
    main()
