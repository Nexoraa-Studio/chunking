"""Stage 3b: Coarse LLM-sized chunks via exact DP over structural heading chunks.

Per the research recommendation: for an ordered short sequence of structural
units, the right tool is a *globally* optimal contiguous partition, not local
thresholding and not clustering. We optimize
    total = sum over segments of cost(segment)
with
    cost(S) = (1 - mean_cosine_sim_within(S))          # cohesion
            + alpha * ((chars(S) - target_chars) / target_chars)^2
            + beta  * major_heading_crossings_inside(S)

via dynamic programming. At n ~= 54, this is cheap (O(n^2) with prefix sums).

Auto-calibration
----------------
target_k = round(total_chars / 2000) clamped to [12, 30]
target_chars = total_chars / target_k
alpha, beta = fixed small constants tuned to be commensurate with cohesion

Outputs
-------
data/chunks/semantic_coarse.jsonl  : final LLM-sized chunks
data/interim/coarse_trace.json     : segment partition, per-segment cost
outputs/metrics/coarse_stats.json  : size distribution and total cost
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
from src.pipeline.utils.io import read_jsonl, write_json, write_jsonl


def _chunk_embedding(text: str, embedder: Embedder) -> np.ndarray:
    """Mean-pool sentence embeddings, then re-normalize."""
    sents = split_sentences(text) or [text]
    embs = embedder.encode(sents)
    v = embs.mean(axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _major_number(num: str | None) -> str:
    if not num:
        return ""
    return num.split(".")[0]


def _boundary_flags(chunks: list[dict]) -> np.ndarray:
    """flags[k] = 1 iff transition from chunk k-1 to chunk k crosses a major
    heading (different top-level section number)."""
    n = len(chunks)
    flags = np.zeros(n, dtype=np.int32)
    for k in range(1, n):
        a = _major_number(chunks[k - 1].get("number"))
        b = _major_number(chunks[k].get("number"))
        if a and b and a != b:
            flags[k] = 1
    return flags


def _target_k(total_chars: int, target_chunk_chars: int = 2000,
              k_min: int = 12, k_max: int = 30) -> int:
    k = round(total_chars / target_chunk_chars)
    return max(k_min, min(k_max, k))


def dp_segment(chunks: list[dict], embeddings: np.ndarray, *,
               target_chars: float, alpha: float = 1.0, beta: float = 0.3,
               ) -> tuple[list[list[int]], np.ndarray, float]:
    """Exact DP on sequence [0..n). Returns (segments, dp_table, total_cost).

    embeddings: (n, d) L2-normalized.
    """
    n = len(chunks)
    if n == 0:
        return [], np.array([]), 0.0

    # prefix sum of embeddings -> fast segment-sum
    cum_emb = np.vstack([np.zeros(embeddings.shape[1]), np.cumsum(embeddings, axis=0)])
    chars = np.array([c["n_chars"] for c in chunks], dtype=np.int64)
    cum_chars = np.concatenate([[0], np.cumsum(chars)])
    bnd = _boundary_flags(chunks)
    cum_bnd = np.concatenate([[0], np.cumsum(bnd)])

    def seg_cost(i: int, j: int) -> float:
        """Inclusive [i..j], 0-indexed."""
        k = j - i + 1
        # cohesion
        if k == 1:
            cohesion = 0.0
        else:
            s = cum_emb[j + 1] - cum_emb[i]
            mean_sim = float((np.dot(s, s) - k) / (k * (k - 1)))
            cohesion = 1.0 - mean_sim
        # length prior
        L = int(cum_chars[j + 1] - cum_chars[i])
        length_pen = alpha * ((L - target_chars) / target_chars) ** 2
        # interior major-heading crossings (boundaries at positions i+1..j)
        cross = int(cum_bnd[j + 1] - cum_bnd[i + 1]) if i < j else 0
        return cohesion + length_pen + beta * cross

    INF = float("inf")
    dp = np.full(n + 1, INF, dtype=np.float64)
    dp[0] = 0.0
    parent = np.full(n + 1, -1, dtype=np.int64)
    seg_costs = np.full((n + 1, n + 1), np.nan)

    for j in range(1, n + 1):
        for i in range(1, j + 1):
            c = seg_cost(i - 1, j - 1)
            seg_costs[i][j] = c
            v = dp[i - 1] + c
            if v < dp[j]:
                dp[j] = v
                parent[j] = i - 1

    # reconstruct
    segments: list[list[int]] = []
    j = n
    while j > 0:
        i = int(parent[j])
        segments.append(list(range(i, j)))
        j = i
    segments.reverse()
    return segments, dp, float(dp[n])


def main() -> None:
    structural = list(read_jsonl(paths.CHUNKS / "structural.jsonl"))
    if not structural:
        print("[coarse] structural.jsonl missing — run stage 2 first", flush=True)
        return

    total_chars = sum(c["n_chars"] for c in structural)
    target_k = _target_k(total_chars)
    target_chars = total_chars / target_k
    print(f"[coarse] fingerprint: n_structural={len(structural)}  "
          f"total_chars={total_chars:,}  target_k={target_k}  "
          f"target_chars/chunk={target_chars:.0f}", flush=True)

    print(f"[coarse] embedding {len(structural)} structural chunks ...", flush=True)
    embedder = Embedder(cache_dir=paths.EMBEDDINGS)
    embedder.encode(["warmup."])
    t0 = time.time()
    embs = np.stack([_chunk_embedding(c["text"], embedder) for c in structural])
    t_emb = time.time() - t0

    print(f"[coarse] running DP (O(n^2), n={len(structural)}) ...", flush=True)
    t0 = time.time()
    segments, dp_table, total_cost = dp_segment(
        structural, embs, target_chars=target_chars, alpha=1.0, beta=0.3,
    )
    t_dp = time.time() - t0

    # Assemble output chunks
    out: list[dict] = []
    traces = []
    for seg_idx, seg in enumerate(segments):
        first, last = seg[0], seg[-1]
        text = "\n\n".join(structural[k]["text"] for k in seg)
        pages = sorted({p for k in seg for p in (structural[k].get("pages") or [])})
        heading_first = structural[first].get("heading_path") or structural[first].get("heading")
        heading_last = structural[last].get("heading_path") or structural[last].get("heading")
        # cohesion score for this segment (informational)
        k = len(seg)
        if k == 1:
            mean_sim = 1.0
        else:
            s = embs[first:last + 1].sum(axis=0)
            mean_sim = float((np.dot(s, s) - k) / (k * (k - 1)))
        rec = {
            "id": seg_idx,
            "strategy": "semantic_coarse",
            "structural_ids": seg,
            "n_structural": len(seg),
            "heading_first": heading_first,
            "heading_last": heading_last,
            "pages": pages,
            "n_chars": len(text),
            "mean_internal_cosine": round(mean_sim, 4),
            "contains_table": any(structural[k_]["contains_table"] for k_ in seg),
            "text": text,
        }
        out.append(rec)
        traces.append({"id": seg_idx, "range": [first, last],
                       "mean_internal_cosine": mean_sim,
                       "n_structural": len(seg), "n_chars": len(text)})

    write_jsonl(paths.CHUNKS / "semantic_coarse.jsonl", out)
    write_json(paths.INTERIM / "coarse_trace.json",
               {"target_k": target_k, "target_chars": target_chars,
                "total_cost": total_cost, "alpha": 1.0, "beta": 0.3,
                "segments": traces})

    sizes = [c["n_chars"] for c in out]
    stats = {
        "method": "exact DP on structural chunks (cohesion + length + heading penalty)",
        "n_structural_input": len(structural),
        "n_chunks_output": len(out),
        "target_k": target_k,
        "target_chars": round(target_chars, 1),
        "alpha": 1.0,
        "beta": 0.3,
        "total_cost": round(total_cost, 4),
        "chars_mean": round(sum(sizes) / max(1, len(sizes)), 1),
        "chars_min": min(sizes) if sizes else 0,
        "chars_max": max(sizes) if sizes else 0,
        "elapsed_embed_s": round(t_emb, 2),
        "elapsed_dp_s": round(t_dp, 4),
    }
    write_json(paths.METRICS / "coarse_stats.json", stats)
    print(f"[coarse] done: {len(out)} chunks (target {target_k})  "
          f"mean={stats['chars_mean']:.0f}c  range=[{stats['chars_min']},{stats['chars_max']}]  "
          f"total_cost={total_cost:.3f}  embed={t_emb:.1f}s dp={t_dp*1000:.1f}ms",
          flush=True)


if __name__ == "__main__":
    main()
