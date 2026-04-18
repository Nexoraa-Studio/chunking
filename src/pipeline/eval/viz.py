"""Stage 4: Visualizations for chunk quality.

Produces four plots in outputs/figures/ (all PNG):

  chunk_sizes.png         — size distribution per strategy (overlaid histograms)
  adjacent_distance.png   — Kamradt-style curve: cosine distance between
                             adjacent chunks across position, per strategy
  umap_chunks.png         — 2D UMAP of chunk embeddings, colored by chunk id,
                             one subplot per strategy
  umap_trajectory.png     — sentence-level UMAP; lines connect sentences in
                             reading order; colored by coarse chunk membership

All use matplotlib only, no seaborn. UMAP is used strictly for visualization.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.pipeline.chunk.sentence_split import split_sentences
from src.pipeline.embed.embedder import Embedder
from src.pipeline.utils import paths
from src.pipeline.utils.io import read_jsonl

STRATEGIES = [
    ("structural", "data/chunks/structural.jsonl", "#7cc4ff"),
    ("coarse-DP", "data/chunks/semantic_coarse.jsonl", "#ffb86b"),
]


def _load(rel: str) -> list[dict]:
    p = paths.ROOT / rel
    return list(read_jsonl(p)) if p.exists() else []


def _chunk_embedding(text: str, embedder: Embedder) -> np.ndarray:
    sents = split_sentences(text) or [text]
    v = embedder.encode(sents).mean(axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# ---- plot 1: chunk sizes ---------------------------------------------

def plot_sizes() -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=130)
    for name, rel, color in STRATEGIES:
        chunks = _load(rel)
        if not chunks:
            continue
        sizes = np.array([c["n_chars"] for c in chunks])
        ax.hist(sizes, bins=20, alpha=0.55, color=color,
                label=f"{name} (n={len(sizes)}, mean={sizes.mean():.0f})",
                edgecolor="black", linewidth=0.4)
    ax.set_xlabel("chunk size (chars)")
    ax.set_ylabel("# chunks")
    ax.set_title("chunk size distribution")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = paths.FIGURES / "chunk_sizes.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] wrote {out.name}", flush=True)


# ---- plot 2: adjacent cosine distance curve --------------------------

def plot_adjacent_distance(embedder: Embedder) -> None:
    fig, axes = plt.subplots(len(STRATEGIES), 1, figsize=(10, 8.5), dpi=130,
                             sharex=False)
    for ax, (name, rel, color) in zip(axes, STRATEGIES):
        chunks = _load(rel)
        if not chunks:
            ax.set_title(f"{name} (no file)")
            ax.axis("off")
            continue
        embs = np.stack([_chunk_embedding(c["text"], embedder) for c in chunks])
        dists = np.array([1.0 - float(np.dot(embs[i], embs[i + 1]))
                          for i in range(len(embs) - 1)])
        x = np.arange(len(dists))
        thr = float(np.percentile(dists, 90))
        ax.plot(x, dists, color=color, lw=1.2, marker="o", markersize=3)
        ax.axhline(thr, color="red", ls="--", lw=0.8, alpha=0.6,
                   label=f"P90 = {thr:.3f}")
        peaks = np.where(dists >= thr)[0]
        ax.scatter(peaks, dists[peaks], color="red", s=22, zorder=5,
                   label=f"P90 exceed ({len(peaks)})")
        ax.set_title(f"{name} — adjacent cosine distance (n_chunks={len(chunks)})")
        ax.set_xlabel("chunk boundary position")
        ax.set_ylabel("cos distance")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    out = paths.FIGURES / "adjacent_distance.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] wrote {out.name}", flush=True)


# ---- plot 3: UMAP of chunks per strategy -----------------------------

def plot_umap_chunks(embedder: Embedder) -> None:
    import umap
    fig, axes = plt.subplots(1, len(STRATEGIES), figsize=(15, 5.2), dpi=130)
    for ax, (name, rel, _) in zip(axes, STRATEGIES):
        chunks = _load(rel)
        if len(chunks) < 3:
            ax.set_title(f"{name} (too few chunks)")
            ax.axis("off")
            continue
        embs = np.stack([_chunk_embedding(c["text"], embedder) for c in chunks])
        n_nb = max(2, min(15, len(chunks) - 1))
        reducer = umap.UMAP(n_components=2, n_neighbors=n_nb, min_dist=0.1,
                            metric="cosine", random_state=42, n_jobs=1)
        xy = reducer.fit_transform(embs)
        ids = np.arange(len(chunks))
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=ids, cmap="viridis",
                        s=60 if name == "coarse-DP" else 28, edgecolors="white",
                        linewidth=0.5)
        # connect in reading order with a faint trail
        ax.plot(xy[:, 0], xy[:, 1], color="gray", alpha=0.25, lw=0.6, zorder=0)
        # annotate coarse chunks with their id (since there are only ~20)
        if name == "coarse-DP":
            for i, (x, y) in enumerate(xy):
                ax.text(x, y, str(i), fontsize=8, ha="center", va="center",
                        color="white", fontweight="bold")
        ax.set_title(f"{name} (n={len(chunks)})")
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        ax.grid(alpha=0.2)
    fig.suptitle("UMAP of chunk embeddings (colored by reading order)", y=1.02)
    fig.tight_layout()
    out = paths.FIGURES / "umap_chunks.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] wrote {out.name}", flush=True)


# ---- plot 4: sentence trajectory colored by coarse chunk -------------

def plot_sentence_trajectory(embedder: Embedder) -> None:
    import umap
    coarse = _load("data/chunks/semantic_coarse.jsonl")
    if not coarse:
        print("[viz] skip trajectory (no coarse file)", flush=True)
        return

    all_sents: list[str] = []
    chunk_ids: list[int] = []
    for c in coarse:
        sents = split_sentences(c["text"])
        if not sents:
            continue
        all_sents.extend(sents)
        chunk_ids.extend([c["id"]] * len(sents))
    if len(all_sents) < 5:
        print("[viz] skip trajectory (too few sentences)", flush=True)
        return

    embs = embedder.encode(all_sents)
    n_nb = max(2, min(15, len(all_sents) - 1))
    reducer = umap.UMAP(n_components=2, n_neighbors=n_nb, min_dist=0.1,
                        metric="cosine", random_state=42, n_jobs=1)
    xy = reducer.fit_transform(embs)
    chunk_ids_arr = np.array(chunk_ids)

    fig, ax = plt.subplots(figsize=(11, 7), dpi=130)
    # reading-order trajectory
    ax.plot(xy[:, 0], xy[:, 1], color="black", alpha=0.15, lw=0.6, zorder=0)
    # scatter colored by coarse chunk
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=chunk_ids_arr, cmap="tab20",
                    s=18, edgecolors="white", linewidth=0.3)
    # mark sentence-0 of each chunk with a ring
    for cid in np.unique(chunk_ids_arr):
        idxs = np.where(chunk_ids_arr == cid)[0]
        first = idxs[0]
        ax.scatter(xy[first, 0], xy[first, 1], facecolors="none",
                   edgecolors="red", s=120, lw=1.1, zorder=3)
        ax.text(xy[first, 0], xy[first, 1], f" {cid}", fontsize=8,
                color="red", fontweight="bold", zorder=4)
    ax.set_title(f"sentence trajectory (n={len(all_sents)} sents) — colored by coarse chunk\n"
                 f"red rings = first sentence of each chunk")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out = paths.FIGURES / "umap_trajectory.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] wrote {out.name}", flush=True)


def main() -> None:
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    embedder = Embedder(cache_dir=paths.EMBEDDINGS)
    embedder.encode(["warmup."])
    plot_sizes()
    plot_adjacent_distance(embedder)
    plot_umap_chunks(embedder)
    plot_sentence_trajectory(embedder)
    print("[viz] done", flush=True)


if __name__ == "__main__":
    main()
