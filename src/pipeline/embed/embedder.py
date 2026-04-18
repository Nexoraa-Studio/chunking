"""Thin sentence-transformers wrapper with on-disk cache.

Cache key = sha1(model_name + text). Embeddings saved per batch to a single
numpy .npy file keyed by a parallel list of hashes (hash_list.json). Keeps
embed calls cheap across reruns.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


class Embedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 cache_dir: Path | None = None):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.cache_dir = cache_dir
        self._cache: dict[str, np.ndarray] = {}
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_cache()

    def _key(self, text: str) -> str:
        h = hashlib.sha1()
        h.update(self.model_name.encode())
        h.update(b"\x00")
        h.update(text.encode("utf-8"))
        return h.hexdigest()

    def _cache_paths(self) -> tuple[Path, Path]:
        safe = self.model_name.replace("/", "__")
        return (self.cache_dir / f"{safe}.npy", self.cache_dir / f"{safe}.keys.json")

    def _load_cache(self) -> None:
        vec_p, key_p = self._cache_paths()
        if vec_p.exists() and key_p.exists():
            vecs = np.load(vec_p)
            keys = json.loads(key_p.read_text())
            for k, v in zip(keys, vecs):
                self._cache[k] = v

    def _save_cache(self) -> None:
        if self.cache_dir is None:
            return
        vec_p, key_p = self._cache_paths()
        keys = list(self._cache.keys())
        if not keys:
            return
        vecs = np.stack([self._cache[k] for k in keys])
        np.save(vec_p, vecs)
        key_p.write_text(json.dumps(keys))

    def encode(self, texts: list[str], batch_size: int = 64,
               show_progress: bool = False) -> np.ndarray:
        keys = [self._key(t) for t in texts]
        missing = [(i, t, k) for i, (t, k) in enumerate(zip(texts, keys)) if k not in self._cache]
        if missing:
            to_encode = [m[1] for m in missing]
            vecs = self.model.encode(
                to_encode,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            for (i, _t, k), v in zip(missing, vecs):
                self._cache[k] = v.astype(np.float32)
            self._save_cache()
        return np.stack([self._cache[k] for k in keys])
