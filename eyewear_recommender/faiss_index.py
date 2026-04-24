"""
FAISS flat index on L2-normalized CLIP image embeddings: inner product = cosine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import faiss
import numpy as np

__all__ = ["build_index", "search", "FAISSIndex"]


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / n).astype(np.float32)


class FAISSIndex:
    """IndexFlatIP over normalized vectors (cosine via dot product)."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._index: Any = faiss.IndexFlatIP(dim)

    def add(self, vectors: np.ndarray) -> None:
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        v = _l2_normalize_rows(vectors)
        self._index.add(v)

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> tuple[np.ndarray, np.ndarray]:
        if query_embedding.ndim == 1:
            q = query_embedding.reshape(1, -1).astype(np.float32)
        else:
            q = query_embedding.astype(np.float32)
        q = _l2_normalize_rows(q)
        ntot = self.ntotal()
        if ntot == 0:
            d = np.zeros((1, 0), dtype=np.float32)
            idx = np.zeros((1, 0), dtype=np.int64)
            return d, idx
        k = min(int(top_k), ntot)
        k = max(k, 1)
        return self._index.search(q, k)

    def ntotal(self) -> int:
        n = self._index.ntotal
        if callable(n):
            return int(n())
        return int(n)

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))

    @classmethod
    def read(cls, path: str | Path) -> "FAISSIndex":
        raw: Any = faiss.read_index(str(path))
        d = int(getattr(raw, "d", 512))
        obj = cls(d)
        obj._index = raw
        return obj


def build_index(product_embeddings: np.ndarray) -> FAISSIndex:
    if product_embeddings.size == 0:
        d = int(product_embeddings.shape[1]) if product_embeddings.ndim == 2 else 512
        return FAISSIndex(d)
    d = int(product_embeddings.shape[1])
    index = FAISSIndex(d)
    index.add(product_embeddings)
    return index


def search(
    index: FAISSIndex, query_embedding: np.ndarray, top_k: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    return index.search(query_embedding, top_k=top_k)
