"""
Face-shape query vectors (same dimension as catalog embeddings).
Override via EMBEDDING_JSON env or use defaults; dimension must match Settings.embedding_dim.
"""

from __future__ import annotations

import json
import os
from typing import Any


def default_prototypes() -> dict[str, list[float]]:
    from glasses_recommend import FACE_SHAPE_PROTOTYPES

    return {k: list(v) for k, v in FACE_SHAPE_PROTOTYPES.items()}


def load_face_shape_prototypes(embedding_dim: int) -> dict[str, list[float]]:
    """
    Load prototypes from FACE_SHAPE_PROTOTYPES, or EMBEDDING_JSON path in env,
    or EMBEDDING_PROTO_JSON string. Validates vector length.
    """
    path = os.environ.get("EMBEDDING_JSON")
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            raw: Any = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError("EMBEDDING_JSON must be a JSON object of name -> [float, ...]")
        return _validate(raw, embedding_dim)
    raw_s = os.environ.get("EMBEDDING_PROTO_JSON")
    if raw_s:
        raw = json.loads(raw_s)
        if not isinstance(raw, dict):
            raise ValueError("EMBEDDING_PROTO_JSON must be a JSON object")
        return _validate(raw, embedding_dim)
    p = default_prototypes()
    return _validate(p, embedding_dim)


def _validate(p: dict[str, Any], dim: int) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for k, v in p.items():
        if not isinstance(v, list) or not all(isinstance(x, (int, float)) for x in v):
            raise ValueError(f"Invalid embedding vector for {k!r}")
        if len(v) != dim:
            raise ValueError(
                f"Embedding for {k!r} has length {len(v)} but embedding_dim is {dim}",
            )
        out[k] = [float(x) for x in v]
    return out
