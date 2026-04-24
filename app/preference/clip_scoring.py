"""
CLIP query vs per-product ``clip_embedding`` (512-d L2-normalized, same as ``eyewear_recommender`` / FAISS index build).

Uses :class:`eyewear_recommender.clip_backend.CLIPBackend` (Hugging Face `openai/clip-vit-base-patch32`) when
``transformers`` + ``torch`` are installed. Falls back to 0.5 if unavailable.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)

_clip_singleton: Any = None


def is_available() -> bool:
    try:
        import torch  # noqa: F401
        from transformers import CLIPModel  # noqa: F401
    except ImportError:
        return False
    return True


def _get_clip() -> Any | None:
    global _clip_singleton
    if _clip_singleton is not None:
        return _clip_singleton
    if not is_available():
        return None
    try:
        from eyewear_recommender.clip_backend import load_clip

        _clip_singleton = load_clip()
        return _clip_singleton
    except (ImportError, OSError) as e:
        log.debug("CLIP backend load failed: %s", e)
        return None


def encode_user_style(
    text: str | None,
    image_bytes: bytes | None,
) -> NDArray[np.float32] | None:
    """
    L2-normalized query vector: fused text+image when both present (same as reference `encode_text_and_image_fused`).
    """
    t = (text or "").strip()
    if not t and not image_bytes:
        return None
    clip = _get_clip()
    if clip is None:
        return None
    from PIL import Image

    if image_bytes:
        im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    else:
        im = None
    if im is None:
        return clip.encode_text(t)
    if not t:
        return clip.encode_image(im)
    return clip.encode_text_and_image_fused(t, im, text_weight=0.5)


def clip_preference_01(product: dict[str, Any], query: NDArray[np.float32] | None) -> float:
    if query is None:
        return 0.5
    emb = product.get("clip_embedding")
    if not isinstance(emb, list) or not emb:
        return 0.5
    v = np.asarray(emb, dtype=np.float32)
    q = query.astype(np.float32)
    if v.shape != q.shape:
        return 0.5
    c = float(np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v) + 1e-12))
    return (c + 1.0) / 2.0
