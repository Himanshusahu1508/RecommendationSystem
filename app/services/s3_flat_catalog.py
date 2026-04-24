"""
Build a product catalog from a *flat* S3 bucket of eyewear images named:
  lusmt{PRODUCT_ID}_{VIEW}.jpg
Example: lusmt00438_0.jpg … lusmt00438_3.jpg  → one product lusmt00438 with 4 views.

When no manifest.json exists, we assign synthetic per-product tags and normalized
embedding vectors (derived from the product id) so the existing hybrid ranker
(tag match vs face-shape rules + cosine) can differentiate items. For true
visual similarity to the user’s face, replace with a shared embedding model.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any

from app.config import Settings
from app.services.s3_image import get_catalog_s3_client

# Filename at end of key: lusmt00438_2.jpg
_LUSMT = re.compile(r"^(?P<pid>lusmt\d+)_(?P<view>\d+)\.(?P<ext>jpe?g)$", re.IGNORECASE)

_STYLES = [
    "wayfarer",
    "round",
    "aviator",
    "rectangular",
    "cat_eye",
    "oval",
    "metal",
    "acetate",
    "geometric",
    "keyhole",
    "soft",
    "classic",
    "bold",
]


def _synthetic_tags_for_product_id(pid: str) -> list[str]:
    h = int(hashlib.sha256(pid.encode()).hexdigest()[:12], 16)
    a, b, c = h % 12, (h // 7) % 12, (h // 19) % 12
    tags = [_STYLES[a], _STYLES[b], _STYLES[c]]
    return list(dict.fromkeys(tags))  # dedupe, keep order


def _synthetic_color_family_tag(pid: str) -> str:
    """
    Stable one-of-three palette tag for age-based color scoring (vibrant / neutral / plain).
    Replace with real product color when you have catalog metadata.
    """
    h = int(hashlib.sha256(f"color-family:{pid}".encode()).hexdigest()[:8], 16)
    return ("color_vibrant", "color_neutral", "color_plain")[h % 3]


def _synthetic_unit_embedding(pid: str, dim: int) -> list[float]:
    h = hashlib.sha256(f"eyewear-emb:{pid}".encode()).digest()
    n = min(dim, 32)
    v = [float((h[i] + h[(i * 3) % 32]) % 256) / 128.0 - 1.0 for i in range(dim)]
    s = sum(x * x for x in v) ** 0.5
    if s < 1e-9:
        return v
    return [x / s for x in v]


def _list_all_object_keys(s: Settings) -> list[str]:
    b = s.s3_catalog_bucket
    if not b:
        return []
    client = get_catalog_s3_client(s)
    prefix = (s.s3_catalog_prefix or "").lstrip("/")
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=b, Prefix=prefix, PaginationConfig={"PageSize": 1000}):
        for obj in page.get("Contents", []) or []:
            k = obj["Key"]
            if not k.lower().endswith((".jpg", ".jpeg", ".JPG", ".JPEG")):
                continue
            keys.append(k)
    return keys


def build_lusmt_flat_catalog(s: Settings) -> list[dict[str, Any]]:
    """
    Return catalog rows: id, name, s3_key (primary view), s3_bucket, s3_image_keys, frame_tags, embedding, face_shapes, popularity.
    """
    keys = _list_all_object_keys(s)
    by_pid: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for k in keys:
        base = k.rsplit("/", 1)[-1]
        m = _LUSMT.match(base)
        if not m:
            continue
        pid = m.group("pid")
        view = int(m.group("view"))
        by_pid[pid].append((view, k))
    if not by_pid:
        return []

    dim = s.embedding_dim
    out: list[dict[str, Any]] = []
    b = s.s3_catalog_bucket
    assert b
    for pid in sorted(by_pid):
        items = sorted(by_pid[pid], key=lambda t: t[0])
        primary = items[0][1]
        all_keys = [x[1] for x in items]
        style_tags = _synthetic_tags_for_product_id(pid)
        color_tag = _synthetic_color_family_tag(pid)
        out.append(
            {
                "id": pid,
                "name": f"Frame {pid}",
                "s3_key": primary,
                "s3_bucket": b,
                "s3_image_keys": all_keys,
                "face_shapes": ["all"],
                "frame_tags": [*style_tags, color_tag],
                "color_family": color_tag.removeprefix("color_"),
                "popularity": 0.5,
                "embedding": _synthetic_unit_embedding(pid, dim),
                "catalog_mode": "s3_flat_lusmt",
            },
        )
    return out
