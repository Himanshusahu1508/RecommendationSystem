"""
Build a product catalog from a *flat* S3 prefix: JPEG objects named ``PRODUCTID_VIEWINDEX.jpg``
(e.g. ``lusmt00438_0.jpg``, ``cp0031_0.jpeg``). Multiple views per product share the same id with
different view indices.

Synthetic tags and embeddings are derived from each product id until you supply a manifest
and real embeddings from S3.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any

from app.config import Settings
from app.services.catalog_s3_prefix import effective_catalog_s3_prefix
from app.services.s3_image import get_catalog_s3_client

# Filename at end of key: lusmt00438_2.jpg, FRAME-A_0.jpg, sku123_1.jpeg — {productId}_{viewIndex}.jpg
_FRAME_IMAGE = re.compile(
    r"^(?P<pid>[\w-]+)_(?P<view>\d+)\.(?P<ext>jpe?g)$",
    re.IGNORECASE,
)

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
    v = [float((h[i] + h[(i * 3) % 32]) % 256) / 128.0 - 1.0 for i in range(dim)]
    s = sum(x * x for x in v) ** 0.5
    if s < 1e-9:
        return v
    return [x / s for x in v]


def _list_all_object_keys(s: Settings, *, list_prefix: str | None = None) -> list[str]:
    b = s.s3_catalog_bucket
    if not b:
        return []
    client = get_catalog_s3_client(s)
    if list_prefix is not None:
        prefix = (list_prefix or "").lstrip("/")
    else:
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


def build_lusmt_flat_catalog(
    s: Settings,
    *,
    glass_category: str | None = None,
) -> list[dict[str, Any]]:
    """
    Return catalog rows: id, name, s3_key (primary view), s3_bucket, s3_image_keys, frame_tags, embedding, face_shapes, popularity.

    ``glass_category``: ``sunglass`` / ``eyeglass`` narrows under ``S3_GLASS_PARENT`` and optional
    per-type extra segments (see :func:`app.services.catalog_s3_prefix.effective_catalog_s3_prefix`). Omit to list only under ``S3_CATALOG_PREFIX``.
    """
    eff = effective_catalog_s3_prefix(s, glass_category)
    keys = _list_all_object_keys(s, list_prefix=eff)
    by_pid: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for k in keys:
        base = k.rsplit("/", 1)[-1]
        m = _FRAME_IMAGE.match(base)
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


def diagnose_flat_catalog(
    s: Settings,
    *,
    glass_category: str | None = None,
    sample: int = 5,
) -> dict[str, Any]:
    """
    Why the flat catalog might be empty: effective prefix, JPEG count, regex match count, sample keys.
    """
    eff = effective_catalog_s3_prefix(s, glass_category)
    keys = _list_all_object_keys(s, list_prefix=eff)
    unmatched: list[str] = []
    matched_bases: list[str] = []
    for k in keys:
        base = k.rsplit("/", 1)[-1]
        if _FRAME_IMAGE.match(base):
            matched_bases.append(base)
        else:
            if len(unmatched) < 16:
                unmatched.append(base)
    root = (s.s3_catalog_prefix or "").strip()
    path_hint: str | None = None
    if root and (not keys or not matched_bases):
        path_hint = (
            f"If objects are at s3://{s.s3_catalog_bucket or 'BUCKET'}/Glass/... (not under `{root}/Glass/...`), "
            "set S3_CATALOG_PREFIX empty. The `all_images` folder in your bucket is only under `Glass/Sunglass/`; "
            "sunglass listing uses S3_GLASS_SUNGLASS_EXTRA_PREFIX=all_images, not the global catalog prefix."
        )
    return {
        "s3_bucket": s.s3_catalog_bucket,
        "effective_prefix": eff,
        "s3_catalog_prefix_root": root or None,
        "s3_glass_sunglass_extra_prefix": getattr(s, "s3_glass_sunglass_extra_prefix", None),
        "s3_glass_eyeglass_extra_prefix": getattr(s, "s3_glass_eyeglass_extra_prefix", None),
        "use_glass_subfolders": getattr(s, "s3_use_glass_subfolders", True),
        "jpg_object_count": len(keys),
        "filename_pattern_matched": len(matched_bases),
        "expected_filename_shape": "PRODUCTID_VIEWINDEX.jpg (e.g. lusmt00438_0.jpg, cp0031_0.jpeg)",
        "sample_object_keys": keys[:sample],
        "sample_unmatched_filenames": unmatched[:8],
        "path_hint": path_hint,
    }
