"""
Load product catalog (with embeddings) from S3: one manifest JSON + optional per-SKU embedding files.

Expected manifest: JSON array of product objects. Each row can include:
- embedding: list[float] (inline)
- embedding_s3_key: key inside bucket (if no leading slash, joined with prefix)
- or omit both and use S3_EMBEDDING_KEY_PATTERN with {id}

Image keys for display/CMS are optional; ranking only needs metadata + embedding.
"""

from __future__ import annotations

import json
import time
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError

from app.config import Settings
from app.services.s3_image import download_s3_object, get_catalog_s3_client

# Simple TTL cache: (bucket, manifest_key) -> (timestamp, data)
_cache: dict[tuple[str, str], tuple[float, list[dict[str, Any]]]] = {}
_CATALOG_TTL_S = 120.0


def _join_s3_key(prefix: str, key: str) -> str:
    p = (prefix or "").strip("/")
    k = key.strip().lstrip("/")
    if not p:
        return k
    return f"{p}/{k}"


def _parse_embedding_json(raw: bytes) -> list[float]:
    data: Any = json.loads(raw.decode("utf-8"))
    if isinstance(data, list):
        return [float(x) for x in data]
    if isinstance(data, dict) and "embedding" in data:
        return [float(x) for x in data["embedding"]]
    if isinstance(data, dict) and "vector" in data:
        return [float(x) for x in data["vector"]]
    raise ValueError("Embedding JSON must be a list or {embedding: [...] / vector: ...}")


def _fetch_object(bucket: str, key: str, s: Settings) -> bytes:
    """Uses friend’s S3 keys when S3_CATALOG_ACCESS_KEY_ID is set; else your default AWS keys."""
    return download_s3_object(bucket, key, client=get_catalog_s3_client(s))


def load_catalog_from_s3(
    s: Settings,
    *,
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    if not s.s3_catalog_bucket:
        raise ValueError("s3_catalog_bucket is not set")

    manifest_key = _join_s3_key(s.s3_catalog_prefix, s.s3_catalog_manifest_key)
    ck = (s.s3_catalog_bucket, manifest_key)
    now = time.time()
    if use_cache and ck in _cache:
        ts, data = _cache[ck]
        if now - ts < _CATALOG_TTL_S:
            return [dict(p) for p in data]  # copy

    try:
        raw = _fetch_object(s.s3_catalog_bucket, manifest_key, s)
    except (ClientError, BotoCoreError, OSError) as e:
        raise RuntimeError(f"Failed to read manifest s3://{s.s3_catalog_bucket}/{manifest_key}: {e}") from e

    try:
        items = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid manifest JSON: {e}") from e
    if not isinstance(items, list):
        raise ValueError("Manifest must be a JSON array of product objects")

    out: list[dict[str, Any]] = []
    for row in items:
        if not isinstance(row, dict):
            continue
        p = dict(row)
        pid = str(p.get("id", ""))
        emb: list[float] | None = None
        if "embedding" in p and p["embedding"] is not None:
            emb = [float(x) for x in p["embedding"]]
        else:
            emb_key = p.get("embedding_s3_key")
            if not emb_key and pid and s.s3_embedding_key_pattern:
                emb_key = s.s3_embedding_key_pattern.format(id=pid)
            if emb_key:
                ek = emb_key if emb_key.startswith("/") else _join_s3_key(s.s3_catalog_prefix, str(emb_key).lstrip("/"))
                ekb = s.s3_catalog_bucket
                try:
                    emb_raw = _fetch_object(ekb, ek, s)
                    emb = _parse_embedding_json(emb_raw)
                except (ClientError, BotoCoreError, OSError, ValueError) as e:
                    raise RuntimeError(
                        f"Failed to load embedding for id={pid!r} s3://{ekb}/{ek}: {e}",
                    ) from e
        if emb is not None and len(emb) != s.embedding_dim:
            raise ValueError(
                f"Product {pid!r} embedding length {len(emb)} != EMBEDDING_DIM {s.embedding_dim}",
            )
        if emb is not None:
            p["embedding"] = emb
        out.append(p)

    if use_cache:
        _cache[ck] = (now, [dict(x) for x in out])
    return out


def clear_catalog_cache() -> None:
    _cache.clear()


def load_regional_from_s3(s: Settings) -> dict[str, Any] | None:
    if not s.s3_catalog_bucket or not s.s3_regional_json_key:
        return None
    key = _join_s3_key(s.s3_catalog_prefix, s.s3_regional_json_key)
    try:
        raw = _fetch_object(s.s3_catalog_bucket, key, s)
        data: Any = json.loads(raw.decode("utf-8"))
    except (ClientError, BotoCoreError, OSError, json.JSONDecodeError) as e:
        raise RuntimeError(
            f"Failed regional JSON s3://{s.s3_catalog_bucket}/{key}: {e}",
        ) from e
    if not isinstance(data, dict):
        raise ValueError("Regional JSON in S3 must be a JSON object of regions")
    return data
