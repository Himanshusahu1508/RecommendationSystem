"""
Load catalog manifest (and optional per-SKU embedding JSON) from S3.

Uses the same AWS credentials as the rest of the app (env or instance role).
Set in .env: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION,
plus S3_CATALOG_BUCKET and S3_CATALOG_MANIFEST_KEY.
"""

from __future__ import annotations

import json
import os
from typing import Any

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from app.config import Settings, get_settings


def _s3_client(region: str | None) -> Any:
    return boto3.client(
        "s3",
        config=Config(signature_version="s3v4"),
        region_name=region or os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION"),
    )


def read_s3_json(bucket: str, key: str, settings: Settings | None = None) -> Any:
    s = settings or get_settings()
    region = s.aws_region or os.environ.get("AWS_DEFAULT_REGION")
    c = _s3_client(region)
    r = c.get_object(Bucket=bucket, Key=key)
    return json.loads(r["Body"].read().decode("utf-8"))


def load_catalog_manifest_from_s3(settings: Settings | None = None) -> list[dict[str, Any]]:
    s = settings or get_settings()
    b = s.s3_catalog_bucket
    if not b:
        raise ValueError("Set s3_catalog_bucket in environment")
    k = catalog_s3_key(s, s.s3_catalog_manifest_key)
    data = read_s3_json(b, k, s)
    if not isinstance(data, list):
        raise ValueError("Catalog manifest must be a JSON array of product objects")
    return data


def _parse_embedding_file(raw: dict[str, Any] | list[Any]) -> list[float]:
    if isinstance(raw, list):
        return [float(x) for x in raw]
    emb = raw.get("embedding")
    if not isinstance(emb, list):
        raise ValueError("Sidecar must be [float|] or {embedding: [float, ...]}")
    return [float(x) for x in emb]


def catalog_s3_key(s: Settings, key_or_relative: str) -> str:
    k = key_or_relative.strip().lstrip("/")
    p = (s.s3_catalog_prefix or "").strip().strip("/")
    if p:
        return f"{p}/{k}"
    return k


def enrich_embeddings_from_s3(
    products: list[dict[str, Any]],
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """
    For rows missing 'embedding', fetch JSON from S3 using:
    - 'embedding_s3_key' (path inside bucket, optional under s3_catalog_prefix), or
    - s3_embedding_key_pattern (e.g. 'embeddings/{id}.json') when 'id' is set.
    """
    s = settings or get_settings()
    if not products or all(p.get("embedding") is not None for p in products):
        return [dict(p) for p in products]
    b = s.s3_catalog_bucket
    if not b:
        raise ValueError("s3_catalog_bucket is required to fetch sidecar embeddings from S3")
    region = s.aws_region or os.environ.get("AWS_DEFAULT_REGION")
    c = _s3_client(region)
    out: list[dict[str, Any]] = []
    for p in products:
        row = dict(p)
        if row.get("embedding") is not None:
            out.append(row)
            continue
        ekey: str | None = row.get("embedding_s3_key")
        if not ekey and s.s3_embedding_key_pattern and row.get("id") is not None:
            ekey = s.s3_embedding_key_pattern.replace("{id}", str(row["id"]))
        if not ekey:
            out.append(row)
            continue
        full_key = catalog_s3_key(s, ekey)
        try:
            body = c.get_object(Bucket=b, Key=full_key)["Body"].read()
            j = json.loads(body.decode("utf-8"))
            row["embedding"] = _parse_embedding_file(j)  # type: ignore[arg-type]
        except ClientError as e:
            code = (e.response or {}).get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey"):
                print(f"Warning: missing embedding object s3://{b}/{full_key} for id={row.get('id')!r}")
            else:
                raise
        out.append(row)
    return out


def validate_embedding_dims(products: list[dict[str, Any]], expected_dim: int) -> None:
    for p in products:
        e = p.get("embedding")
        if e is None:
            continue
        if not isinstance(e, list) or len(e) != expected_dim:
            raise ValueError(
                f"Product {p.get('id')!r}: embedding length must be {expected_dim}, got {len(e) if isinstance(e, list) else 'n/a'}",
            )
