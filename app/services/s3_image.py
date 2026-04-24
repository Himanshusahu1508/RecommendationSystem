from __future__ import annotations

import os
from typing import Any, Optional

import boto3
from botocore.client import Config

from app.config import Settings


def _region() -> Optional[str]:
    return os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")


def get_default_s3_client() -> Any:
    """S3 with default credential chain (your .env AWS_* — used for user selfie objects, etc.)."""
    return boto3.client(
        "s3",
        config=Config(signature_version="s3v4"),
        region_name=_region(),
    )


def get_catalog_s3_client(s: Settings) -> Any:
    """
    S3 client for catalog bucket: use S3_CATALOG_* keys if set (e.g. friend’s bucket),
    otherwise same as default chain (your keys for both Rekognition and S3).
    """
    region = s.s3_catalog_region or s.aws_region or _region()
    if s.s3_catalog_access_key_id and s.s3_catalog_secret_access_key:
        return boto3.client(
            "s3",
            aws_access_key_id=s.s3_catalog_access_key_id,
            aws_secret_access_key=s.s3_catalog_secret_access_key,
            region_name=region,
            config=Config(signature_version="s3v4"),
        )
    return boto3.client(
        "s3",
        config=Config(signature_version="s3v4"),
        region_name=region,
    )


def download_s3_object(bucket: str, key: str, *, client: Any | None = None) -> bytes:
    """GetObject. Pass `client` from get_catalog_s3_client for friend’s bucket; else default chain."""
    c = client or get_default_s3_client()
    r = c.get_object(Bucket=bucket, Key=key)
    return r["Body"].read()


def presign_get_url(
    s: Settings,
    bucket: str,
    key: str,
    *,
    expires: int = 3600,
) -> str:
    """HTTPS URL to open an object in a browser (uses catalog creds for that bucket)."""
    client = get_catalog_s3_client(s)
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires,
    )


def enrich_recommendations_with_presign(
    s: Settings,
    out: dict[str, Any],
    *,
    url_expires: int = 7200,
) -> dict[str, Any]:
    """
    Mutates each recommendation dict with `eyewear_image_urls` (presigned GET) from
    s3_key / s3_image_keys. No-op if not ok or no recommendations.
    """
    if not out.get("ok") or "recommendations" not in out:
        return out
    for r in out["recommendations"]:
        keys: list[str] = list(r.get("s3_image_keys") or [])
        if not keys and r.get("s3_key"):
            keys = [r["s3_key"]]
        b = r.get("s3_bucket") or s.s3_catalog_bucket
        if b and keys:
            r["eyewear_image_urls"] = [presign_get_url(s, b, k, expires=url_expires) for k in keys]
    return out
