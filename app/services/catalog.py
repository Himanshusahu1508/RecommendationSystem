from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.db import ProductRow


def list_catalog_products(db: Session) -> list[dict[str, Any]]:
    rows = db.scalars(select(ProductRow).order_by(ProductRow.id)).all()
    return [dict(r.payload) for r in rows if r.payload]


def get_catalog_products(
    db: Session,
    s: Settings,
    *,
    glass_category: str | None = None,
) -> list[dict[str, Any]]:
    """
    Use DB (seeded) or S3 manifest + embeddings, depending on CATALOG_SOURCE and S3_CATALOG_BUCKET.

    ``glass_category`` (``sunglass`` / ``eyeglass`` / ``normal``): narrows S3 listing to
    ``…/glass/<subfolder>/`` when configured; ignored for DB-only catalog.
    """
    src = (s.catalog_source or "auto").strip().lower()
    if src == "db":
        return list_catalog_products(db)
    if src == "s3":
        if not s.s3_catalog_bucket:
            raise ValueError("catalog_source=s3 requires S3_CATALOG_BUCKET")
        from app.services.s3_catalog import load_catalog_from_s3

        return load_catalog_from_s3(s, glass_category=glass_category)
    if src == "auto":
        if s.s3_catalog_bucket:
            from app.services.s3_catalog import load_catalog_from_s3

            return load_catalog_from_s3(s, glass_category=glass_category)
        return list_catalog_products(db)
    raise ValueError(f"Unknown CATALOG_SOURCE: {s.catalog_source!r}")
