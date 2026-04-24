from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.db import RegionalCohortRow


def get_regional_map(db: Session, s: Settings) -> dict[str, Any] | None:
    """Load cohort stats from S3 JSON if configured, else from DB."""
    if s.s3_catalog_bucket and s.s3_regional_json_key:
        from app.services.s3_catalog import load_regional_from_s3

        return load_regional_from_s3(s)
    return regional_map_for_recommendation(db)


def regional_map_for_recommendation(db: Session) -> dict[str, Any] | None:
    """
    Return structure expected by ranking_signals.score_region_affinity
    (keys per region: product_affinity, tag_affinity), plus top-level "default" fallback.
    """
    rows = db.scalars(select(RegionalCohortRow)).all()
    m: dict[str, Any] = {}
    for r in rows:
        m[r.region_code] = {
            "product_affinity": r.product_affinity or {},
            "tag_affinity": r.tag_affinity or {},
        }
    if not m:
        return None
    return m
