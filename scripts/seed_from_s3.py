#!/usr/bin/env python3
"""
Load catalog (manifest + embeddings) from S3 into the local/Postgres `products` table.

Requires: S3_CATALOG_BUCKET, AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (or IAM role),
          and optional S3_CATALOG_PREFIX, S3_CATALOG_MANIFEST_KEY, S3_EMBEDDING_KEY_PATTERN

Run from project root:  python scripts/seed_from_s3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import ProductRow, get_engine, init_db
from app.services.s3_catalog import clear_catalog_cache, load_catalog_from_s3


def main() -> int:
    s = get_settings()
    if not s.s3_catalog_bucket:
        print("Set S3_CATALOG_BUCKET in .env", file=sys.stderr)
        return 1
    init_db()
    engine = get_engine()
    try:
        products = load_catalog_from_s3(s, use_cache=False)
    except (RuntimeError, ValueError) as e:
        print(f"Load failed: {e}", file=sys.stderr)
        m = s.s3_catalog_prefix
        if m and not m.endswith("/"):
            m = f"{m}/"
        man = f"{m}{s.s3_catalog_manifest_key}".lstrip("/")
        print(
            f"\nExpected manifest at: s3://{s.s3_catalog_bucket}/{man}\n"
            "List what is actually in the bucket:  python scripts/s3_list_prefix.py --prefix \"\"\n"
            "Then set in .env: S3_CATALOG_PREFIX=  (path to your folder) and S3_CATALOG_MANIFEST_KEY=your-file.json",
            file=sys.stderr,
        )
        return 1
    with Session(engine) as session:
        session.execute(delete(ProductRow))
        for p in products:
            pid = str(p.get("id", ""))
            if not pid:
                continue
            name = str(p.get("name", ""))
            session.add(
                ProductRow(
                    id=pid,
                    name=name,
                    payload=p,
                    s3_key=p.get("s3_key"),
                    s3_bucket=p.get("s3_bucket") or s.s3_catalog_bucket,
                )
            )
        session.commit()
    clear_catalog_cache()
    print(f"Seeded {len(products)} products from s3://{s.s3_catalog_bucket}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
