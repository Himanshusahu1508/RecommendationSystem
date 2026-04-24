#!/usr/bin/env python3
"""
Load glasses_catalog.json + regional_affinity.json into the SQL database.
Run from project root: python scripts/seed_from_json.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import delete

from app.db import AppUser, OrderLineRow, OrderRow, ProductRow, RegionalCohortRow, get_engine, init_db
from sqlalchemy.orm import Session


def main() -> int:
    init_db()
    engine = get_engine()
    cat_path = ROOT / "glasses_catalog.json"
    reg_path = ROOT / "regional_affinity.json"
    with open(cat_path, encoding="utf-8") as f:
        catalog = json.load(f)
    with open(reg_path, encoding="utf-8") as f:
        regional = json.load(f)
    if not isinstance(catalog, list):
        return 1

    with Session(engine) as session:
        session.execute(delete(OrderLineRow))
        session.execute(delete(OrderRow))
        session.execute(delete(AppUser))
        session.execute(delete(ProductRow))
        session.execute(delete(RegionalCohortRow))
        session.commit()

    with Session(engine) as session:
        for p in catalog:
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
                    s3_bucket=p.get("s3_bucket"),
                )
            )
        u = AppUser(
            public_id="demo-1",
            region="IN-MH",
        )
        session.add(u)
        session.flush()
        o = OrderRow(user_id=u.id, created_at="")
        session.add(o)
        session.flush()
        session.add(OrderLineRow(order_id=o.id, product_id="g-004", quantity=1))
        for region_code, data in regional.items():
            if not isinstance(data, dict):
                continue
            session.add(
                RegionalCohortRow(
                    region_code=region_code,
                    product_affinity=data.get("product_affinity") or {},
                    tag_affinity=data.get("tag_affinity") or {},
                )
            )
        session.commit()
    print("Seeded products, regional_cohort, and demo user public_id=demo-1 (last order g-004).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
