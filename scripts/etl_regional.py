#!/usr/bin/env python3
"""
Recompute regional cohort affinities from order history (same shape as regional_affinity.json).
Run: python scripts/etl_regional.py

For each (user.region, product_id) count lines; normalize counts to 0.3..1.0 for product_affinity.
Tag affinities: aggregate from product payload frame_tags weighted by count (simplified).
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db import AppUser, OrderLineRow, OrderRow, ProductRow, RegionalCohortRow, get_engine, init_db


def main() -> int:
    init_db()
    engine = get_engine()
    with Session(engine) as session:
        # (region, product_id) -> count
        counts: dict[tuple[str, str], int] = defaultdict(int)
        lines = session.execute(
            select(AppUser.region, OrderLineRow.product_id)
            .join(OrderRow, OrderLineRow.order_id == OrderRow.id)
            .join(AppUser, OrderRow.user_id == AppUser.id)
        ).all()
        for region, pid in lines:
            if not region:
                continue
            counts[(region, str(pid))] += 1

        if not counts:
            print("No orders; nothing to aggregate.")
            return 0

        # product_affinity per region
        by_reg: dict[str, dict[str, float]] = defaultdict(dict)
        max_c: dict[str, int] = defaultdict(int)
        for (region, pid), c in counts.items():
            by_reg[region][pid] = float(c)
            max_c[region] = max(max_c[region], c)
        product_affinity: dict[str, dict[str, float]] = {}
        for region, pmap in by_reg.items():
            mx = max(max_c[region], 1)
            product_affinity[region] = {
                pid: round(0.3 + 0.7 * (v / mx), 4) for pid, v in pmap.items()
            }

        # Tag counts: sum frame_tags from products purchased in region
        tag_counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        prows = {str(r.id): r for r in session.scalars(select(ProductRow))}
        for (region, pid), c in counts.items():
            pr = prows.get(pid)
            if not pr or not pr.payload:
                continue
            for t in pr.payload.get("frame_tags") or []:
                tag_counts[region][str(t)] += float(c)
        tag_affinity: dict[str, dict[str, float]] = {}
        for region, tmap in tag_counts.items():
            mx = max(tmap.values()) if tmap else 1.0
            tag_affinity[region] = {
                k: round(0.85 + 0.25 * (v / mx), 4) for k, v in tmap.items()
            }

        session.execute(delete(RegionalCohortRow))
        regions = set(product_affinity) | set(tag_affinity) | set(tag_counts.keys())
        if "default" not in regions:
            regions.add("default")
        for region in regions:
            session.add(
                RegionalCohortRow(
                    region_code=region,
                    product_affinity=product_affinity.get(region, {}),
                    tag_affinity=tag_affinity.get(region, {}),
                )
            )
        session.commit()
    print("Updated regional_cohort from orders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
