"""Map DB user + last order to ranking_signals UserContext fields."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import AppUser, OrderLineRow, OrderRow


def get_last_order_product_ids_for_user(db: Session, user_id: int) -> list[str]:
    oid = db.scalars(
        select(OrderRow.id)
        .where(OrderRow.user_id == user_id)
        .order_by(OrderRow.id.desc())
        .limit(1)
    ).first()
    if oid is None:
        return []
    lines = db.scalars(
        select(OrderLineRow.product_id).where(OrderLineRow.order_id == oid)
    ).all()
    return [str(p) for p in lines]


def build_user_context_dict(db: Session, public_id: str) -> dict[str, Any] | None:
    u = db.scalars(select(AppUser).where(AppUser.public_id == public_id).limit(1)).first()
    if u is None:
        return None
    product_ids = get_last_order_product_ids_for_user(db, u.id)
    ctx: dict[str, Any] = {
        "user_id": public_id,
        "region": u.region,
        "last_order_product_ids": product_ids,
    }
    if u.gender_override is not None:
        ctx["gender_override"] = u.gender_override
    if u.age_override is not None:
        ctx["age_override"] = u.age_override
    return ctx
