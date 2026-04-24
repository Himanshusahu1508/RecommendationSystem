from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import JSON, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


class ProductRow(Base):
    __tablename__ = "products"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), default="")
    # Full catalog object for the recommender (face_shapes, frame_tags, embedding, …)
    payload: Mapped[dict] = mapped_column(JSON)
    s3_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    s3_bucket: Mapped[str | None] = mapped_column(String(256), nullable=True)


class AppUser(Base):
    __tablename__ = "app_users"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    public_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    region: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # Optional: profile overrides for UserContext
    gender_override: Mapped[str | None] = mapped_column(String(32), nullable=True)
    age_override: Mapped[int | None] = mapped_column(nullable=True)


class OrderRow(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(index=True)
    created_at: Mapped[str] = mapped_column(String(32), default="", index=True)


class OrderLineRow(Base):
    __tablename__ = "order_lines"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    order_id: Mapped[int] = mapped_column(index=True)
    product_id: Mapped[str] = mapped_column(String(64), index=True)
    quantity: Mapped[int] = mapped_column(default=1)


class RegionalCohortRow(Base):
    __tablename__ = "regional_cohort"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    region_code: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    # Same shape as regional_affinity.json per region: product_affinity, tag_affinity
    product_affinity: Mapped[dict] = mapped_column(JSON, default=dict)
    tag_affinity: Mapped[dict] = mapped_column(JSON, default=dict)
    updated_at: Mapped[str] = mapped_column(String(32), default="")


def get_engine():
    s = get_settings()
    return create_engine(
        s.database_url,
        connect_args={"check_same_thread": False} if s.database_url.startswith("sqlite") else {},
    )


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())


def get_session() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
