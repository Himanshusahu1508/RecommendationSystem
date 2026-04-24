from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Annotated, Any, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import get_settings, Settings
from app.db import get_session, init_db
from app.embedding_config import load_face_shape_prototypes
from app.services import catalog, regional, s3_image, user_context
from glasses_recommend import _image_size_from_bytes, recommend_from_bytes
from ranking_signals import RankingWeights

settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()] or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _apply_aws_env(s: Settings) -> None:
    if s.aws_region and not os.environ.get("AWS_DEFAULT_REGION"):
        os.environ["AWS_DEFAULT_REGION"] = s.aws_region


def _check_bearer(authorization: Optional[str]) -> None:
    s = get_settings()
    if s.disable_auth or not s.api_bearer_token:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token"
        )
    if authorization[7:].strip() != s.api_bearer_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/ready")
def ready(db: Session = Depends(get_session)) -> dict[str, Any]:
    try:
        db.execute(text("SELECT 1"))
        return {"ready": True, "database": "ok"}
    except OSError as e:
        return {"ready": False, "database": str(e)}


@app.post(f"{settings.api_prefix}/recommend", response_model=None)
def recommend(
    db: Session = Depends(get_session),
    authorization: Annotated[Optional[str], Header()] = None,
    image: Optional[UploadFile] = File(default=None, description="User selfie (JPEG/PNG)"),
    s3_key: Optional[str] = Form(default=None, description="Alternative: object key in S3"),
    s3_bucket: Optional[str] = Form(default=None, description="Bucket for s3_key (or env S3_USER_IMAGES_BUCKET)"),
    user_public_id: Optional[str] = Form(default=None, description="User id in app_users to load order + region"),
    user_context_json: Optional[str] = Form(
        default=None, description="Override context JSON (e.g. for tests) if public_id not set",
    ),
    top_n: int = Form(default=5),
    no_quality: bool = Form(default=False),
    style_prompt: Optional[str] = Form(
        default=None,
        description="Optional free text for CLIP style encoding (with style_reference_image)",
    ),
    glass_category: Optional[str] = Form(
        default=None,
        description="Catalog slice: sunglass | eyeglass | normal — under …/glass/<folder>/ (see S3_GLASS_* env)",
    ),
    style_reference_image: Optional[UploadFile] = File(
        default=None,
        description="Optional mood/style image for CLIP+hybrid (not the selfie; uses image field for face)",
    ),
) -> Any:
    _apply_aws_env(settings)
    _check_bearer(authorization)

    if not image and not s3_key:
        raise HTTPException(
            400, detail="Provide either multipart 'image' or form field 's3_key'",
        )

    if image and s3_key:
        raise HTTPException(400, detail="Send only one of: image file or s3_key")

    try:
        if s3_key:
            bucket = s3_bucket or settings.s3_user_images_bucket
            if not bucket:
                raise HTTPException(
                    400,
                    detail="Set s3_bucket or S3_USER_IMAGES_BUCKET in environment",
                )
            image_bytes = s3_image.download_s3_object(bucket, s3_key)
        else:
            assert image is not None
            image_bytes = image.file.read()
    except OSError as e:
        raise HTTPException(400, detail=f"Could not read image: {e}") from e
    except Exception as e:  # noqa: BLE001 — surface boto errors
        raise HTTPException(502, detail=f"Storage error: {e!s}") from e

    w, h = _image_size_from_bytes(image_bytes)
    ref_bytes: bytes | None = None
    if style_reference_image is not None:
        ref_bytes = style_reference_image.file.read()

    try:
        products = catalog.get_catalog_products(db, settings, glass_category=glass_category)
    except ValueError as e:
        raise HTTPException(400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(502, detail=str(e)) from e
    if not products:
        raise HTTPException(
            503,
            detail="Empty catalog. Set S3_CATALOG_BUCKET+manifest, or run scripts/seed_from_json.py",
        )

    try:
        reg = regional.get_regional_map(db, settings)
    except (RuntimeError, OSError) as e:
        raise HTTPException(502, detail=f"Regional data: {e!s}") from e

    ctx: dict[str, Any] | None = None
    if user_public_id:
        ctx = user_context.build_user_context_dict(db, user_public_id)
        if ctx is None:
            raise HTTPException(404, detail=f"Unknown user_public_id: {user_public_id}")
    elif user_context_json:
        try:
            ctx = json.loads(user_context_json)
        except json.JSONDecodeError as e:
            raise HTTPException(400, detail=f"Invalid user_context_json: {e}") from e

    try:
        prototypes = load_face_shape_prototypes(settings.embedding_dim)
    except ValueError as e:
        raise HTTPException(500, detail=str(e)) from e

    out = recommend_from_bytes(
        image_bytes,
        w,
        h,
        products,
        regional=reg,
        check_quality=not no_quality,
        top_n=top_n,
        user_context=ctx,
        face_shape_prototypes=prototypes,
        ranking_weights=RankingWeights(),
        style_prompt=style_prompt,
        style_reference_image_bytes=ref_bytes,
        use_preference_hybrid=settings.use_preference_hybrid,
        hybrid_w_preference=settings.hybrid_w_preference,
        hybrid_w_face=settings.hybrid_w_face,
        use_clip_preference=settings.clip_preference_enabled,
    )
    if glass_category is not None:
        out = {**out, "catalog_glass_category": glass_category}
    return out


def create_app() -> FastAPI:
    return app
