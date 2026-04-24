"""Application settings. Prefer IAM instance/task role for AWS; never commit secrets."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API
    app_name: str = "RS2 Recommend API"
    api_prefix: str = "/v1"
    disable_auth: bool = True
    # Optional: HS256 shared secret; if set, require Authorization: Bearer for /v1/recommend
    api_bearer_token: Optional[str] = None

    # SQL (use Postgres in production; sqlite for local)
    database_url: str = "sqlite:///./rs2.db"

    # AWS (boto3 default chain: env keys, then instance profile on ECS/EC2/Lambda)
    aws_region: Optional[str] = None  # or AWS_DEFAULT_REGION

    # S3: user selfie (API) or catalog (seed)
    s3_user_images_bucket: Optional[str] = None
    s3_user_images_key_prefix: str = "uploads/"

    # Optional: second account for catalog bucket only (friend’s keys). If unset, catalog S3 uses same as AWS_ACCESS_KEY_ID.
    s3_catalog_access_key_id: Optional[str] = None
    s3_catalog_secret_access_key: Optional[str] = None
    s3_catalog_region: Optional[str] = None  # falls back to aws_region / AWS_DEFAULT_REGION

    # Catalog on S3: manifest (JSON array) + per-row embeddings inline or in separate JSON in bucket.
    s3_catalog_bucket: Optional[str] = None
    s3_catalog_prefix: str = ""
    s3_catalog_manifest_key: str = "manifest.json"
    # When manifest rows omit "embedding", fetch using this path pattern (under prefix) or per-row "embedding_s3_key"
    s3_embedding_key_pattern: str = "embeddings/{id}.json"
    # Optional: regional_affinity shape JSON in the same bucket
    s3_regional_json_key: Optional[str] = None
    # auto: use S3 if s3_catalog_bucket is set, else DB | db | s3
    catalog_source: str = "auto"

    # Embedding / ranking (align with your model in production)
    embedding_model_id: str = "prototype-8d"
    embedding_dim: int = 8

    # CORS: comma-separated for web and mobile app origins
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"


@lru_cache
def get_settings() -> Settings:
    return Settings()
