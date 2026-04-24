#!/usr/bin/env python3
r"""
Local image → AWS Rekognition (face) → rank eyewear from your S3 **flat** catalog
(`lusmt{ID}_{view}.jpg` under `S3_CATALOG_PREFIX`), add presigned image URLs.

Same ranking as `webcam_s3_eyewear.py`, but reads a file path for batch / testing.

.env: AWS_* (Rekognition), S3_CATALOG_BUCKET, optional S3_CATALOG_PREFIX, S3_CATALOG_ACCESS_KEY_*,
EMBEDDING_DIM=8. Catalog uses synthetic tags/embeddings per product id until you add a manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from app.config import get_settings
from app.embedding_config import load_face_shape_prototypes
from app.services.s3_flat_catalog import build_lusmt_flat_catalog
from app.services.s3_image import enrich_recommendations_with_presign
from glasses_recommend import _image_size_from_bytes, load_regional, recommend_from_bytes
from ranking_signals import RankingWeights


def _print_summary(out: dict[str, Any]) -> None:
    if not out.get("ok"):
        return
    sh = out.get("face_shape", "?")
    rules = out.get("rules") or {}
    tags = rules.get("preferred_frame_tags", [])
    print(
        f"\n--- Estimated face shape: {sh.upper()} ---\n"
        f"Rule-based frame styles to prefer: {', '.join(tags)}\n"
        f"(JSON below includes presigned S3 links for recommended frames.)\n",
        file=sys.stderr,
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description="Image file → face analysis → top eyewear from S3 catalog (flat lusmt* keys)",
    )
    p.add_argument("image", help="Path to JPEG/PNG (single face, good light)")
    p.add_argument("--top", type=int, default=5, dest="top_n")
    p.add_argument("--no-quality", action="store_true")
    p.add_argument("--json", action="store_true", help="Print JSON only to stdout")
    p.add_argument(
        "--regional-json",
        type=Path,
        default=ROOT / "regional_affinity.json",
        help="Optional regional weights; use a non-existent path to skip",
    )
    p.add_argument("--user-json", type=Path, default=None, help="User context (region, last orders, …)")
    args = p.parse_args()

    s = get_settings()

    if not os.environ.get("AWS_ACCESS_KEY_ID") and not os.environ.get("AWS_SESSION_TOKEN"):
        print(
            json.dumps(
                {
                    "error": "Set AWS creds in .env (or use a profile) for Rekognition",
                },
            ),
            file=sys.stderr,
        )
        return 1

    if not s.s3_catalog_bucket:
        print(
            json.dumps({"error": "Set S3_CATALOG_BUCKET in .env"}),
            file=sys.stderr,
        )
        return 1

    if not os.path.isfile(args.image):
        print(json.dumps({"error": f"file_not_found: {args.image}"}), file=sys.stderr)
        return 1

    products = build_lusmt_flat_catalog(s)
    if not products:
        print(
            json.dumps(
                {
                    "error": "No lusmt*_*_*.jpg under prefix. Check S3_CATALOG_PREFIX; "
                    "list keys: python scripts/s3_list_prefix.py",
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1

    reg = None
    if args.regional_json and args.regional_json.is_file():
        reg = load_regional(args.regional_json)
    if args.user_json and args.user_json.is_file():
        with open(args.user_json, encoding="utf-8") as f:
            user_context = json.load(f)
    else:
        user_context = None

    with open(args.image, "rb") as f:
        image_bytes = f.read()
    w, h = _image_size_from_bytes(image_bytes)

    try:
        prototypes = load_face_shape_prototypes(s.embedding_dim)
    except ValueError as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 1

    try:
        out = recommend_from_bytes(
            image_bytes,
            w,
            h,
            products,
            regional=reg,
            check_quality=not args.no_quality,
            top_n=args.top_n,
            user_context=user_context,
            face_shape_prototypes=prototypes,
            ranking_weights=RankingWeights(),
        )
        out = enrich_recommendations_with_presign(s, out)
    except (ClientError, BotoCoreError) as e:
        print(json.dumps({"ok": False, "error": str(e), "stage": "aws"}), file=sys.stderr)
        return 1

    if not out.get("ok"):
        print(json.dumps(out, indent=2, default=str))
        return 1

    if args.json:
        print(json.dumps(out, indent=2, default=str))
        return 0

    _print_summary(out)
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
