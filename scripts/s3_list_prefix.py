#!/usr/bin/env python3
"""
List object keys in the catalog bucket so you can set S3_CATALOG_PREFIX and S3_CATALOG_MANIFEST_KEY.

Usage:
  python scripts/s3_list_prefix.py
  python scripts/s3_list_prefix.py --prefix ""           # bucket root
  python scripts/s3_list_prefix.py --prefix some/folder/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from app.config import get_settings
from app.services.s3_image import get_catalog_s3_client


def main() -> int:
    s = get_settings()
    p = argparse.ArgumentParser(description="List S3 keys in catalog bucket")
    p.add_argument(
        "--prefix",
        default=None,
        help="S3 prefix to list (default: from S3_CATALOG_PREFIX in .env, or empty)",
    )
    p.add_argument(
        "--glass",
        choices=("sunglass", "eyeglass", "normal"),
        default=None,
        help="List under effective …/glass/<subfolder>/ (same as app catalog toggle); omit for raw prefix",
    )
    p.add_argument("--max", type=int, default=200, help="Max keys to print")
    args = p.parse_args()

    if not s.s3_catalog_bucket:
        print("Set S3_CATALOG_BUCKET in .env", file=sys.stderr)
        return 1

    from app.services.catalog_s3_prefix import effective_catalog_s3_prefix

    prefix = args.prefix
    if prefix is None:
        gcat = args.glass
        if gcat == "normal":
            gcat = "eyeglass"
        if gcat is not None:
            prefix = effective_catalog_s3_prefix(s, gcat)
        else:
            prefix = s.s3_catalog_prefix or ""

    client = get_catalog_s3_client(s)
    n = 0
    try:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=s.s3_catalog_bucket, Prefix=prefix, PaginationConfig={"PageSize": 100}):
            for obj in page.get("Contents", []) or []:
                print(obj["Key"])
                n += 1
                if n >= args.max:
                    if n == args.max:
                        print("... (use --max to show more)", file=sys.stderr)
                    return 0
    except OSError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if n == 0:
        print(
            f"No objects under s3://{s.s3_catalog_bucket}/{prefix!r}\n"
            "Try: python scripts/s3_list_prefix.py --prefix \"\"",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
