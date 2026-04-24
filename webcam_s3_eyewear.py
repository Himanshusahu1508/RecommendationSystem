#!/usr/bin/env python3
r"""
1) Open the front camera (index 0 on most Macs).
2) Press SPACE: capture a frame → AWS Rekognition (face must be single, good quality by default).
3) Infers face shape (round / long / oval / heart) and applies rule-based style tags.
4) Loads the eyewear image catalog from S3 (files like lusmt00438_0.jpg … lusmt00438_3.jpg).
5) Ranks products and adds presigned URLs to view recommended frames.

Set in .env: AWS_* for Rekognition, S3_CATALOG_BUCKET (and optional S3_CATALOG_ACCESS_KEY_* for a friend’s bucket),
S3_CATALOG_PREFIX=  (empty for images at bucket root), EMBEDDING_DIM=8

Run from project root with the project venv so imports resolve:
  .venv/bin/python webcam_s3_eyewear.py

On macOS, grant Camera access to Terminal / iTerm / Cursor (or the capture will fail).
If the window does not open or the device is wrong, try:  --device 1

If Rekognition rejects the frame (pose/quality), retry with better light or:  --no-quality

Synthetic tags/embeddings are used per product id so the hybrid ranker can differentiate rows until you
add a real manifest with vectors. For best “similar to face” you need a shared embedding model.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
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


def _open_video_capture(device: int) -> cv2.VideoCapture:
    """
    On macOS, OpenCV often needs the AVFoundation backend for the built-in camera.
    Fall back to the default backend if that fails.
    """
    if platform.system() == "Darwin":
        cap = cv2.VideoCapture(device, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(device)


def _print_summary(out: dict[str, Any]) -> None:
    if not out.get("ok"):
        return
    sh = out.get("face_shape", "?")
    rules = out.get("rules") or {}
    tags = rules.get("preferred_frame_tags", [])
    print(
        f"\n--- Your estimated face shape: {sh.upper()} ---\n"
        f"Rule-based frame styles to prefer: {', '.join(tags)}\n"
        f"(Open the JSON below for presigned S3 image links to recommended eyewear.)\n",
        file=sys.stderr,
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Webcam → face shape + S3 eyewear catalog")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--top", type=int, default=5, dest="top_n")
    p.add_argument("--no-quality", action="store_true")
    p.add_argument("--mirror-preview", action="store_true")
    p.add_argument(
        "--regional-json",
        type=Path,
        default=ROOT / "regional_affinity.json",
        help="Optional JSON for regional weighting; omit if not a file",
    )
    p.add_argument(
        "--user-json",
        type=Path,
        default=None,
    )
    args = p.parse_args()
    s = get_settings()

    if not os.environ.get("AWS_ACCESS_KEY_ID") and not os.environ.get("AWS_SESSION_TOKEN"):
        print(
            json.dumps(
                {
                    "error": "Set AWS_ACCESS_KEY_ID (and secret) in .env for Rekognition, "
                    "or use an AWS profile / SSO that boto3 can resolve",
                },
            ),
            file=sys.stderr,
        )
        return 1

    if not s.s3_catalog_bucket:
        print(
            json.dumps(
                {
                    "error": "Set S3_CATALOG_BUCKET in .env to the bucket with lusmt*_*_*.jpg files",
                },
            ),
            file=sys.stderr,
        )
        return 1

    products = build_lusmt_flat_catalog(s)
    if not products:
        print(
            json.dumps(
                {
                    "error": "No lusmt*_*_*.jpg files found. Check S3_CATALOG_PREFIX and list with: "
                    "python scripts/s3_list_prefix.py",
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

    cap = _open_video_capture(args.device)
    if not cap.isOpened():
        print(
            json.dumps(
                {
                    "error": "Could not open camera",
                    "device": args.device,
                    "hint": (
                        "On macOS: System Settings → Privacy & Security → Camera → enable for your terminal/IDE. "
                        "Try --device 1. For batch tests without a camera, use: python image_s3_recommend.py <photo.jpg>"
                    ),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1

    print("Camera on. One face, good light. **SPACE** = capture, **Q** = quit.\n", file=sys.stderr)
    path: str | None = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                return 1
            show = cv2.flip(frame, 1) if args.mirror_preview else frame
            cv2.imshow("Eyewear match — SPACE capture, Q quit", show)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return 0
            if key == ord(" "):
                fd, path = tempfile.mkstemp(suffix=".jpg", prefix="eyewear_")
                os.close(fd)
                cv2.imwrite(path, frame)
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not path:
        return 0

    try:
        with open(path, "rb") as f:
            image_bytes = f.read()
        w, h = _image_size_from_bytes(image_bytes)
        try:
            prototypes = load_face_shape_prototypes(s.embedding_dim)
        except ValueError as e:
            print(json.dumps({"error": str(e)}), file=sys.stderr)
            return 1
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
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

    _print_summary(out)
    print(json.dumps(out, indent=2, default=str))
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
