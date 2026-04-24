"""
Amazon Rekognition face similarity: collection + index + search by image.

Flow:
  1) init    — create a face collection (stores face vectors in AWS).
  2) index   — add a face from a local image (ExternalImageId = your user/photo id).
  3) search  — find the most similar indexed faces for a query image.

Requires IAM: rekognition:CreateCollection, IndexFaces, SearchFacesByImage (and
DescribeCollection if you extend this). Same credentials as facial_recognition.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv


def get_client() -> Any:
    region = os.environ.get("AWS_DEFAULT_REGION") or "ap-south-1"
    return boto3.client("rekognition", region_name=region)


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def cmd_init(args: argparse.Namespace) -> int:
    cid = args.collection_id.strip()
    try:
        r = get_client().create_collection(CollectionId=cid)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "ResourceAlreadyExistsException":
            print(json.dumps({"ok": True, "message": "Collection already exists", "CollectionId": cid}))
            return 0
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, "status": r.get("StatusCode"), "CollectionId": cid, "arn": r.get("CollectionArn")}, default=str))
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    cid = args.collection_id.strip()
    ext_id = args.external_id.strip()
    b = read_bytes(args.image)
    try:
        r = get_client().index_faces(
            CollectionId=cid,
            Image={"Bytes": b},
            ExternalImageId=ext_id,
            MaxFaces=1,
            QualityFilter="AUTO",
        )
    except (ClientError, BotoCoreError) as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 1
    recs = r.get("FaceRecords") or []
    if not recs:
        print(json.dumps({"error": "No face indexed (empty FaceRecords). Try another image or quality."}), file=sys.stderr)
        return 1
    f = recs[0].get("Face", {})
    print(
        json.dumps(
            {
                "ExternalImageId": ext_id,
                "FaceId": f.get("FaceId"),
                "BoundingBox": f.get("BoundingBox"),
                "Confidence": f.get("Confidence"),
            },
            indent=2,
            default=str,
        )
    )
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    cid = args.collection_id.strip()
    b = read_bytes(args.image)
    try:
        r = get_client().search_faces_by_image(
            CollectionId=cid,
            Image={"Bytes": b},
            FaceMatchThreshold=args.threshold,
            MaxFaces=args.max_faces,
        )
    except (ClientError, BotoCoreError) as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 1
    matches = []
    for m in r.get("FaceMatches") or []:
        f = m.get("Face", {}) or {}
        matches.append(
            {
                "Similarity": m.get("Similarity"),
                "FaceId": f.get("FaceId"),
                "ExternalImageId": f.get("ExternalImageId"),
            }
        )
    print(json.dumps({"FaceMatches": matches, "SearchedFaceBoundingBox": r.get("SearchedFaceBoundingBox")}, indent=2, default=str))
    return 0


def main() -> int:
    load_dotenv(Path(__file__).resolve().parent / ".env")
    if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get("AWS_SECRET_ACCESS_KEY"):
        print(json.dumps({"error": "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env"}), file=sys.stderr)
        return 1

    default_col = (os.environ.get("REKOGNITION_COLLECTION_ID") or "rs2-faces").strip()

    p = argparse.ArgumentParser(description="Rekognition face collection similarity search")
    sub = p.add_subparsers(dest="cmd", required=True)

    s0 = sub.add_parser("init", help="Create collection (run once per project/region)")
    s0.add_argument("--collection-id", default=default_col, help="Collection name (alphanumeric, hyphens, underscore)")
    s0.set_defaults(func=cmd_init)

    s1 = sub.add_parser("index", help="Index one face from a file into the collection")
    s1.add_argument("--collection-id", default=default_col)
    s1.add_argument("image", help="Path to image")
    s1.add_argument("external_id", help="Your id for this face (e.g. user-123 or photo_id)")
    s1.set_defaults(func=cmd_index)

    s2 = sub.add_parser("search", help="Search collection for faces similar to this image")
    s2.add_argument("--collection-id", default=default_col)
    s2.add_argument("--threshold", type=float, default=80.0, help="Min similarity 0-100 (default 80)")
    s2.add_argument("--max-faces", type=int, default=5)
    s2.add_argument("image", help="Query image path")
    s2.set_defaults(func=cmd_search)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
