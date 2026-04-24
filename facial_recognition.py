"""
AWS Rekognition DetectFaces: read a local image, return the raw FaceDetails
payload (Attributes=ALL) as JSON.
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


def get_rekognition_client() -> Any:
    """
    boto3 client for Rekognition. Uses AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
    and AWS_DEFAULT_REGION (defaults to ap-south-1).
    """
    region = os.environ.get("AWS_DEFAULT_REGION") or "ap-south-1"
    return boto3.client("rekognition", region_name=region)


def read_image_file(local_path: str) -> bytes:
    """Load a local file as bytes for the DetectFaces API."""
    with open(local_path, "rb") as f:
        return f.read()


def call_detect_faces(client: Any, image_bytes: bytes) -> dict[str, Any]:
    """
    Call DetectFaces with Attributes=ALL. Response includes FaceDetails (each
    with BoundingBox, AgeRange, Gender, Landmarks, Emotions, Pose, etc.).
    """
    return client.detect_faces(
        Image={"Bytes": image_bytes},
        Attributes=["ALL"],
    )


def face_details_payload(response: dict[str, Any]) -> dict[str, Any]:
    """
    Keep only the shape you want: {"FaceDetails": [ ... ]} (API face objects).
    """
    return {"FaceDetails": list(response.get("FaceDetails", []))}


def json_from_rekognition(image_bytes: bytes) -> dict[str, Any]:
    """
    Run Rekognition and return {"FaceDetails": ...}. On client errors, raises.
    """
    client = get_rekognition_client()
    response = call_detect_faces(client, image_bytes)
    return face_details_payload(response)


def main() -> int:
    load_dotenv(Path(__file__).resolve().parent / ".env")

    p = argparse.ArgumentParser(
        description="Print Rekognition FaceDetails JSON for a local image.",
    )
    p.add_argument(
        "image",
        help="Path to a local image (JPEG/PNG) supported by Rekognition.",
    )
    args = p.parse_args()
    path = args.image

    if not os.path.isfile(path):
        print(json.dumps({"error": f"File not found: {path}"}, indent=2), file=sys.stderr)
        return 1

    if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get("AWS_SECRET_ACCESS_KEY"):
        print(
            json.dumps(
                {
                    "error": "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env or the shell.",
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 1

    try:
        raw = read_image_file(path)
        out = json_from_rekognition(raw)
    except (ClientError, BotoCoreError, OSError) as e:
        print(json.dumps({"error": str(e)}, indent=2), file=sys.stderr)
        return 1

    # Same structure as the API face list: BoundingBox, AgeRange, Landmarks, etc.
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
