#!/usr/bin/env python3
"""
Build CLIP image embeddings + FAISS index from a local folder of images (same stack as eyewear_recommender).

  python scripts/build_clip_faiss_index.py --images-dir /path/to/Glass/Sunglass/all_images --out-dir artifacts/sunglass

Writes:
  - ``index.faiss`` — FAISS IndexFlatIP
  - ``metadata.json`` — list of {id, relpath, clip_embedding} for merging into the RS2 catalog
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from eyewear_recommender import config
from eyewear_recommender.clip_backend import load_clip
from eyewear_recommender.faiss_index import build_index

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _list_images(d: Path) -> list[Path]:
    return sorted(
        p for p in d.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", type=Path, required=True, help="Folder of product images (recursive)")
    p.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "clip_index")
    p.add_argument("--clip-model", default=config.CLIP_MODEL_ID)
    args = p.parse_args()

    images_dir: Path = args.images_dir
    if not images_dir.is_dir():
        print(f"Not a directory: {images_dir}", file=sys.stderr)
        return 1
    paths = _list_images(images_dir)
    if not paths:
        print(f"No images under {images_dir}", file=sys.stderr)
        return 1

    print(f"Loading CLIP: {args.clip_model}", file=sys.stderr)
    clip = load_clip(model_id=args.clip_model)
    embs: list[np.ndarray] = []
    items: list[dict] = []
    for i, path in enumerate(paths):
        if i % 20 == 0:
            print(f"  [{i+1}/{len(paths)}] {path.name}", file=sys.stderr, flush=True)
        rel = str(path.relative_to(images_dir)).replace("\\", "/")
        im = Image.open(path).convert("RGB")
        e = clip.encode_image(im)
        embs.append(e)
        pid = path.stem.split("_")[0] if "_" in path.stem else f"item_{i:05d}"
        items.append(
            {
                "id": pid,
                "relpath": rel,
                "clip_embedding": e.astype(float).round(6).tolist(),
            },
        )

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    mat = np.stack(embs, axis=0).astype(np.float32)
    idx = build_index(mat)
    idx_path = out_dir / "index.faiss"
    idx.write(idx_path)
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "clip_model_id": clip.model_id,
                "dim": clip.dim,
                "count": len(items),
                "items": items,
            },
            f,
            indent=2,
        )
    print(f"Wrote {idx_path} and {meta_path} ({len(items)} vectors, dim={clip.dim})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
