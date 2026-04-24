"""
Hybrid glasses recommendation: (1) exactly one face, (2) rule filter from face shape,
(3) personalized rank: face + popularity + demographics + last order + region cohort.

Replace JSON files with your DB / feature store in production.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from PIL import Image

import ear_utils
import ranking_signals as rs
from face_checks import single_face_or_error
from facial_recognition import call_detect_faces, get_rekognition_client, read_image_file

# Heuristic: ratio = jaw_w / face_h (see _jaw_width_and_face_height_px). **Higher** = wider jaw vs eye–chin height.
# Six buckets in **increasing** order of ratio: rectangle, square, round, oval, long, heart.
# Five cutoffs split the line: R < b0, b0≤R<b1, …, R ≥ b4 → heart.
FACE_JAW_WH_BOUNDS: tuple[float, float, float, float, float] = (0.60, 0.70, 0.85, 0.95, 1.1)
FACE_SHAPE_ORDER: tuple[str, ...] = (
    "rectangle",
    "square",
    "round",
    "oval",
    "long",
    "heart",
)

# Prototype vectors (same length as catalog embeddings). One per face shape. Swap for model outputs later.
FACE_SHAPE_PROTOTYPES: dict[str, list[float]] = {
    "rectangle": [0.88, 0.08, 0.12, 0.0, 0.2, 0.9, 0.0, 0.15],
    "square": [0.75, 0.15, 0.2, 0.1, 0.3, 0.75, 0.1, 0.3],
    "round": [0.9, 0.1, 0.2, 0.0, 0.3, 0.85, 0.0, 0.4],
    "oval": [0.45, 0.45, 0.5, 0.4, 0.5, 0.45, 0.45, 0.5],
    "long": [0.1, 0.85, 0.8, 0.2, 0.1, 0.2, 0.25, 0.6],
    "heart": [0.2, 0.55, 0.7, 0.15, 0.2, 0.35, 0.8, 0.5],
}


def _px_dist(
    a: tuple[float, float],
    b: tuple[float, float],
    iw: float,
    ih: float,
) -> float:
    return math.hypot((a[0] - b[0]) * iw, (a[1] - b[1]) * ih)


def _jaw_width_and_face_height_px(
    face_detail: dict[str, Any],
    image_width: int,
    image_height: int,
) -> tuple[float, float] | None:
    """
    Jaw width and vertical face height (eye line → chin) in **pixels** from landmarks.
    Used for face shape + height/width ratio in ranking. Returns None if landmarks missing.
    """
    lm = ear_utils.landmarks_by_type(face_detail)
    w, h = float(image_width), float(image_height)
    for req in ("midJawlineLeft", "midJawlineRight", "chinBottom"):
        if req not in lm:
            return None
    jaw_w = _px_dist(lm["midJawlineLeft"], lm["midJawlineRight"], w, h)
    if "leftEyeLeft" in lm and "rightEyeRight" in lm:
        eye_y = (lm["leftEyeLeft"][1] + lm["rightEyeRight"][1]) / 2.0
    elif "nose" in lm:
        eye_y = float(lm["nose"][1]) - 0.06
    else:
        return None
    chin_y = float(lm["chinBottom"][1])
    face_h = abs(eye_y - chin_y) * h
    if face_h < 1.0:
        return None
    return jaw_w, face_h


def face_geometry_summary(
    face_detail: dict[str, Any],
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    """
    Exposes face proportion metrics for ranking and debugging.

    * **height_over_width** = face_h / jaw_w (larger ≈ more elongated / taller-looking face in image)
    * **jaw_width_to_face_height** = jaw_w / face_h (this is what ``infer_face_shape`` thresholds on; higher ≈ rounder)
    * **bbox_height_over_width** = Rekognition face box aspect (normalized 0..1; comparable across images)
    """
    out: dict[str, Any] = {
        "height_over_width": None,
        "jaw_width_to_face_height": None,
    }
    wh = _jaw_width_and_face_height_px(face_detail, image_width, image_height)
    if wh is not None:
        jaw_w, face_h = wh
        out["jaw_width_px"] = round(jaw_w, 2)
        out["face_height_eye_to_chin_px"] = round(face_h, 2)
        out["height_over_width"] = face_h / max(jaw_w, 1e-9)
        out["jaw_width_to_face_height"] = jaw_w / max(face_h, 1e-9)
    bb = face_detail.get("BoundingBox") or {}
    if isinstance(bb, dict) and bb.get("Width") and bb.get("Height"):
        bw, bh = float(bb["Width"]), float(bb["Height"])
        if bw > 1e-9:
            out["bbox_height_over_width"] = bh / bw
    return out


def validate_jaw_wh_bounds(
    bounds: tuple[float, ...],
    *,
    min_v: float = 0.2,
    max_v: float = 1.8,
) -> str | None:
    """Return an error string if invalid; None if b0<…<b4 with five values."""
    if len(bounds) != 5:
        return "Need exactly 5 cutoffs (b0…b4)."
    for x in bounds:
        if not (min_v <= x <= max_v):
            return f"Each cutoff must be between {min_v} and {max_v}."
    if not all(bounds[i] < bounds[i + 1] for i in range(4)):
        return "Cutoffs must be strictly increasing: b0 < b1 < b2 < b3 < b4."
    return None


def face_shape_from_jaw_ratio(
    ratio: float,
    bounds: tuple[float, float, float, float, float] | None = None,
) -> str:
    """
    Map jaw_w/face_h to one of FACE_SHAPE_ORDER buckets (increasing ratio).
    """
    b0, b1, b2, b3, b4 = bounds if bounds is not None else FACE_JAW_WH_BOUNDS
    if ratio < b0:
        return "rectangle"
    if ratio < b1:
        return "square"
    if ratio < b2:
        return "round"
    if ratio < b3:
        return "oval"
    if ratio < b4:
        return "long"
    return "heart"


def infer_face_shape(
    face_detail: dict[str, Any],
    image_width: int,
    image_height: int,
    jaw_wh_bounds: tuple[float, float, float, float, float] | None = None,
) -> str:
    """
    Heuristic face shape from Rekognition jaw/eye/chin landmarks.
    Six shape buckets from jaw/height ratio; 'oval' if geometry missing.
    """
    wh = _jaw_width_and_face_height_px(face_detail, image_width, image_height)
    if wh is None:
        return "oval"
    jaw_w, face_h = wh
    return face_shape_from_jaw_ratio(jaw_w / face_h, jaw_wh_bounds)


def rules_for_face_shape(face_shape: str) -> dict[str, Any]:
    """Rule layer: which frame tags to prefer (hybrid first stage)."""
    m: dict[str, list[str]] = {
        "rectangle": ["angular", "rectangular", "narrow", "metal", "geometric", "bold", "classic"],
        "square": ["wayfarer", "geometric", "classic", "acetate", "aviator", "metal", "round"],
        "round": ["angular", "rectangular", "geometric", "narrow", "metal", "bold"],
        "oval": ["wayfarer", "classic", "acetate", "aviator", "round", "rectangular", "geometric"],
        "long": ["round", "soft", "keyhole", "wayfarer", "acetate", "oval", "cat_eye"],
        "heart": ["round", "soft", "aviator", "cat_eye", "keyhole", "oval", "wayfarer"],
    }
    return {
        "face_shape": face_shape,
        "preferred_frame_tags": m.get(face_shape, m["oval"]),
    }


def load_catalog(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Catalog must be a JSON array of products")
    return data


def filter_by_face_shape(products: list[dict[str, Any]], face_shape: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in products:
        fs = p.get("face_shapes") or []
        if "all" in fs or face_shape in fs:
            out.append(p)
    return out


def load_regional(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _catalog_by_id(products: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(p["id"]): p for p in products if p.get("id") is not None}


def _image_size_from_bytes(image_bytes: bytes) -> tuple[int, int]:
    with Image.open(BytesIO(image_bytes)) as im:
        return int(im.size[0]), int(im.size[1])


def _image_size(path: str) -> tuple[int, int]:
    with Image.open(path) as im:
        return int(im.size[0]), int(im.size[1])


def recommend_from_bytes(
    image_bytes: bytes,
    image_width: int,
    image_height: int,
    products: list[dict[str, Any]],
    *,
    regional: dict[str, Any] | None = None,
    check_quality: bool = True,
    top_n: int = 5,
    user_context: dict[str, Any] | None = None,
    face_shape_prototypes: dict[str, list[float]] | None = None,
    ranking_weights: rs.RankingWeights | None = None,
    style_prompt: str | None = None,
    jaw_wh_bounds: tuple[float, float, float, float, float] | None = None,
    style_reference_image_bytes: bytes | None = None,
    use_preference_hybrid: bool = True,
    hybrid_w_preference: float = 0.6,
    hybrid_w_face: float = 0.4,
    use_clip_preference: bool = False,
) -> dict[str, Any]:
    """
    Core recommendation: Rekognition on bytes + in-memory catalog and regional data.
    Used by the HTTP API; CLI uses recommend_pipeline on disk paths.
    If ``jaw_wh_bounds`` is set (5 increasing cutoffs for jaw_w/face_h), it overrides
    :data:`FACE_JAW_WH_BOUNDS` for face-shape bucketing.

    **CLIP preference hybrid** (optional): when ``style_prompt`` (free text) and/or a
    **style reference image** is provided *and* CLIP is enabled and the catalog has
    ``clip_embedding`` rows, the final score blends CLIP preference with the **inner
    rule-based** score (``w_prompt`` cleared in the inner pass). Configure weights via
    ``hybrid_w_*`` and ``use_clip_preference`` (e.g. from :class:`~app.config.Settings`).
    There is no separate keyword or substring search over the catalog.
    When hybrid does not run, :func:`ranking_signals.personalized_rank` runs without
    style text tokens (``style_prompt`` / ``keyword`` slots neutral).
    """
    if jaw_wh_bounds is not None:
        be = validate_jaw_wh_bounds(jaw_wh_bounds)
        if be:
            return {
                "ok": False,
                "stage": "invalid_jaw_wh_bounds",
                "error": be,
            }
    w, h = image_width, image_height
    client = get_rekognition_client()
    response = call_detect_faces(client, image_bytes)
    details = list(response.get("FaceDetails", []))

    face, err, count = single_face_or_error(details)
    if err is not None or face is None:
        return {
            "ok": False,
            "stage": "single_face_check",
            "error": err,
            "face_count": count,
        }

    if check_quality:
        q = ear_utils.rule_based_frame_check(face, w, h)
        if not q.get("recommend"):
            return {
                "ok": False,
                "stage": "quality",
                "error": "quality_check_failed",
                "reasons": q.get("reasons"),
                "quality_detail": q,
            }

    face_geometry = face_geometry_summary(face, w, h)
    hw = face_geometry.get("height_over_width")
    height_over_width = float(hw) if isinstance(hw, (int, float)) and hw is not None else None

    _bounds_eff = jaw_wh_bounds if jaw_wh_bounds is not None else FACE_JAW_WH_BOUNDS
    shape = infer_face_shape(face, w, h, jaw_wh_bounds=jaw_wh_bounds)
    rules = rules_for_face_shape(shape)

    candidates = filter_by_face_shape(products, shape)
    if not candidates:
        return {
            "ok": True,
            "face_shape": shape,
            "face_shape_buckets": {
                "order": list(FACE_SHAPE_ORDER),
                "jaw_width_to_height_cutoffs": list(_bounds_eff),
                "jaw_width_to_height_cutoffs_source": "custom"
                if jaw_wh_bounds is not None
                else "default",
            },
            "face_geometry": face_geometry,
            "rules": rules,
            "recommendations": [],
            "note": "No catalog items for this face shape; widen face_shapes in catalog.",
        }

    proto = face_shape_prototypes or FACE_SHAPE_PROTOTYPES
    query_embedding = proto.get(shape, proto.get("oval", FACE_SHAPE_PROTOTYPES["oval"]))

    demo_face = rs.demographics_from_face(face)
    ctx = rs.UserContext.from_dict(user_context)
    demographics_used = ctx.merge_demographics(demo_face)

    by_id = _catalog_by_id(products)
    lo_ids = list(ctx.last_order_product_ids)
    last_products = [by_id[i] for i in lo_ids if i in by_id]

    rw = ranking_weights or rs.RankingWeights()
    wnorm = rw.normalized()

    from app.preference.hybrid_merge import (
        apply_preference_hybrid,
        clip_hybrid_viable,
        preference_scores_for_products,
        ranking_weights_sans_style_prompt,
    )

    viable, viable_reason = clip_hybrid_viable(
        style_prompt,
        style_reference_image_bytes,
        use_clip=use_clip_preference,
        candidates=candidates,
    )
    hp = min(1.0, max(0.0, float(hybrid_w_preference)))
    hf = min(1.0, max(0.0, float(hybrid_w_face)))
    if use_preference_hybrid and viable and (hp + hf) > 1e-9:
        rw_sans = ranking_weights_sans_style_prompt(rw)
        ranked_inner = rs.personalized_rank(
            candidates,
            rules,
            query_embedding,
            demographics_used,
            last_products,
            ctx.region,
            regional,
            rw_sans,
            height_over_width=height_over_width,
            style_prompt=None,
            keyword_query=None,
        )
        pref_map, pref_mode = preference_scores_for_products(
            candidates,
            style_prompt,
            style_reference_image_bytes=style_reference_image_bytes,
        )
        ranked_merged = apply_preference_hybrid(
            ranked_inner,
            pref_map,
            hp,
            hf,
        )
        ranked = ranked_merged[:top_n]
        hybrid_meta = {
            "enabled": True,
            "preference_mode": pref_mode,
            "w_preference": hp,
            "w_face_rules": hf,
            "clip": True,
        }
    else:
        ranked = rs.personalized_rank(
            candidates,
            rules,
            query_embedding,
            demographics_used,
            last_products,
            ctx.region,
            regional,
            rw,
            height_over_width=height_over_width,
            style_prompt=None,
            keyword_query=None,
        )[:top_n]
        if not use_preference_hybrid:
            _reason = "use_preference_hybrid is False"
        elif (hp + hf) <= 1e-9:
            _reason = "zero hybrid weights"
        else:
            _reason = viable_reason
        hybrid_meta = {
            "enabled": False,
            "reason": _reason,
        }

    out: dict[str, Any] = {
        "ok": True,
        "face_shape": shape,
        "face_shape_buckets": {
            "order": list(FACE_SHAPE_ORDER),
            "jaw_width_to_height_cutoffs": list(_bounds_eff),
            "jaw_width_to_height_cutoffs_source": "custom"
            if jaw_wh_bounds is not None
            else "default",
        },
        "face_geometry": face_geometry,
        "rules": rules,
        "demographics": demographics_used,
        "user": {
            "user_id": ctx.user_id,
            "region": ctx.region,
            "last_order_product_ids": lo_ids,
        },
        "ranking_weights": wnorm,
        "clip_style_text": (style_prompt or "").strip(),
        "hybrid": hybrid_meta,
        "recommendations": ranked,
    }
    return out


def recommend_pipeline(
    image_path: str,
    catalog_path: Path,
    *,
    check_quality: bool = True,
    top_n: int = 5,
    user_context: dict[str, Any] | None = None,
    regional_path: Path | None = None,
    ranking_weights: rs.RankingWeights | None = None,
) -> dict[str, Any]:
    """
    Full flow: image file → read bytes → same as recommend_from_bytes.
    """
    try:
        products = load_catalog(catalog_path)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        return {"ok": False, "stage": "catalog", "error": str(e)}

    image_bytes = read_image_file(image_path)
    w, h = _image_size_from_bytes(image_bytes)
    regional = load_regional(regional_path)
    return recommend_from_bytes(
        image_bytes,
        w,
        h,
        products,
        regional=regional,
        check_quality=check_quality,
        top_n=top_n,
        user_context=user_context,
        ranking_weights=ranking_weights,
    )


def main() -> int:
    load_dotenv(Path(__file__).resolve().parent / ".env")
    root = Path(__file__).resolve().parent
    default_cat = root / "glasses_catalog.json"
    default_reg = root / "regional_affinity.json"

    p = argparse.ArgumentParser(
        description="Glasses recommend: face + demographics + last order + region",
    )
    p.add_argument("image", help="Path to a local image")
    p.add_argument("--catalog", type=Path, default=default_cat, help="JSON product catalog")
    p.add_argument(
        "--user-json",
        type=Path,
        default=None,
        help="User context: region, last_order_product_ids, optional gender/age overrides",
    )
    p.add_argument(
        "--regional-json",
        type=Path,
        default=default_reg,
        help="Region → product_affinity + tag_affinity (use nonexistent path to disable)",
    )
    p.add_argument("--top", type=int, default=5, dest="top_n")
    p.add_argument("--no-quality", action="store_true", help="Skip pose/EAR/quality gate")
    args = p.parse_args()

    if not os.path.isfile(args.image):
        print(json.dumps({"error": f"File not found: {args.image}"}), file=sys.stderr)
        return 1
    if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get("AWS_SECRET_ACCESS_KEY"):
        print(json.dumps({"error": "Set AWS creds in .env"}), file=sys.stderr)
        return 1

    user_context: dict[str, Any] | None = None
    if args.user_json and args.user_json.is_file():
        with open(args.user_json, encoding="utf-8") as uf:
            user_context = json.load(uf)
    reg_path: Path | None = args.regional_json if args.regional_json.is_file() else None

    try:
        out = recommend_pipeline(
            args.image,
            args.catalog,
            check_quality=not args.no_quality,
            top_n=args.top_n,
            user_context=user_context,
            regional_path=reg_path,
        )
    except (ClientError, BotoCoreError, OSError) as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return 1

    print(json.dumps(out, indent=2, default=str))
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
