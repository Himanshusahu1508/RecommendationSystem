"""
Merge **CLIP-only** preference channel with the rule-based score
(inner ``w_prompt`` cleared in hybrid to avoid double-counting).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import ranking_signals as rs
from app.preference import clip_scoring


def has_clip_style_input(
    style_text: str | None,
    style_reference_image_bytes: bytes | None,
) -> bool:
    """True if the user provided text and/or a reference image for CLIP encoding."""
    if style_reference_image_bytes:
        return True
    if (style_text or "").strip():
        return True
    return False


def _candidates_have_clip_embedding(candidates: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(p.get("clip_embedding"), list) and len(p.get("clip_embedding") or []) > 0
        for p in candidates
    )


def clip_hybrid_viable(
    style_text: str | None,
    style_reference_image_bytes: bytes | None,
    *,
    use_clip: bool,
    candidates: list[dict[str, Any]],
) -> tuple[bool, str]:
    """
    Whether CLIP preference can run. Returns (viable, reason when false).
    """
    if not has_clip_style_input(style_text, style_reference_image_bytes):
        return False, "no_style_text_or_image"
    if not use_clip:
        return False, "clip_preference_disabled"
    if not clip_scoring.is_available():
        return False, "clip_backend_unavailable"
    if not _candidates_have_clip_embedding(candidates):
        return False, "no_product_clip_embedding"
    return True, "ok"


def preference_scores_for_products(
    candidates: list[dict[str, Any]],
    style_text: str | None,
    *,
    style_reference_image_bytes: bytes | None,
) -> tuple[dict[str, float], str]:
    """
    CLIP-only preference: cosine similarity vs per-product ``clip_embedding``, mapped to [0,1].
    """
    out: dict[str, float] = {}
    q = clip_scoring.encode_user_style(
        (style_text or "").strip() or None,
        style_reference_image_bytes,
    )
    if q is None:
        for p in candidates:
            out[str(p.get("id", ""))] = 0.5
        return out, "clip_neutral"
    for p in candidates:
        pid = str(p.get("id", ""))
        out[pid] = clip_scoring.clip_preference_01(p, q)
    return out, "clip"


def _minmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [0.5] * len(scores)
    return [(x - lo) / (hi - lo) for x in scores]


def apply_preference_hybrid(
    ranked_inner: list[dict[str, Any]],
    preference_by_id: dict[str, float],
    w_preference: float,
    w_face_rules: float,
) -> list[dict[str, Any]]:
    """
    ranked_inner: output of personalized_rank with **w_prompt=0** (no text in inner sum).
    preference_by_id: 0..1 per product id.
    Renormalize w_preference + w_face_rules to sum 1.
    """
    t = w_preference + w_face_rules
    if t <= 0:
        t = 1.0
    wp, wf = w_preference / t, w_face_rules / t
    raw = [float(r.get("score", 0.0)) for r in ranked_inner]
    mm = _minmax(raw)
    merged: list[dict[str, Any]] = []
    for i, row in enumerate(ranked_inner):
        r = dict(row)
        s_in = mm[i] if i < len(mm) else 0.5
        pid = str(r.get("id", ""))
        s_pref = float(preference_by_id.get(pid, 0.5))
        final = wp * s_pref + wf * s_in
        sb = dict(r.get("score_breakdown") or {})
        sb["hybrid"] = {
            "w_preference": round(wp, 4),
            "w_face_rules_inner": round(wf, 4),
            "preference_channel": round(s_pref, 4),
            "face_rules_inner_norm": round(s_in, 4),
        }
        r["score"] = round(final, 5)
        r["score_breakdown"] = sb
        merged.append(r)
    merged.sort(key=lambda x: -float(x.get("score", 0.0)))
    return merged


def ranking_weights_sans_style_prompt(rw: rs.RankingWeights) -> rs.RankingWeights:
    """Same weights with ``w_prompt`` set to 0; :meth:`RankingWeights.normalized` redistributes."""
    return replace(rw, w_prompt=0.0)
