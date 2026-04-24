"""
Scoring components for personalized glasses ranking.

- Face + rules: tag match + **cosine similarity** of catalog `embedding` to the
  face-shape query vector (same slot as your “similarity search” style signal; combined in `w_face`).
- Demographics: align Rekognition/override gender + age with product metadata.
- Color (lifecycle): favor vibrant / bold for Kids & Teenagers, neutral for Adult, plain / muted for Old.
- Purchase: boost frames similar to the user’s last order (tag / style overlap).
- Region: boost items popular with other buyers in the same region.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RankingWeights:
    """Weights should be non-negative; the pipeline renormalizes to sum to 1."""

    w_face: float = 0.20  # face-shape rules + tag + style embedding
    w_pop: float = 0.08  # global catalog popularity
    w_demo: float = 0.12  # gender + age fit
    w_purchase: float = 0.20  # last order similarity
    w_region: float = 0.16  # regional cohort
    w_prompt: float = 0.18  # style + keyword prompts vs product text/tags (user preference)
    w_color: float = 0.06  # age-lifecycle color palette (vibrant vs plain)

    def normalized(self) -> dict[str, float]:
        t = (
            self.w_face
            + self.w_pop
            + self.w_demo
            + self.w_purchase
            + self.w_region
            + self.w_prompt
            + self.w_color
        )
        if t <= 0:
            t = 1.0
        return {
            "w_face": self.w_face / t,
            "w_pop": self.w_pop / t,
            "w_demo": self.w_demo / t,
            "w_purchase": self.w_purchase / t,
            "w_region": self.w_region / t,
            "w_prompt": self.w_prompt / t,
            "w_color": self.w_color / t,
        }


def demographics_from_face(face_detail: dict[str, Any]) -> dict[str, Any]:
    """
    Use Rekognition Gender + AgeRange when present. Caller may override with
    account/profile data for higher trust than a single photo.
    """
    g = face_detail.get("Gender") or {}
    ar = face_detail.get("AgeRange") or {}
    gender_val = (g.get("Value") or "").strip()
    low = int(ar.get("Low", 0) or 0)
    high = int(ar.get("High", 0) or 0)
    age_mid = (low + high) / 2.0 if (low and high) else None
    band = _age_to_band(age_mid) if age_mid is not None else None
    age_lifecycle = age_lifecycle_bracket(age_mid) if age_mid is not None else None
    return {
        "gender_label": gender_val or None,
        "gender_confidence": float(g.get("Confidence") or 0.0),
        "age_low": low or None,
        "age_high": high or None,
        "age_mid": age_mid,
        "age_band": band,
        "age_lifecycle": age_lifecycle,
    }


def _age_to_band(age: float) -> str:
    if age < 22:
        return "youth"
    if age < 45:
        return "adult"
    if age < 60:
        return "mature"
    return "senior"


def age_lifecycle_bracket(age: float) -> str:
    """
    UI-facing lifecycle labels (Kids, Teenager, Adult, Old).
    """
    if age < 13:
        return "Kids"
    if age < 18:
        return "Teenager"
    if age < 60:
        return "Adult"
    return "Old"


# Product age_bands (any case) that count as a match for each lifecycle label
_LIFECYCLE_TO_PRODUCT_BANDS: dict[str, frozenset[str]] = {
    "Kids": frozenset(
        {
            "kids",
            "child",
            "children",
            "youth",
            "all",
        },
    ),
    "Teenager": frozenset(
        {
            "teen",
            "teenager",
            "youth",
            "adolescent",
            "all",
        },
    ),
    "Adult": frozenset({"adult", "mature", "youth", "all"}),
    "Old": frozenset({"senior", "mature", "old", "elder", "all"}),
}


def _gender_bucket(label: str | None) -> str | None:
    if not label:
        return None
    u = label.lower()
    if u in ("male", "m"):
        return "M"
    if u in ("female", "f"):
        return "F"
    return "all"


def _normalize_product_gender(t: str | None) -> str:
    if not t:
        return "all"
    u = t.strip().upper()
    if u in ("ALL", "ANY", "UNISEX"):
        return "all"
    if u in ("M", "MALE"):
        return "M"
    if u in ("F", "FEMALE"):
        return "F"
    return "all"


def score_demographic_match(
    product: dict[str, Any],
    demo: dict[str, Any],
) -> float:
    """
    0..1: product `target_gender` + `age_bands` vs inferred user demo.
    Missing user gender/age → soften toward neutral.
    """
    tg = _normalize_product_gender(str(product.get("target_gender") or "all"))
    ug = _gender_bucket(demo.get("gender_label"))
    if tg == "all":
        s_g = 1.0
    elif ug is None:
        s_g = 0.72
    else:
        s_g = 1.0 if ug == tg else 0.32

    bands_raw: list[str] = list(product.get("age_bands") or ["all"])
    bands_norm = {str(b).lower().strip() for b in bands_raw}
    if "all" in bands_norm or not bands_norm:
        s_a = 1.0
    elif demo.get("age_lifecycle"):
        lc = str(demo["age_lifecycle"])
        allowed = _LIFECYCLE_TO_PRODUCT_BANDS.get(lc, frozenset())
        s_a = 1.0 if (bands_norm & allowed) else 0.38
    elif demo.get("age_band"):
        s_a = 1.0 if demo["age_band"] in bands_raw else 0.38
    elif demo.get("age_mid") is not None:
        ab = _age_to_band(float(demo["age_mid"]))
        s_a = 1.0 if ab in bands_raw else 0.38
    else:
        s_a = 0.72
    return 0.55 * s_g + 0.45 * s_a


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.5
    u = a | b
    if not u:
        return 0.0
    return len(a & b) / len(u)


# Landmark-based face height/width (face_h / jaw_w). Tune with your distribution.
ELONGATION_HW_LO = 0.80
ELONGATION_HW_HI = 1.58
# Wider (lower) H/W: stronger match to these frame tags. Taller (higher) H/W: the other set.
# Overlaps optician "rules" but is a continuous nudge *within* the same coarsely-binned face shape.
TAGS_FIT_WIDE_FACE = {
    "angular",
    "rectangular",
    "geometric",
    "narrow",
    "metal",
    "bold",
    "classic",
}
TAGS_FIT_TALL_FACE = {
    "round",
    "soft",
    "wayfarer",
    "keyhole",
    "oval",
    "aviator",
    "cat_eye",
    "acetate",
}


def score_elongation_frame_fit(
    product: dict[str, Any],
    height_over_width: float,
) -> float:
    """
    0..1: how well product `frame_tags` align with an **elongation** level between
    wide (low H/W) and tall (high H/W). Complements the discrete face_shape bucket
    and tag_rule matching.
    """
    tags = set(product.get("frame_tags") or [])
    if not tags:
        return 0.5
    span = ELONGATION_HW_HI - ELONGATION_HW_LO
    if span <= 0:
        t = 0.5
    else:
        t = (float(height_over_width) - ELONGATION_HW_LO) / span
    t = max(0.0, min(1.0, t))  # 0 ≈ low H/W (wider in proportion), 1 ≈ high H/W (taller)
    w_pref = TAGS_FIT_WIDE_FACE
    h_pref = TAGS_FIT_TALL_FACE
    w_hit = len(tags & w_pref) / max(len(w_pref), 1)
    h_hit = len(tags & h_pref) / max(len(h_pref), 1)
    return (1.0 - t) * w_hit + t * h_hit


def score_purchase_affinity(
    product: dict[str, Any],
    last_products: list[dict[str, Any]],
) -> float:
    """
    0..1: style continuity from last purchase — tag overlap; plus bonus if same
    line / collection id in your DB later.
    """
    if not last_products:
        return 0.5
    ptags = set(product.get("frame_tags") or [])
    best = 0.0
    for lp in last_products:
        ltags = set(lp.get("frame_tags") or [])
        pid = str(product.get("id", ""))
        lp_id = str(lp.get("id", ""))
        jac = _jaccard(ptags, ltags)
        same = 1.0 if pid and pid == lp_id else 0.0
        s = 0.85 * jac + 0.15 * same
        best = max(best, s)
    return best


# Synthetic / catalog tags: one of these should appear on S3 flat rows (or match via keywords)
_COLOR_FAMILY_SYNTH: frozenset[str] = frozenset({"color_vibrant", "color_neutral", "color_plain"})

# When tags/text lack synth tags, use keyword hints in frame style names
_VIBRANT_KEYWORDS: frozenset[str] = frozenset(
    {
        "vibrant",
        "bold",
        "bright",
        "red",
        "blue",
        "green",
        "yellow",
        "pink",
        "pop",
        "playful",
        "sporty",
        "multi",
        "gradient",
        "neon",
        "fun",
    },
)
_PLAIN_KEYWORDS: frozenset[str] = frozenset(
    {
        "plain",
        "subdued",
        "muted",
        "matte",
        "black",
        "brown",
        "tortoise",
        "beige",
        "horn",
        "understated",
        "conservative",
        "classic_tortoise",
    },
)


def _color_family_from_text(tags: set[str], name: str, pid: str) -> str | None:
    """
    If explicit synthetic family tag is present, return vibrant|neutral|plain.
    Else infer from words in tags/name/id (heuristic).
    """
    inter = tags & _COLOR_FAMILY_SYNTH
    if "color_vibrant" in inter:
        return "vibrant"
    if "color_neutral" in inter:
        return "neutral"
    if "color_plain" in inter:
        return "plain"
    blob = " ".join(tags) + " " + name.lower() + " " + str(pid).lower()
    pwords = {w for w in re.split(r"[^\w]+", blob) if len(w) > 1}
    if pwords & _VIBRANT_KEYWORDS and not (pwords & _PLAIN_KEYWORDS):
        return "vibrant"
    if pwords & _PLAIN_KEYWORDS and not (pwords & _VIBRANT_KEYWORDS):
        return "plain"
    if pwords & _VIBRANT_KEYWORDS and (pwords & _PLAIN_KEYWORDS):
        return "neutral"
    return None


def score_color_lifecycle_match(product: dict[str, Any], demo: dict[str, Any]) -> float:
    """
    0..1: for Kids/Teenager prefer vibrant; Adult neutral; Old plain.
    No lifecycle → 0.5
    """
    tags = {str(t).lower() for t in (product.get("frame_tags") or []) if t is not None}
    name = str(product.get("name") or "")
    pid = str(product.get("id") or "")
    family = _color_family_from_text(tags, name, pid)
    lc = demo.get("age_lifecycle")

    if lc in ("Kids", "Teenager"):
        if family == "vibrant":
            return 1.0
        if family == "neutral":
            return 0.55
        if family == "plain":
            return 0.22
        return 0.5
    if lc == "Adult":
        if family == "neutral":
            return 0.9
        if family == "vibrant":
            return 0.6
        if family == "plain":
            return 0.5
        return 0.5
    if lc == "Old":
        if family == "plain":
            return 1.0
        if family == "neutral":
            return 0.62
        if family == "vibrant":
            return 0.25
        return 0.5
    return 0.5


def parse_search_tokens(*queries: str | None) -> list[str]:
    """
    Non-empty, de-duplicated tokens (length > 1) from style/keyword prompt strings.
    """
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        if not q or not str(q).strip():
            continue
        for t in re.split(r"[^\w]+", str(q).strip()):
            tl = t.lower()
            if len(tl) < 2 or tl in seen:
                continue
            seen.add(tl)
            out.append(tl)
    return out


def _product_search_blob(product: dict[str, Any]) -> tuple[str, list[str]]:
    """Lowercase name, id, and per-tag strings for matching."""
    tags = [str(x).lower() for x in (product.get("frame_tags") or []) if x is not None]
    name = str(product.get("name") or "").lower()
    pid = str(product.get("id") or "").lower()
    blob = f"{name} {pid} {' '.join(tags)}"
    return blob, tags


def product_matches_any_search_token(product: dict[str, Any], tokens: list[str]) -> bool:
    if not tokens:
        return True
    blob, tags = _product_search_blob(product)
    name = str(product.get("name") or "").lower()
    pid = str(product.get("id") or "").lower()
    for t in tokens:
        if t in blob or t in name or t in pid or any(t in tag for tag in tags):
            return True
    return False


def score_text_query_match(
    product: dict[str, Any],
    style_prompt: str | None,
    keyword_query: str | None = None,
) -> float:
    """
    0..1: recall of query tokens (from **style** + **keyword** prompts) against
    product name, id, and frame_tags. Neutral 0.5 when no tokens.
    """
    tokens = parse_search_tokens(style_prompt, keyword_query)
    if not tokens:
        return 0.5
    blob, _tags = _product_search_blob(product)
    hits = sum(1 for t in tokens if t in blob)
    return min(1.0, hits / len(tokens))


def score_style_prompt_match(product: dict[str, Any], prompt: str | None) -> float:
    """Back-compat: same as `score_text_query_match(product, prompt, None)`."""
    return score_text_query_match(product, prompt, None)


def score_region_affinity(
    product: dict[str, Any],
    region: str | None,
    regional: dict[str, Any] | None,
) -> float:
    """
    0..1: regional product affinity + mean tag multipliers, normalized to [0,1].
    """
    if not region or not regional:
        return 0.5
    reg = regional.get(region) or regional.get("default")
    if not reg or not isinstance(reg, dict):
        return 0.5
    pid = str(product.get("id", ""))
    pa: dict[str, float] = dict(reg.get("product_affinity") or {})
    ta: dict[str, float] = dict(reg.get("tag_affinity") or {})
    s_prod = float(pa.get(pid, 0.5))
    tags = list(product.get("frame_tags") or [])
    if tags and ta:
        mults = [float(ta.get(t, 1.0)) for t in tags]
        # map multiplier ~0.7..1.3 into 0..1
        m = sum(mults) / len(mults)
        s_tag = max(0.0, min(1.0, (m - 0.7) / 0.6))
    else:
        s_tag = 0.5
    s_prod = max(0.0, min(1.0, s_prod))
    return 0.6 * s_prod + 0.4 * s_tag


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na * nb < 1e-12:
        return 0.0
    return dot / (na * nb)


def score_face_hybrid(
    product: dict[str, Any],
    rules: dict[str, Any],
    query_embedding: list[float],
    *,
    w_tag: float = 0.55,
    w_emb: float = 0.45,
    height_over_width: float | None = None,
) -> float:
    """Alias: combined face score used by w_face (tags + **embedding cosine** vs query vector)."""
    return float(
        face_hybrid_with_similarity(
            product,
            rules,
            query_embedding,
            w_tag=w_tag,
            w_emb=w_emb,
            height_over_width=height_over_width,
        )["face_combined"],
    )


def face_hybrid_with_similarity(
    product: dict[str, Any],
    rules: dict[str, Any],
    query_embedding: list[float],
    *,
    w_tag: float = 0.55,
    w_emb: float = 0.45,
    w_elong: float = 0.28,
    height_over_width: float | None = None,
) -> dict[str, Any]:
    """
    Face hybrid score with explicit **embedding_similarity** (cosine vs query vector).
    When `height_over_width` is set, a third term scores tag fit for wide vs tall face
    (continuous function of H/W; see ``score_elongation_frame_fit``).
    The scalar `face_combined` is what gets weighted by w_face in the final rank.
    """
    pref = set(rules.get("preferred_frame_tags") or [])
    tags = set(product.get("frame_tags") or [])
    tag_hits = len(tags & pref) / max(len(pref), 1)
    emb = product.get("embedding")
    embedding_similarity = _cosine(emb, query_embedding) if isinstance(emb, list) and emb else 0.0
    if height_over_width is None:
        t = w_tag + w_emb
        if t <= 0:
            t = 1.0
        face_combined = (w_tag * tag_hits + w_emb * embedding_similarity) / t
        return {
            "face_combined": face_combined,
            "tag_match_ratio": tag_hits,
            "embedding_similarity": embedding_similarity,
            "elongation_fit": None,
        }
    s_elong = score_elongation_frame_fit(product, height_over_width)
    wt, we, wel = 0.40, 0.32, w_elong
    denom = wt + we + wel
    if denom <= 0:
        denom = 1.0
    face_combined = (wt * tag_hits + we * embedding_similarity + wel * s_elong) / denom
    return {
        "face_combined": face_combined,
        "tag_match_ratio": tag_hits,
        "embedding_similarity": embedding_similarity,
        "elongation_fit": round(s_elong, 4),
    }


def personalized_rank(
    candidates: list[dict[str, Any]],
    rules: dict[str, Any],
    query_embedding: list[float],
    demo: dict[str, Any],
    last_products: list[dict[str, Any]],
    region: str | None,
    regional: dict[str, Any] | None,
    weights: RankingWeights | None = None,
    height_over_width: float | None = None,
    style_prompt: str | None = None,
    keyword_query: str | None = None,
) -> list[dict[str, Any]]:
    """
    Full blend: face rules + global popularity + demographics + last order + region.
    Each item includes `score`, `score_breakdown`, and original product fields.

    When `height_over_width` is set (face height / jaw width from landmarks), the face
    sub-score also blends **elongation_fit** (tag alignment for wide vs tall proportion).
    ``style_prompt`` and ``keyword_query`` are combined for the text/keyword score.
    """
    w = (weights or RankingWeights()).normalized()
    ranked: list[tuple[float, dict[str, Any]]] = []
    for p in candidates:
        fh = face_hybrid_with_similarity(
            p,
            rules,
            query_embedding,
            height_over_width=height_over_width,
        )
        s_face = float(fh["face_combined"])
        pop = float(p.get("popularity", 0.5))
        s_demo = score_demographic_match(p, demo)
        s_pur = score_purchase_affinity(p, last_products)
        s_reg = score_region_affinity(p, region, regional)
        s_text = score_text_query_match(p, style_prompt, keyword_query)
        s_color = score_color_lifecycle_match(p, demo)
        total = (
            w["w_face"] * s_face
            + w["w_pop"] * pop
            + w["w_demo"] * s_demo
            + w["w_purchase"] * s_pur
            + w["w_region"] * s_reg
            + w["w_prompt"] * s_text
            + w["w_color"] * s_color
        )
        breakdown: dict[str, Any] = {
            "face_combined": round(s_face, 4),
            "tag_match_ratio": round(fh["tag_match_ratio"], 4),
            "embedding_similarity": round(fh["embedding_similarity"], 4),
            "popularity": round(pop, 4),
            "demographics": round(s_demo, 4),
            "purchase_continuity": round(s_pur, 4),
            "region_cohort": round(s_reg, 4),
            "text_query_match": round(s_text, 4),
            "color_lifecycle": round(s_color, 4),
            "weights": w,
        }
        if fh.get("elongation_fit") is not None:
            breakdown["elongation_fit"] = fh["elongation_fit"]
        row = {
            **p,
            "score": round(total, 5),
            "score_breakdown": breakdown,
        }
        ranked.append((total, row))
    ranked.sort(key=lambda x: -x[0])
    return [r for _, r in ranked]


@dataclass
class UserContext:
    """Optional account / order context. Override demographic fields when you trust them over the photo."""

    user_id: str | None = None
    region: str | None = None
    last_order_product_ids: list[str] = field(default_factory=list)
    gender_override: str | None = None
    age_override: int | None = None
    age_lifecycle_override: str | None = None  # Kids | Teenager | Adult | Old

    @staticmethod
    def from_dict(d: dict[str, Any] | None) -> UserContext:
        if not d:
            return UserContext()
        return UserContext(
            user_id=d.get("user_id"),
            region=d.get("region"),
            last_order_product_ids=list(d.get("last_order_product_ids") or []),
            gender_override=d.get("gender_override") or d.get("gender"),
            age_override=d.get("age_override") if d.get("age_override") is not None else d.get("age"),
            age_lifecycle_override=d.get("age_lifecycle_override"),
        )

    def merge_demographics(self, from_face: dict[str, Any]) -> dict[str, Any]:
        out = dict(from_face)
        if self.gender_override is not None:
            out["gender_label"] = str(self.gender_override)
        if self.age_override is not None:
            out["age_mid"] = float(self.age_override)
            out["age_band"] = _age_to_band(float(self.age_override))
        if self.age_lifecycle_override is not None:
            out["age_lifecycle"] = str(self.age_lifecycle_override)
        elif out.get("age_mid") is not None:
            out["age_lifecycle"] = age_lifecycle_bracket(float(out["age_mid"]))
        return out
