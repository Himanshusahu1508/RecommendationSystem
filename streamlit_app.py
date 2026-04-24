"""
Eyewear recommendation: face photo + AWS Rekognition + S3 catalog + optional CLIP style channel.

  .venv/bin/streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from app.config import refresh_settings_cache

refresh_settings_cache()

_PAGE_CSS = """
<style>
    .main .block-container { padding-top: 1.25rem; padding-bottom: 3rem; max-width: 1200px; }
    h1.title-main { font-weight: 650; letter-spacing: -0.03em; margin-bottom: 0.15rem; }
    p.subtitle { color: rgba(49, 51, 63, 0.65); font-size: 1.05rem; margin-top: 0; }
    div[data-testid="stSidebarContent"] { background: linear-gradient(180deg, #fafbfc 0%, #f4f6f8 100%); }
    .result-card {
        border: 1px solid rgba(49, 51, 63, 0.1);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.75rem;
        background: #fff;
    }
</style>
"""


@st.cache_data(ttl=120, show_spinner="Loading eyewear catalog from S3…")
def _s3_product_rows(glass_key: str, _settings_cache_key: str) -> list[dict]:
    from app.config import get_settings
    from app.services.s3_flat_catalog import build_lusmt_flat_catalog

    s = get_settings()
    if not s.s3_catalog_bucket:
        return []
    gc = glass_key.strip() if glass_key.strip() else None
    return build_lusmt_flat_catalog(s, glass_category=gc)


def _regional() -> dict | None:
    p = ROOT / "regional_affinity.json"
    if not p.is_file():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _fmt_ratio(v: object) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return "—"


def main() -> None:
    st.set_page_config(
        page_title="RS2 — Eyewear Studio",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_PAGE_CSS, unsafe_allow_html=True)

    from app.config import get_settings
    from app.services.catalog_s3_prefix import (
        catalog_listing_fingerprint,
        effective_catalog_s3_prefix,
    )

    s0 = get_settings()

    with st.sidebar:
        st.markdown("### Session")
        if not os.environ.get("AWS_ACCESS_KEY_ID") and not os.environ.get("AWS_SESSION_TOKEN"):
            st.error("Add AWS credentials to `.env` for Rekognition and S3.")
        else:
            st.success("AWS credentials loaded")
        st.caption(
            f"Hybrid blend · preference **{s0.hybrid_w_preference:.0%}** · face rules **{s0.hybrid_w_face:.0%}** "
            f"(set in `.env`) · CLIP **{'on' if s0.clip_preference_enabled else 'off'}**"
        )
        glass_key = ""
        if s0.s3_catalog_bucket and getattr(s0, "s3_use_glass_subfolders", True):
            pick = st.radio(
                "Catalog",
                ("Sunglasses", "Normal glasses"),
                horizontal=False,
                label_visibility="visible",
            )
            glass_key = "sunglass" if pick == "Sunglasses" else "eyeglass"
            eff = effective_catalog_s3_prefix(s0, glass_key)
            st.caption(f"Prefix: `{eff or '(root)'}`")
        else:
            st.caption("Using full `S3_CATALOG_PREFIX` (subfolders off or no bucket)")

        top_n = st.slider("How many frames", 1, 15, 5)
        no_q = st.toggle("Skip face-quality gate", value=False, help="Allow weaker poses / lighting (testing only)")

    st.markdown('<h1 class="title-main">Eyewear Studio</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Face fit from your photo · optional CLIP style from words or a reference image · catalog from S3</p>',
        unsafe_allow_html=True,
    )

    if not s0.s3_catalog_bucket:
        st.warning("Set `S3_CATALOG_BUCKET` in `.env` to load products.")
        return

    products = _s3_product_rows(glass_key, catalog_listing_fingerprint(s0))
    if not products:
        from app.services.s3_flat_catalog import diagnose_flat_catalog

        gc = glass_key.strip() if glass_key.strip() else None
        st.error("No catalog rows from S3.")
        st.json(diagnose_flat_catalog(s0, glass_category=gc), expanded=True)
        return

    st.caption(f"**{len(products)}** frames in view · blend & CLIP flags come from environment (not the UI)")

    col_face, col_style = st.columns((1, 1), gap="large")
    with col_face:
        st.markdown("##### Your face")
        u1, u2 = st.columns(2)
        with u1:
            up = st.file_uploader("Upload", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
        with u2:
            cam = st.camera_input("Camera", label_visibility="collapsed")
        image_bytes: bytes | None = up.getvalue() if up is not None else None
        if image_bytes is None and cam is not None:
            image_bytes = cam.getvalue()

    with col_style:
        st.markdown("##### Style (CLIP)")
        st.caption("Same visual encoder as your catalog `clip_embedding` build — no keyword search over product text.")
        style_text = st.text_area(
            "Describe the look",
            height=110,
            placeholder="e.g. thin gold wire, bold acetate, sporty wrap …",
            label_visibility="visible",
            help="Encoded with CLIP. Leave empty if you only use an inspiration image.",
        )
        style_ref = st.file_uploader(
            "Inspiration image (optional)",
            type=["png", "jpg", "jpeg", "webp"],
            help="A reference look (not your face). Combined with text when both are set.",
        )

    with st.expander("Demographics & context", expanded=False):
        c3, c4, c5 = st.columns(3)
        with c3:
            gender = st.radio("Gender", ("From photo", "Male", "Female"), horizontal=True)
        with c4:
            age_mode = st.radio("Age", ("From photo", "Override years"), horizontal=True)
            age_years: int | None = None
            if age_mode == "Override years":
                age_years = st.number_input("Years", 0, 120, 30, label_visibility="collapsed")
        with c5:
            life = st.selectbox(
                "Color palette bracket",
                ("Auto", "Kids", "Teenager", "Adult", "Old"),
            )
        region = st.text_input("Region key", value="default")
        last_orders = st.text_input("Last order product ids (comma-separated)", placeholder="optional")

    go = st.button("Get recommendations", type="primary", use_container_width=True)

    if not go:
        st.info("Add your face photo, optionally add style text or a reference image, then run **Get recommendations**.")
        return
    if not image_bytes:
        st.warning("Upload a face image or use the camera.")
        return

    from app.embedding_config import load_face_shape_prototypes
    from app.services.s3_image import enrich_recommendations_with_presign
    from glasses_recommend import _image_size_from_bytes, recommend_from_bytes
    from ranking_signals import RankingWeights

    try:
        prototypes = load_face_shape_prototypes(s0.embedding_dim)
    except ValueError as e:
        st.error(str(e))
        return

    user_context: dict = {"region": region or None, "last_order_product_ids": []}
    if last_orders.strip():
        user_context["last_order_product_ids"] = [x.strip() for x in last_orders.split(",") if x.strip()]
    if gender == "Male":
        user_context["gender_override"] = "Male"
    elif gender == "Female":
        user_context["gender_override"] = "Female"
    if age_mode == "Override years" and age_years is not None:
        user_context["age_override"] = int(age_years)
    if life != "Auto":
        user_context["age_lifecycle_override"] = life

    w, h = _image_size_from_bytes(image_bytes)
    ref_bytes = style_ref.getvalue() if style_ref is not None else None
    st_prompt = (style_text or "").strip() or None

    out = recommend_from_bytes(
        image_bytes,
        w,
        h,
        products,
        regional=_regional(),
        check_quality=not no_q,
        top_n=top_n,
        user_context=user_context,
        face_shape_prototypes=prototypes,
        ranking_weights=RankingWeights(),
        style_prompt=st_prompt,
        style_reference_image_bytes=ref_bytes,
        use_preference_hybrid=s0.use_preference_hybrid,
        hybrid_w_preference=s0.hybrid_w_preference,
        hybrid_w_face=s0.hybrid_w_face,
        use_clip_preference=s0.clip_preference_enabled,
    )
    out = enrich_recommendations_with_presign(s0, out)

    if not out.get("ok"):
        st.json(out)
        return

    demo = out.get("demographics") or {}
    fg = out.get("face_geometry") or {}
    lc = demo.get("age_lifecycle")
    binfo = out.get("face_shape_buckets") or {}

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Face shape", str(out.get("face_shape", "—")))
    with m2:
        st.metric("Gender", str(demo.get("gender_label", "—")))
    with m3:
        st.metric("Age (est.)", f"{demo.get('age_low', '?')}-{demo.get('age_high', '?')}")
    with m4:
        st.metric("Lifecycle", str(lc or "—"))

    jratio = fg.get("jaw_width_to_face_height")
    st.caption(
        f"Jaw / face height ratio: **{_fmt_ratio(jratio)}** · bucket cutoffs (fixed in code): "
        f"`{binfo.get('jaw_width_to_height_cutoffs', [])}`"
    )

    hy = out.get("hybrid")
    if isinstance(hy, dict):
        if hy.get("enabled"):
            st.caption(
                f"**CLIP style blend** · `{hy.get('preference_mode', '?')}` "
                f"· read `hybrid` in the JSON for weights."
            )
        else:
            st.caption(f"Style blend: *off* — `{hy.get('reason', '')}`")

    if st_prompt or ref_bytes:
        st.caption("CLIP query: " + ("text + image" if (st_prompt and ref_bytes) else ("text" if st_prompt else "image")))

    if out.get("ok") and not (out.get("recommendations") or []):
        st.warning(out.get("note") or "No recommendations in this run.")

    for i, r in enumerate(out.get("recommendations") or [], 1):
        sb = r.get("score_breakdown") or {}
        cscore = sb.get("color_lifecycle")
        cfam = r.get("color_family")
        ctag = None
        for t in r.get("frame_tags") or []:
            if str(t).startswith("color_"):
                ctag = str(t)
                break
        st.markdown(
            f'<div class="result-card"><h3 style="margin:0 0 0.35rem 0;">{i}. {r.get("name", r.get("id"))}</h3>'
            f'<p style="margin:0;color:#444;">Score <code>{r.get("score")}</code></p></div>',
            unsafe_allow_html=True,
        )
        c_a, c_b, c_c = st.columns(3)
        with c_b:
            if cscore is not None:
                st.metric("Color fit", f"{float(cscore):.2f}")
        with c_c:
            fam = cfam or (ctag.replace("color_", "") if ctag else "—")
            st.metric("Palette", str(fam))
        sb_show = {k: v for k, v in sb.items() if k != "weights"}
        with st.expander("Score detail", expanded=False):
            st.json(sb_show)
        for url in r.get("eyewear_image_urls") or []:
            st.image(url, width=420)


if __name__ == "__main__":
    main()
