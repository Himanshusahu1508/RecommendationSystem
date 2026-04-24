"""
Eyewear recommendation UI: face photo + rule-based face shape (6 buckets) + S3 catalog.

Run from project root:
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


@st.cache_data(ttl=120, show_spinner="Loading eyewear catalog from S3…")
def _s3_product_rows() -> list[dict]:
    from app.config import get_settings
    from app.services.s3_flat_catalog import build_lusmt_flat_catalog

    s = get_settings()
    if not s.s3_catalog_bucket:
        return []
    return build_lusmt_flat_catalog(s)


def _regional() -> dict | None:
    p = ROOT / "regional_affinity.json"
    if not p.is_file():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    st.set_page_config(page_title="Eyewear recommend", layout="wide")
    st.title("Eyewear recommendation")
    st.caption(
        "Rule-based face shape (jaw width ÷ height, six buckets), gender/age from AWS or overrides, "
        "age-based **color palette** (vibrant for Kids/Teen, plain for Old), style prompt, regional cohort — "
        "S3 flat catalog.",
    )

    if not os.environ.get("AWS_ACCESS_KEY_ID") and not os.environ.get("AWS_SESSION_TOKEN"):
        st.error("Set AWS credentials in `.env` (or use a profile) for Rekognition and S3.")
        return

    products = _s3_product_rows()
    if not products:
        st.error(
            "No `lusmt*_*_*.jpg` products found. Set `S3_CATALOG_BUCKET` and `S3_CATALOG_PREFIX` in `.env`.",
        )
        return

    col_a, col_b = st.columns(2)
    with col_a:
        up = st.file_uploader("Upload face photo (JPEG/PNG)", type=["jpg", "jpeg", "png", "webp"])
    with col_b:
        cam = st.camera_input("Or capture from camera")

    image_bytes: bytes | None = None
    if up is not None:
        image_bytes = up.getvalue()
    elif cam is not None:
        image_bytes = cam.getvalue()

    st.subheader("Segmentation (optional overrides)")
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.radio(
            "Gender",
            ("From photo (Rekognition)", "Male", "Female"),
            horizontal=True,
        )
    with c2:
        age_mode = st.radio("Age", ("From photo", "Override age (years)"), horizontal=True)
        age_years: int | None = None
        if age_mode == "Override age (years)":
            age_years = st.number_input("Years", min_value=0, max_value=120, value=30)
    with c3:
        life = st.selectbox(
            "Age bracket (demographics + color palette rank)",
            (
                "Auto (from photo or age above)",
                "Kids",
                "Teenager",
                "Adult",
                "Old",
            ),
            help="Also drives color scoring: Kids/Teen → prefer vibrant; Old → prefer plain; Adult → neutral.",
        )

    with st.expander("Color ranking by age (how frames are boosted)", expanded=False):
        st.markdown(
            "| Age group | Palette preference |\n"
            "|-----------|--------------------|\n"
            "| **Kids**, **Teenager** | Vibrant / bold (`color_vibrant`, bright keywords) |\n"
            "| **Adult** | Neutral (`color_neutral`) |\n"
            "| **Old** | Plain / muted (`color_plain`, understated keywords) |\n"
        )
        st.caption(
            "S3 catalog rows carry synthetic `color_vibrant` / `color_neutral` / `color_plain` tags. "
            "Score appears as **color_lifecycle** in each result (0–1)."
        )

    region = st.text_input("Region (regional_affinity.json key)", value="default")
    last_orders = st.text_input(
        "Last order product ids (comma-separated, optional)",
        placeholder="e.g. lusmt00438, lusmt00102",
    )
    style_prompt = st.text_area(
        "Special style (keywords; boosts frames whose name/tags match)",
        placeholder="e.g. aviator metal bold",
        height=80,
    )

    from glasses_recommend import FACE_JAW_WH_BOUNDS, FACE_SHAPE_ORDER, validate_jaw_wh_bounds

    if "jaw_cfg_init" not in st.session_state:
        for i, v in enumerate(FACE_JAW_WH_BOUNDS):
            st.session_state[f"jb{i}"] = float(v)
        st.session_state["jaw_cfg_init"] = True

    with st.expander("Face-shape thresholds (jaw width ÷ face height ratio)", expanded=False):
        st.markdown(
            "Six buckets in **increasing** order: **"
            + " → **".join(FACE_SHAPE_ORDER)
            + "**. Five cutoffs `b0…b4` must satisfy `b0 < b1 < … < b4`."
        )
        _custom = st.toggle(
            "Use custom cutoffs (otherwise use code defaults from `glasses_recommend.FACE_JAW_WH_BOUNDS`)",
            value=False,
            key="custom_jaw_bounds",
        )
        st.caption(
            "Regions: (rectangle &lt; b0) | [b0,b1) square | [b1,b2) round | "
            "[b2,b3) oval | [b3,b4) long | heart ≥ b4"
        )
        if _custom:
            r1, r2, r3, r4, r5 = st.columns(5)
            for i, col in enumerate((r1, r2, r3, r4, r5)):
                with col:
                    st.number_input(
                        f"b{i}",
                        min_value=0.2,
                        max_value=1.8,
                        step=0.01,
                        format="%.2f",
                        key=f"jb{i}",
                        help="All five must be strictly increasing. Lower b = stricter 'narrow' left buckets.",
                    )
            st.caption(
                f"**Code defaults:** {list(FACE_JAW_WH_BOUNDS)}"
            )
            if st.button("Reset cutoffs to code defaults", key="reset_jaw"):
                for i, v in enumerate(FACE_JAW_WH_BOUNDS):
                    st.session_state[f"jb{i}"] = float(v)
                st.rerun()

    st.divider()
    c4, c5, c6 = st.columns(3)
    with c4:
        top_n = st.slider("Top N", 1, 15, 5)
    with c5:
        no_q = st.checkbox("Skip quality gate (pose/eyes EAR — faster, less strict)", value=False)
    with c6:
        go = st.button("Recommend", type="primary")

    if not go:
        st.info("Upload or capture a face, set options, then **Recommend**.")
        return
    if not image_bytes:
        st.warning("Add an image or camera capture first.")
        return

    from app.config import get_settings
    from app.embedding_config import load_face_shape_prototypes
    from app.services.s3_image import enrich_recommendations_with_presign
    from glasses_recommend import _image_size_from_bytes, recommend_from_bytes
    from ranking_signals import RankingWeights

    jaw_arg: tuple[float, ...] | None = None
    if st.session_state.get("custom_jaw_bounds", False):
        b = tuple(float(st.session_state[f"jb{i}"]) for i in range(5))
        verr = validate_jaw_wh_bounds(b)
        if verr:
            st.error(verr)
            return
        jaw_arg = b

    s = get_settings()
    try:
        prototypes = load_face_shape_prototypes(s.embedding_dim)
    except ValueError as e:
        st.error(str(e))
        return

    user_context: dict = {"region": region or None, "last_order_product_ids": []}
    if last_orders.strip():
        user_context["last_order_product_ids"] = [
            x.strip() for x in last_orders.split(",") if x.strip()
        ]
    if gender == "Male":
        user_context["gender_override"] = "Male"
    elif gender == "Female":
        user_context["gender_override"] = "Female"
    if age_mode == "Override age (years)" and age_years is not None:
        user_context["age_override"] = int(age_years)
    if life != "Auto (from photo or age above)":
        user_context["age_lifecycle_override"] = life

    w, h = _image_size_from_bytes(image_bytes)
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
        style_prompt=style_prompt.strip() or None,
        jaw_wh_bounds=jaw_arg,
    )
    out = enrich_recommendations_with_presign(s, out)

    if not out.get("ok"):
        st.json(out)
        return

    demo = out.get("demographics") or {}
    fg = out.get("face_geometry") or {}
    lc = demo.get("age_lifecycle")
    if lc in ("Kids", "Teenager"):
        _color_hint = "Prefer **vibrant** frames for this age group."
    elif lc == "Adult":
        _color_hint = "Prefer **neutral** palettes."
    elif lc == "Old":
        _color_hint = "Prefer **plain / muted** frames for this age group."
    else:
        _color_hint = "Set **Age bracket** (or use a clear face photo) for stronger color bias."
    st.success(
        f"**Face shape:** {out.get('face_shape', '?')}  ·  **Gender (used):** {demo.get('gender_label', '?')}  ·  "
        f"**Age lifecycle:** {lc or '?'} "
        f"(est. {demo.get('age_low', '?')}-{demo.get('age_high', '?')} y)",
    )
    st.info(f"**Color policy:** {_color_hint}")

    wn = (out.get("ranking_weights") or {}) if isinstance(out.get("ranking_weights"), dict) else {}
    wcol = wn.get("w_color")
    if wcol is not None:
        st.caption(
            f"Color signal weight in blend: `w_color` ≈ **{float(wcol):.3f}** (see `color_lifecycle` per frame below).",
        )

    binfo = out.get("face_shape_buckets") or {}
    st.caption(
        f"jaw_w / face_h = {fg.get('jaw_width_to_face_height', '?')!s}  ·  "
        f"cutoffs: {binfo.get('jaw_width_to_height_cutoffs')} "
        f"({binfo.get('jaw_width_to_height_cutoffs_source', 'default')})  ·  "
        f"order: {binfo.get('order')}",
    )

    for i, r in enumerate(out.get("recommendations") or [], 1):
        sb = r.get("score_breakdown") or {}
        cscore = sb.get("color_lifecycle")
        cfam = r.get("color_family")
        ctag = None
        for t in r.get("frame_tags") or []:
            s = str(t)
            if s.startswith("color_"):
                ctag = s
                break
        m1, m2, m3 = st.columns([2.2, 1, 1])
        with m1:
            st.markdown(f"### {i}. {r.get('name', r.get('id'))} — score `{r.get('score')}`")
        with m2:
            if cscore is not None:
                st.metric("Color fit (0–1)", f"{float(cscore):.2f}")
        with m3:
            fam = cfam or (ctag.replace("color_", "") if ctag else "—")
            st.metric("Frame palette tag", str(fam))
        sb_show = dict(sb)
        if "weights" in sb_show:
            del sb_show["weights"]
        st.json(sb_show, expanded=False)
        for url in r.get("eyewear_image_urls") or []:
            st.image(url, width=400)


if __name__ == "__main__":
    main()
