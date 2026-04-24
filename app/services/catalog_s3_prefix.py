"""Build effective S3 prefix under a shared parent (e.g. ``Glass/Sunglass/all_images`` vs ``Glass/Normal_glass``)."""

from __future__ import annotations

from app.config import Settings


def _extra_segment_for_category(s: Settings, cat: str) -> str:
    """Optional subfolder after ``…/Sunglass/`` or ``…/Normal_glass/`` (asymmetric layouts)."""
    if cat in ("sunglass", "sunglasses"):
        return (getattr(s, "s3_glass_sunglass_extra_prefix", None) or "").strip().strip("/")
    if cat in ("eyeglass", "eyeglasses", "normal", "normal_glass"):
        return (getattr(s, "s3_glass_eyeglass_extra_prefix", None) or "").strip().strip("/")
    return ""


def effective_catalog_s3_prefix(s: Settings, glass_category: str | None) -> str:
    """
    When ``glass_category`` is set, list/manifest under::

        {S3_CATALOG_PREFIX}/{S3_GLASS_PARENT}/{type folder}/[{S3_GLASS_*_EXTRA_PREFIX}/]

    Example: ``Glass/Sunglass/all_images/`` vs ``Glass/Normal_glass/`` (no extra).

    When unset or \"default\", use only ``S3_CATALOG_PREFIX`` (legacy: whole prefix tree).
    Set ``S3_USE_GLASS_SUBFOLDERS=false`` to always use the base prefix (ignore category).
    """
    base = (s.s3_catalog_prefix or "").strip().strip("/")
    if not getattr(s, "s3_use_glass_subfolders", True):
        return base
    if not glass_category or str(glass_category).strip().lower() in ("", "default", "all"):
        return base
    cat = str(glass_category).strip().lower()
    gp = (s.s3_glass_parent or "Glass").strip().strip("/")
    if not gp:
        return base
    if cat in ("sunglass", "sunglasses"):
        sub = (s.s3_glass_sunglass_folder or "Sunglass").strip().strip("/")
    elif cat in ("eyeglass", "eyeglasses", "normal", "normal_glass"):
        sub = (s.s3_glass_eyeglass_folder or "Normal_glass").strip().strip("/")
    else:
        return base
    if not sub:
        return base
    if not base:
        head = f"{gp}/{sub}"
    else:
        head = f"{base}/{gp}/{sub}"
    extra = _extra_segment_for_category(s, cat)
    if extra:
        return f"{head}/{extra}"
    return head


def catalog_listing_fingerprint(s: Settings) -> str:
    """
    String that changes when any S3 list prefix input changes (for UI cache keys, e.g. Streamlit).
    """
    parts = [
        (s.s3_catalog_prefix or "").strip(),
        str(getattr(s, "s3_use_glass_subfolders", True)),
        (s.s3_glass_parent or "").strip(),
        (s.s3_glass_sunglass_folder or "").strip(),
        (s.s3_glass_eyeglass_folder or "").strip(),
        (getattr(s, "s3_glass_sunglass_extra_prefix", None) or "").strip(),
        (getattr(s, "s3_glass_eyeglass_extra_prefix", None) or "").strip(),
    ]
    return "|".join(parts)
