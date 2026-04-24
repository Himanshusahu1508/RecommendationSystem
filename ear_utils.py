"""
Eye Aspect Ratio (EAR)-style metric from AWS Rekognition DetectFaces landmarks.

Rekognition provides left/right eye corners plus top/bottom (see AWS Landmark.Type).
Classic Soukupová–Čech EAR uses six points per eye; here we use the four points as
an eye-opening ratio: vertical_spread / horizontal_spread (per eye), then average.

Coordinates are normalized (0–1); distances are computed in pixel space when
image width/height are provided so the ratio matches physical eye shape.
"""

from __future__ import annotations

import math
from typing import Any


def landmarks_by_type(face_detail: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Map Landmark.Type -> (x, y) normalized 0..1."""
    out: dict[str, tuple[float, float]] = {}
    for lm in face_detail.get("Landmarks") or []:
        t = lm.get("Type")
        if not t:
            continue
        out[t] = (float(lm["X"]), float(lm["Y"]))
    return out


def _dist_px(
    a: tuple[float, float],
    b: tuple[float, float],
    img_w: float,
    img_h: float,
) -> float:
    ax, ay = a[0] * img_w, a[1] * img_h
    bx, by = b[0] * img_w, b[1] * img_h
    return math.hypot(ax - bx, ay - by)


def ear_ratio_from_face_detail(
    face_detail: dict[str, Any],
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    """
    Compute per-eye opening ratio and mean EAR-style score.

    For each eye: vertical = dist(Up, Down), horizontal = dist(Left, Right),
    ratio = vertical / horizontal. Averages left and right when both present.

    Returns keys: ear_left, ear_right, ear_mean, ok (bool), missing (list of str).
    """
    lm = landmarks_by_type(face_detail)
    w, h = float(image_width), float(image_height)

    def one_eye(prefix: str) -> tuple[float | None, list[str]]:
        # AWS names: leftEyeLeft, leftEyeRight, leftEyeUp, leftEyeDown (and rightEye*)
        need = [f"{prefix}Left", f"{prefix}Right", f"{prefix}Up", f"{prefix}Down"]
        miss = [k for k in need if k not in lm]
        if miss:
            return None, miss
        u, d = lm[f"{prefix}Up"], lm[f"{prefix}Down"]
        le, ri = lm[f"{prefix}Left"], lm[f"{prefix}Right"]
        vert = _dist_px(u, d, w, h)
        horiz = _dist_px(le, ri, w, h)
        if horiz < 1e-6:
            return None, [f"{prefix}:zero_width"]
        return vert / horiz, []

    el, ml = one_eye("leftEye")
    er, mr = one_eye("rightEye")
    missing = ml + mr
    if el is None and er is None:
        return {
            "ear_left": None,
            "ear_right": None,
            "ear_mean": None,
            "ok": False,
            "missing": missing,
        }
    if el is not None and er is not None:
        mean = (el + er) / 2.0
    elif el is not None:
        mean = el
    else:
        mean = er
    return {
        "ear_left": el,
        "ear_right": er,
        "ear_mean": mean,
        "ok": mean is not None,
        "missing": missing,
    }


def rule_based_frame_check(
    face_detail: dict[str, Any],
    image_width: int,
    image_height: int,
    *,
    ear_min: float = 0.12,
    ear_max: float = 0.45,
    max_abs_yaw: float = 25.0,
    max_abs_pitch: float = 25.0,
    min_sharpness: float | None = 0.4,
    require_eyes_open: bool = True,
) -> dict[str, Any]:
    """
    Example rule set for “is this frame good for recommendations / matching?”.

    Tune thresholds on your own data; these are starting points only.
    """
    reasons: list[str] = []
    ear = ear_ratio_from_face_detail(face_detail, image_width, image_height)

    if require_eyes_open:
        eo = face_detail.get("EyesOpen") or {}
        if eo.get("Value") is False:
            reasons.append("eyes_closed")
        elif eo.get("Value") is True:
            # API says open; do not fail on sub-80% label confidence (common for webcams / soft light).
            pass
        elif (eo.get("Confidence") or 0) < 80:
            # Value missing or unknown: only fail if landmarks do not contradict (open EAR range).
            m = ear.get("ear_mean")
            if m is None or not (ear_min <= m <= ear_max):
                reasons.append("eyes_open_low_confidence")

    pose = face_detail.get("Pose") or {}
    yaw = float(pose.get("Yaw") or 0.0)
    pitch = float(pose.get("Pitch") or 0.0)
    if abs(yaw) > max_abs_yaw:
        reasons.append("yaw_too_large")
    if abs(pitch) > max_abs_pitch:
        reasons.append("pitch_too_large")

    if min_sharpness is not None:
        q = face_detail.get("Quality") or {}
        sh = float(q.get("Sharpness") or 0.0)
        if sh < min_sharpness:
            reasons.append("too_blurry")

    m = ear.get("ear_mean")
    if m is not None:
        if m < ear_min:
            reasons.append("ear_too_low_squint_or_blink")
        if m > ear_max:
            reasons.append("ear_too_high_unusual")
    else:
        reasons.append("ear_not_computed")

    return {
        "recommend": len(reasons) == 0,
        "reasons": reasons,
        "ear": ear,
        "pose": {
            "Yaw": yaw,
            "Pitch": pitch,
            "Roll": float(pose.get("Roll") or 0),
        },
    }
