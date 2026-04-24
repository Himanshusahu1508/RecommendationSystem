"""
Shared face-count validation for pipelines that require exactly one face.
"""

from __future__ import annotations

from typing import Any


def single_face_or_error(face_details: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str | None, int]:
    """
    Return (face_detail, None, 1) if exactly one face.
    Otherwise return (None, error_code, count) with error_code 'no_face' or 'multiple_faces'.
    """
    n = len(face_details)
    if n == 0:
        return None, "no_face", 0
    if n > 1:
        return None, "multiple_faces", n
    return face_details[0], None, 1
