"""CLIP style preference (text and/or image) merged with the rule-based ranker."""

from app.preference.clip_scoring import is_available as clip_backend_available
from app.preference.hybrid_merge import apply_preference_hybrid, preference_scores_for_products

__all__ = [
    "apply_preference_hybrid",
    "clip_backend_available",
    "preference_scores_for_products",
]
