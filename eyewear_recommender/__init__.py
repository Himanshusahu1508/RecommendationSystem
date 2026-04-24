"""CLIP (Hugging Face) + FAISS index helpers for offline embedding builds."""

from eyewear_recommender.clip_backend import CLIPBackend, load_clip
from eyewear_recommender.faiss_index import FAISSIndex, build_index, search

__all__ = [
    "CLIPBackend",
    "FAISSIndex",
    "build_index",
    "load_clip",
    "search",
]
