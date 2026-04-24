"""
CLIP text + image encoders (L2-normalized float32 vectors; pair with FAISS IndexFlatIP for cosine).
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from typing import Any

from eyewear_recommender import config


def _as_feature_tensor(x: Any) -> "torch.Tensor":
    """
    huggingface/transformers may return a raw tensor or a ModelOutput (e.g. BaseModelOutputWithPooling).
    """
    if isinstance(x, torch.Tensor):
        return x
    for attr in ("text_embeds", "image_embeds", "pooler_output"):
        v = getattr(x, attr, None)
        if v is not None and isinstance(v, torch.Tensor):
            return v
    hs = getattr(x, "last_hidden_state", None)
    if hs is not None and isinstance(hs, torch.Tensor):
        return hs[:, 0, :]
    raise TypeError(f"Cannot get feature tensor from {type(x)!r}")


class CLIPBackend:
    def __init__(self, model_id: str | None = None, device: str | None = None) -> None:
        from transformers import CLIPModel, CLIPProcessor

        model_id = model_id or config.CLIP_MODEL_ID
        self.model_id = model_id
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.model = CLIPModel.from_pretrained(model_id)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        cfg = self.model.config
        self.dim = int(
            getattr(cfg, "projection_dim", None)
            or getattr(getattr(cfg, "text_config", cfg), "hidden_size", 512)
        )

    @torch.inference_mode()
    def encode_text(self, text: str) -> np.ndarray:
        if not (text or "").strip():
            z = np.zeros((self.dim,), dtype=np.float32)
            n = float(np.linalg.norm(z))
            return z if n == 0.0 else (z / n).astype(np.float32)
        inputs = self.processor(
            text=[text], return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        fe = _as_feature_tensor(self.model.get_text_features(**inputs))
        n = fe.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        fe = fe / n
        return fe.float().cpu().numpy().astype(np.float32)[0]

    @torch.inference_mode()
    def encode_image(self, image: Image.Image) -> np.ndarray:
        if image.mode != "RGB":
            image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        fe = _as_feature_tensor(self.model.get_image_features(**inputs))
        n = fe.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        fe = fe / n
        return fe.float().cpu().numpy().astype(np.float32)[0]

    @torch.inference_mode()
    def encode_images_batch(self, images: list[Image.Image]) -> np.ndarray:
        imgs = [im.convert("RGB") if im.mode != "RGB" else im for im in images]
        inputs = self.processor(images=imgs, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        fe = _as_feature_tensor(self.model.get_image_features(**inputs))
        n = fe.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        fe = fe / n
        return fe.float().cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def encode_text_and_image_fused(
        self, text: str, image: Image.Image, text_weight: float = 0.5
    ) -> np.ndarray:
        t = self.encode_text(text)
        i = self.encode_image(image)
        w = max(0.0, min(1.0, text_weight))
        f = w * t + (1.0 - w) * i
        n = float(np.linalg.norm(f))
        if n < 1e-12:
            return t
        f = f / n
        return f.astype(np.float32)


def load_clip(model_id: str | None = None, device: str | None = None) -> CLIPBackend:
    return CLIPBackend(model_id=model_id, device=device)
