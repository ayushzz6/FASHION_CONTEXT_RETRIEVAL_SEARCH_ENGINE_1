"""CLIP wrapper returning global and patch embeddings (with token projection)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

try:
    import open_clip
except ImportError:  # pragma: no cover
    open_clip = None

class CLIPModel:
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cpu",
        patch_model_name: str | None = None,
        patch_pretrained: str | None = None,
    ) -> None:
        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained
        self.patch_model_name = patch_model_name
        self.patch_pretrained = patch_pretrained
        self.semantic_model = None
        self.semantic_preprocess = None
        self.semantic_tokenizer = None
        self.patch_model = None
        self.patch_preprocess = None
        self.patch_tokenizer = None
        self.dim = 512
        self.semantic_dim = 512
        self._patch_tokens = None

        self._init_open_clip_model(model_name, pretrained)
        if patch_model_name and patch_pretrained:
            if (patch_model_name, patch_pretrained) != (model_name, pretrained):
                self._init_patch_model(patch_model_name, patch_pretrained)

    def encode_text(self, text: str) -> np.ndarray:
        return self.encode_text_global(text)

    def encode_text_global(self, text: str) -> np.ndarray:
        return _encode_open_clip_text(self.semantic_model, self.semantic_tokenizer, self.device, text, self.semantic_dim)

    def encode_text_patch(self, text: str) -> np.ndarray:
        if self.patch_model is None:
            return self.encode_text_global(text)
        return _encode_open_clip_text(self.patch_model, self.patch_tokenizer, self.device, text, self.dim)

    def encode_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """Return global embedding, patch embeddings, and patch coords."""
        if self.semantic_model is self.patch_model:
            return _encode_open_clip_image(
                self.semantic_model,
                self.semantic_preprocess,
                self.device,
                image_path,
                self._get_patch_tokens,
                self._clear_patch_tokens,
            )
        global_emb = self.encode_image_global(image_path)
        patches, coords = self.encode_image_patches(image_path)
        return global_emb, patches, coords

    def encode_image_global(self, image_path: str) -> np.ndarray:
        global_emb, _patches, _coords = _encode_open_clip_image(
            self.semantic_model,
            self.semantic_preprocess,
            self.device,
            image_path,
            None,
            None,
        )
        return global_emb

    def encode_image_patches(self, image_path: str) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        if self.patch_model is None or self.patch_preprocess is None:
            rng = np.random.default_rng(abs(hash(image_path)) % (2**32))
            patches = rng.normal(size=(196, self.dim))
            patches = patches / np.linalg.norm(patches, axis=1, keepdims=True)
            coords = _grid_coords(int(math.sqrt(len(patches))))
            return patches.astype(np.float32), coords
        _global, patches, coords = _encode_open_clip_image(
            self.patch_model,
            self.patch_preprocess,
            self.device,
            image_path,
            self._get_patch_tokens,
            self._clear_patch_tokens,
        )
        return patches, coords

    def _save_patch_tokens(self, _module, _input, output) -> None:
        self._patch_tokens = output

    def _get_patch_tokens(self):
        return self._patch_tokens

    def _clear_patch_tokens(self) -> None:
        self._patch_tokens = None

    def _init_patch_model(self, model_name: str, pretrained: str) -> None:
        if open_clip is None:
            raise ImportError("open_clip is required for patch embeddings. Please `pip install open-clip-torch`.")
        self.patch_model, _, self.patch_preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            force_quick_gelu=False,
        )
        self.patch_model.to(self.device)
        self.patch_model.eval()
        self.patch_tokenizer = open_clip.get_tokenizer(model_name)
        if hasattr(self.patch_model, "text_projection"):
            self.dim = self.patch_model.text_projection.shape[1]
        if hasattr(self.patch_model, "visual") and hasattr(self.patch_model.visual, "transformer"):
            self.patch_model.visual.transformer.register_forward_hook(self._save_patch_tokens)

    def _init_open_clip_model(self, model_name: str, pretrained: str) -> None:
        if open_clip is None:
            return
        self.semantic_model, _, self.semantic_preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            force_quick_gelu=False,
        )
        self.semantic_model.to(self.device)
        self.semantic_model.eval()
        self.semantic_tokenizer = open_clip.get_tokenizer(model_name)
        if hasattr(self.semantic_model, "text_projection"):
            self.semantic_dim = self.semantic_model.text_projection.shape[1]
            self.dim = self.semantic_dim
        self.patch_model = self.semantic_model
        self.patch_preprocess = self.semantic_preprocess
        self.patch_tokenizer = self.semantic_tokenizer
        if hasattr(self.semantic_model, "visual") and hasattr(self.semantic_model.visual, "transformer"):
            self.semantic_model.visual.transformer.register_forward_hook(self._save_patch_tokens)


def _grid_coords(side: int) -> List[Tuple[int, int]]:
    coords: List[Tuple[int, int]] = []
    for i in range(side):
        for j in range(side):
            coords.append((i, j))
    return coords


def _encode_open_clip_text(model, tokenizer, device: str, text: str, dim: int) -> np.ndarray:
    if model is None or tokenizer is None:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        vec = rng.normal(size=(dim,))
        vec = vec / np.linalg.norm(vec)
        return vec.astype(np.float32)
    with torch.no_grad():
        tokens = tokenizer([text]).to(device)
        txt = model.encode_text(tokens)
        txt = txt / txt.norm(dim=-1, keepdim=True)
        return txt.cpu().numpy()[0].astype(np.float32)


def _encode_open_clip_image(
    model,
    preprocess,
    device: str,
    image_path: str,
    get_tokens,
    clear_tokens,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
    if model is None or preprocess is None:
        rng = np.random.default_rng(abs(hash(image_path)) % (2**32))
        dim = 512
        global_emb = rng.normal(size=(dim,))
        global_emb = global_emb / np.linalg.norm(global_emb)
        patches = rng.normal(size=(196, dim))
        patches = patches / np.linalg.norm(patches, axis=1, keepdims=True)
        coords = _grid_coords(int(math.sqrt(len(patches))))
        return global_emb.astype(np.float32), patches.astype(np.float32), coords
    img = Image.open(Path(image_path)).convert("RGB")
    with torch.no_grad():
        if clear_tokens is not None:
            clear_tokens()
        pixel = preprocess(img).unsqueeze(0).to(device)
        global_emb = model.encode_image(pixel)
        global_emb = global_emb / global_emb.norm(dim=-1, keepdim=True)
        patches_np = None
        tokens = get_tokens() if get_tokens is not None else None
        if tokens is not None and isinstance(tokens, torch.Tensor) and tokens.ndim == 3:
            patches = tokens[0, 1:, :]
            if hasattr(model.visual, "proj") and model.visual.proj is not None:
                patches = patches @ model.visual.proj
            patches = patches / (patches.norm(dim=-1, keepdim=True) + 1e-8)
            patches_np = patches.cpu().numpy().astype(np.float32)
        if patches_np is None:
            patches_np = np.repeat(global_emb.cpu().numpy(), repeats=49, axis=0).astype(np.float32)
        coords = _grid_coords(int(math.sqrt(patches_np.shape[0])) or 1)
        return global_emb.cpu().numpy()[0].astype(np.float32), patches_np, coords


