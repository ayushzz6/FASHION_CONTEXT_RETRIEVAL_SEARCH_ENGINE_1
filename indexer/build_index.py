"""Indexer CLI: extract globals, patches, colors, env, garments, objects, and store artifacts."""

from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import typer
import yaml
from tqdm import tqdm

from indexer.storage.metadata_store import MetadataStore
from indexer.storage.vector_store import ChromaVectorStore
from indexer.models.clip_model import CLIPModel
from indexer.models.color_extractor import dominant_colors
from indexer.models.env_classifier import EnvAndGarmentClassifier
from indexer.storage.patch_store import save_patches
from indexer.models.object_detector import ObjectDetector

app = typer.Typer()


def _load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _image_id(path: str) -> str:
    return hashlib.md5(path.encode("utf-8")).hexdigest()


def _normalize_probs(probs: Dict[str, float]) -> Dict[str, float]:
    if not probs:
        return {}
    total = float(sum(max(v, 0.0) for v in probs.values()))
    if total <= 0.0:
        return {}
    return {k: float(max(v, 0.0)) / total for k, v in probs.items()}


def _blend_color_sources(palette: Dict[str, float], clip_colors: Dict[str, float], palette_weight: float) -> Dict[str, float]:
    if not clip_colors:
        return _normalize_probs(palette)
    palette = _normalize_probs(palette)
    clip_colors = _normalize_probs(clip_colors)
    blended: Dict[str, float] = {}
    for key in set(palette.keys()) | set(clip_colors.keys()):
        blended[key] = palette_weight * palette.get(key, 0.0) + (1.0 - palette_weight) * clip_colors.get(key, 0.0)
    return _normalize_probs(blended)


def _merge_scores(primary: Dict[str, float], secondary: Dict[str, float], primary_weight: float) -> Dict[str, float]:
    if not primary:
        return secondary
    if not secondary:
        return primary
    merged: Dict[str, float] = {}
    for key in set(primary.keys()) | set(secondary.keys()):
        merged[key] = primary_weight * primary.get(key, 0.0) + (1.0 - primary_weight) * secondary.get(key, 0.0)
    return merged


@app.command()
def build_index(
    images: str | None = typer.Option(None, help="Glob pattern or directory for images (falls back to config indexer.default_images_path)"),
    config: str = typer.Option("config.yaml", help="Config YAML path"),
) -> None:
    cfg = _load_config(config)
    images_arg = images or cfg.get("indexer", {}).get("default_images_path")
    if not images_arg:
        typer.echo("No images path provided and indexer.default_images_path missing in config.")
        raise typer.Exit(code=1)

    images_path = Path(images_arg).resolve()
    if images_path.is_dir():
        img_paths = sorted(
            glob.glob(str(images_path / "**" / "*.jpg"), recursive=True)
            + glob.glob(str(images_path / "**" / "*.png"), recursive=True)
        )
    else:
        img_paths = sorted(glob.glob(str(images_path)))
    if not img_paths:
        typer.echo(f"No images found for: {images_path}")
        raise typer.Exit(code=1)

    clip_model = CLIPModel(
        cfg["model"]["clip_name"],
        cfg["model"]["clip_pretrained"],
        cfg["model"]["device"],
        cfg["model"].get("patch_name"),
        cfg["model"].get("patch_pretrained"),
    )
    classifier = EnvAndGarmentClassifier(clip_model, env_prompts=cfg["retriever"].get("env_prompts"))
    vector_store = ChromaVectorStore(cfg["storage"]["chroma"]["persist_directory"], cfg["storage"]["chroma"]["collection"])
    metadata_store = MetadataStore(cfg["storage"]["metadata_db"])
    detector_cfg = cfg.get("indexer", {}).get("detector", {})
    object_detector = ObjectDetector(
        backend=detector_cfg.get("backend", "none"),
        model_name=detector_cfg.get("model_name", "facebook/detr-resnet-50"),
        device=cfg["model"]["device"],
        min_score=detector_cfg.get("min_score", 0.6),
        top_n=detector_cfg.get("top_n", 8),
        allowed_labels=detector_cfg.get("allowed_labels"),
        local_files_only=detector_cfg.get("local_files_only", True),
    )

    batch_embeds = []
    batch_ids = []
    batch_meta = []

    for idx, path in enumerate(tqdm(img_paths, desc="Indexing", unit="img", ascii=True)):
        image_id = _image_id(path)
        global_emb, patch_embeds, coords = clip_model.encode_image(path)
        # color palette
        bgr = cv2.imread(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else np.zeros((1, 1, 3), dtype=np.uint8)
        palette, brightness = dominant_colors(rgb, k=cfg["indexer"]["top_colors"])
        clip_colors = classifier.classify_colors(global_emb, top_n=cfg["indexer"].get("clip_color_top_n", 5))
        colors = _blend_color_sources(palette, clip_colors, cfg["indexer"].get("color_palette_weight", 0.7))
        # env + garments
        env_label, env_conf, env_embed = classifier.classify_env(global_emb)
        garment_top_n = cfg["indexer"].get("garment_top_n", 5)
        garment_min_score = cfg["indexer"].get("garment_min_score", 0.12)
        garment_patch_weight = cfg["indexer"].get("garment_patch_weight", 0.7)
        garments_patch = classifier.classify_garments_from_patches(patch_embeds, top_n=garment_top_n, min_score=garment_min_score)
        garments_global = classifier.classify_garments(global_emb, top_n=garment_top_n)
        garments = _merge_scores(garments_patch, garments_global, garment_patch_weight)
        # object detector tags
        objects = object_detector.detect(path)

        patch_path = save_patches(image_id, patch_embeds, coords, cfg["storage"]["patches_dir"])
        metadata_store.upsert_image(
            image_id=image_id,
            path=str(Path(path).resolve()),
            env_label=env_label,
            env_conf=env_conf,
            env_embed=env_embed,
            patch_path=patch_path,
            brightness=brightness,
            colors=colors,
            garments=garments,
            objects=objects,
        )
        batch_embeds.append(global_emb)
        batch_ids.append(image_id)
        batch_meta.append({"path": str(Path(path).resolve()), "env_label": env_label, "colors": json.dumps(colors)})

        if len(batch_ids) >= cfg["indexer"]["batch_size"]:
            vector_store.add(batch_ids, np.stack(batch_embeds), batch_meta)
            batch_embeds, batch_ids, batch_meta = [], [], []

    if batch_ids:
        vector_store.add(batch_ids, np.stack(batch_embeds), batch_meta)
    typer.echo(f"Indexed {len(img_paths)} images into Chroma and metadata store.")


if __name__ == "__main__":
    app()
