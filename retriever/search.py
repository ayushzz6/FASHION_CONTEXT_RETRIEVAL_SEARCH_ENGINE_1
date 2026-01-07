"""Retriever search logic with reusable run_search function."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from indexer.storage.metadata_store import MetadataStore
from indexer.storage.vector_store import ChromaVectorStore
from indexer.models.clip_model import CLIPModel
from retriever.query.parse_query import parse_query
from retriever.rerank.compositional import assignment_score
from retriever.rerank.scoring import ScoreParts, color_score, env_score, garment_score, weighted_sum
from indexer.storage.patch_store import load_patches

cfg_path = Path("config.yaml")
cfg = yaml.safe_load(cfg_path.read_text())

clip_model = CLIPModel(
    cfg["model"]["clip_name"],
    cfg["model"]["clip_pretrained"],
    cfg["model"]["device"],
    cfg["model"].get("patch_name"),
    cfg["model"].get("patch_pretrained"),
)
vector_store = ChromaVectorStore(
    cfg["storage"]["chroma"]["persist_directory"],
    cfg["storage"]["chroma"]["collection"],
)
metadata_store = MetadataStore(cfg["storage"]["metadata_db"])

ENV_QUERY_PROMPTS = {
    "office": "a modern office interior",
    "street": "a city street",
    "park": "a green park",
    "home": "a home interior",
}


def _match_ratio(wanted: set[str], present: Dict[str, float], min_score: float) -> float:
    if not wanted:
        return 1.0
    hits = [key for key in wanted if present.get(key, 0.0) >= min_score]
    return len(hits) / max(1, len(wanted))


def _merge_item_scores(primary: Dict[str, float], secondary: Dict[str, float]) -> Dict[str, float]:
    merged = dict(primary)
    for key, score in secondary.items():
        if score > merged.get(key, 0.0):
            merged[key] = score
    return merged


def _passes_attribute_filters(parsed, meta: Dict[str, Any], config: Dict[str, Any]) -> bool:
    filters = config.get("retriever", {}).get("attribute_filters", {})
    if not filters.get("enabled", False):
        return True
    strict = bool(parsed.phrases)
    garment_min = filters.get("garment_min_score", 0.12)
    garment_ratio = filters.get("garment_match_ratio_strict" if strict else "garment_match_ratio", 0.5)
    color_min = filters.get("color_min_prob", 0.08)
    color_ratio = filters.get("color_match_ratio_strict" if strict else "color_match_ratio", 0.5)
    object_enabled = filters.get("object_tags_enabled", False)
    object_min = filters.get("object_min_score", 0.25)
    object_ratio = filters.get("object_match_ratio_strict" if strict else "object_match_ratio", 0.5)
    object_fallback = filters.get("object_use_garment_fallback", True)
    if parsed.garments:
        if _match_ratio(set(parsed.garments), meta.get("garments", {}), garment_min) < garment_ratio:
            return False
    if object_enabled and parsed.items:
        object_scores = meta.get("objects", {}) or {}
        if object_fallback:
            object_scores = _merge_item_scores(object_scores, meta.get("garments", {}))
        if _match_ratio(set(parsed.items), object_scores, object_min) < object_ratio:
            return False
    if parsed.colors:
        if _match_ratio(set(parsed.colors), meta.get("colors", {}), color_min) < color_ratio:
            return False
    if filters.get("require_env", False) and parsed.env:
        if meta.get("env_label") != parsed.env:
            return False
    return True


def run_search(
    query: str,
    k: int = cfg["retriever"]["return_top_k"],
    top_n: int = cfg["retriever"]["ann_top_n"],
    method: str = "greedy",
) -> List[Dict[str, Any]]:
    parsed = parse_query(query)
    text_global = clip_model.encode_text_global(query)
    candidates = vector_store.search(text_global, top_n=top_n)
    if parsed.env:
        query_env_embed = clip_model.encode_text_global(ENV_QUERY_PROMPTS.get(parsed.env, parsed.env))
    else:
        query_env_embed = None

    phrase_embeds = (
        np.stack([clip_model.encode_text_patch(p) for p in parsed.phrases])
        if parsed.phrases
        else np.zeros((0, clip_model.dim), dtype=np.float32)
    )
    results = []
    weights = cfg["retriever"]["weights"]
    query_garments = set(parsed.garments)
    if parsed.style:
        query_garments.add(parsed.style)
    for cand in candidates:
        meta = metadata_store.fetch(cand["id"])
        if not meta:
            continue
        if not _passes_attribute_filters(parsed, meta, cfg):
            continue
        patches, _coords = load_patches(meta["patch_path"])
        comp = assignment_score(phrase_embeds, patches, method=method) if phrase_embeds.size else 0.0
        col_score = color_score(parsed.colors, parsed.brightness_hint, meta.get("colors", {}), meta.get("brightness", 0.5))
        env_embed = np.array(meta["env_embed"], dtype=np.float32) if meta.get("env_embed") else None
        env_s = env_score(meta.get("env_label"), parsed.env, env_embed, query_env_embed)
        g_score = garment_score(query_garments, meta.get("garments", {}))
        global_sim = 1.0 - float(cand.get("distance", 1.0))
        parts = ScoreParts(global_sim=global_sim, compositional=comp, color=col_score, env=env_s, garments=g_score)
        score = weighted_sum(parts, weights)
        results.append(
            {
                "id": cand["id"],
                "score": score,
                "parts": parts.__dict__,
                "path": meta.get("path"),
                "env_label": meta.get("env_label"),
                "colors": meta.get("colors"),
                "garments": meta.get("garments"),
                "objects": meta.get("objects"),
            }
        )
    results = sorted(results, key=lambda r: r["score"], reverse=True)[:k]
    return results
