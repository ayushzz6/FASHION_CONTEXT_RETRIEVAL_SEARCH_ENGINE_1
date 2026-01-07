"""Scoring utilities combining global, compositional, color, env, and garment signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np


@dataclass
class ScoreParts:
    global_sim: float = 0.0
    compositional: float = 0.0
    color: float = 0.0
    env: float = 0.0
    garments: float = 0.0


def weighted_sum(parts: ScoreParts, weights: Dict[str, float]) -> float:
    return (
        weights.get("global", 0.0) * parts.global_sim
        + weights.get("compositional", 0.0) * parts.compositional
        + weights.get("color", 0.0) * parts.color
        + weights.get("env", 0.0) * parts.env
        + weights.get("garments", 0.0) * parts.garments
    )


def color_score(query_colors: Iterable[str], brightness_hint: Optional[str], palette: Dict[str, float], brightness: float) -> float:
    if not query_colors:
        return 0.0
    score = 0.0
    for c in query_colors:
        score += palette.get(c, 0.0)
    score /= len(list(query_colors))
    if brightness_hint == "bright":
        score *= min(1.0, brightness + 0.1)
    elif brightness_hint == "dark":
        score *= min(1.0, 1.0 - brightness + 0.1)
    return float(score)


def env_score(env_label: Optional[str], desired_env: Optional[str], env_embed: Optional[np.ndarray], query_env_embed: Optional[np.ndarray]) -> float:
    base = 0.0
    if env_label and desired_env and env_label == desired_env:
        base = 1.0
    fuzzy = 0.0
    if env_embed is not None and query_env_embed is not None:
        fuzzy = float(similarity(env_embed, query_env_embed))
    return 0.7 * base + 0.3 * fuzzy


def garment_score(query_garments: Iterable[str], stored_garments: Dict[str, float]) -> float:
    if not query_garments:
        return 0.0
    total = 0.0
    for g in query_garments:
        total += stored_garments.get(g, 0.0)
    return float(total / len(list(query_garments)))


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)
