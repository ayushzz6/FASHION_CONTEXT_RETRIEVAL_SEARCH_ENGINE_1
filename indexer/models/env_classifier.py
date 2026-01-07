"""Environment and garment prompt-based classifiers using CLIP text embeddings."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .clip_model import CLIPModel


ENV_PROMPTS = {
    "office": [
        "a modern office interior",
        "a corporate office",
        "a conference room",
        "a coworking space",
    ],
    "street": [
        "a city street",
        "an urban street",
        "a downtown sidewalk",
        "a busy street",
    ],
    "park": [
        "a green park",
        "a park bench",
        "a public park",
        "a park pathway",
    ],
    "home": [
        "a home interior",
        "a living room",
        "a cozy bedroom",
        "a kitchen at home",
    ],
}

COLOR_PROMPTS = {
    "red": ["a red outfit", "a person wearing red", "red clothing"],
    "blue": ["a blue outfit", "a person wearing blue", "blue clothing"],
    "yellow": ["a yellow outfit", "a person wearing yellow", "yellow clothing"],
    "green": ["a green outfit", "a person wearing green", "green clothing"],
    "cyan": ["a teal outfit", "a person wearing teal", "teal clothing"],
    "purple": ["a purple outfit", "a person wearing purple", "purple clothing"],
    "pink": ["a pink outfit", "a person wearing pink", "pink clothing"],
    "orange": ["an orange outfit", "a person wearing orange", "orange clothing"],
    "brown": ["a brown outfit", "a person wearing brown", "brown clothing"],
    "black": ["a black outfit", "a person wearing black", "black clothing"],
    "gray": ["a gray outfit", "a person wearing gray", "gray clothing"],
    "white": ["a white outfit", "a person wearing white", "white clothing"],
}

GARMENT_PROMPTS = {
    "formal": ["blazer", "suit", "button-down shirt", "necktie", "formal trousers", "business attire", "office wear"],
    "casual": ["hoodie", "t-shirt", "jeans", "sneakers", "casual outfit", "weekend outfit"],
    "outerwear": ["coat", "jacket", "raincoat", "puffer jacket", "overcoat", "windbreaker"],
    "shirt": ["shirt", "button-down shirt", "dress shirt", "oxford shirt"],
    "t-shirt": ["t-shirt", "tee"],
    "tie": ["tie", "necktie"],
    "hoodie": ["hoodie", "sweatshirt"],
    "jacket": ["jacket", "blazer", "denim jacket", "bomber jacket"],
    "coat": ["coat", "raincoat", "overcoat", "trench coat"],
    "raincoat": ["raincoat", "rain coat"],
    "blazer": ["blazer"],
    "suit": ["suit", "suit jacket"],
    "pants": ["pants", "trousers", "chinos"],
    "jeans": ["jeans", "denim pants"],
    "dress": ["dress", "gown"],
    "skirt": ["skirt"],
    "sweater": ["sweater", "pullover", "knitwear"],
}


class EnvAndGarmentClassifier:
    def __init__(
        self,
        clip_model: CLIPModel,
        env_prompts: Dict[str, List[str]] | List[str] | None = None,
        garment_prompts: Dict[str, List[str]] | None = None,
    ) -> None:
        self.clip_model = clip_model
        self.env_prompts = _normalize_env_prompts(env_prompts)
        self.env_prompt_embeds = {
            label: np.stack([clip_model.encode_text_global(p) for p in prompts]) for label, prompts in self.env_prompts.items()
        }
        self.env_label_embeds = {label: _mean_embed(embeds) for label, embeds in self.env_prompt_embeds.items()}
        self.garment_prompts = garment_prompts or GARMENT_PROMPTS
        self.garment_embeds_global = {
            tag: np.stack([clip_model.encode_text_global(p) for p in prompts]) for tag, prompts in self.garment_prompts.items()
        }
        self.garment_embeds_patch = {
            tag: np.stack([clip_model.encode_text_patch(p) for p in prompts]) for tag, prompts in self.garment_prompts.items()
        }
        self.color_prompt_embeds = {
            color: np.stack([clip_model.encode_text_global(p) for p in prompts]) for color, prompts in COLOR_PROMPTS.items()
        }

    def classify_env(self, global_emb: np.ndarray) -> Tuple[str, float, List[float]]:
        best_label = None
        best_score = -1.0
        for label, embeds in self.env_prompt_embeds.items():
            sims = embeds @ global_emb / (np.linalg.norm(embeds, axis=1) * np.linalg.norm(global_emb) + 1e-8)
            score = float(np.max(sims))
            if score > best_score:
                best_label = label
                best_score = score
        if best_label is None:
            return "unknown", 0.0, []
        return best_label, float(best_score), self.env_label_embeds[best_label].tolist()

    def classify_garments(self, global_emb: np.ndarray, top_n: int = 3) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for tag, embeds in self.garment_embeds_global.items():
            sims = embeds @ global_emb / (np.linalg.norm(embeds, axis=1) * np.linalg.norm(global_emb) + 1e-8)
            scores[tag] = float(np.max(sims))
        sorted_items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        return dict(sorted_items)

    def classify_garments_from_patches(self, patch_embeds: np.ndarray, top_n: int = 5, min_score: float = 0.12) -> Dict[str, float]:
        if patch_embeds.size == 0:
            return {}
        scores: Dict[str, float] = {}
        for tag, embeds in self.garment_embeds_patch.items():
            sims = patch_embeds @ embeds.T
            scores[tag] = float(np.max(sims)) if sims.size else 0.0
        filtered = {k: v for k, v in scores.items() if v >= min_score}
        if not filtered:
            return {}
        sorted_items = sorted(filtered.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
        return dict(sorted_items)

    def classify_colors(self, global_emb: np.ndarray, top_n: int = 5) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for color, embeds in self.color_prompt_embeds.items():
            sims = embeds @ global_emb / (np.linalg.norm(embeds, axis=1) * np.linalg.norm(global_emb) + 1e-8)
            scores[color] = float(np.max(sims))
        if not scores:
            return {}
        probs = _softmax(np.array(list(scores.values())))
        colors = list(scores.keys())
        sorted_items = sorted(zip(colors, probs), key=lambda kv: kv[1], reverse=True)
        if top_n:
            sorted_items = sorted_items[:top_n]
        return {c: float(p) for c, p in sorted_items}


def _softmax(values: np.ndarray, temperature: float = 10.0) -> np.ndarray:
    scaled = values * temperature
    scaled = scaled - np.max(scaled)
    exp = np.exp(scaled)
    denom = np.sum(exp)
    if denom == 0.0:
        return np.zeros_like(exp)
    return exp / denom


def _mean_embed(embeds: np.ndarray) -> np.ndarray:
    if embeds.size == 0:
        return embeds
    mean = np.mean(embeds, axis=0)
    denom = np.linalg.norm(mean) + 1e-8
    return mean / denom


def _normalize_env_prompts(env_prompts: Dict[str, List[str]] | List[str] | None) -> Dict[str, List[str]]:
    if env_prompts is None:
        return ENV_PROMPTS
    if isinstance(env_prompts, dict):
        return env_prompts
    mapping = {key: [] for key in ENV_PROMPTS.keys()}
    for prompt in env_prompts:
        lower = prompt.lower()
        if "office" in lower or "corporate" in lower:
            mapping["office"].append(prompt)
        elif "park" in lower or "garden" in lower:
            mapping["park"].append(prompt)
        elif "home" in lower or "living" in lower or "bedroom" in lower or "kitchen" in lower:
            mapping["home"].append(prompt)
        elif "street" in lower or "city" in lower or "urban" in lower or "sidewalk" in lower:
            mapping["street"].append(prompt)
    for key, prompts in mapping.items():
        if not prompts:
            mapping[key] = ENV_PROMPTS[key]
    return mapping
