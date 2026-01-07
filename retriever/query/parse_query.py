"""Rule-based query parser for colors, garments, items, and context."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple


COLOR_SYNONYMS = {
    "red": [r"red", r"crimson", r"maroon", r"burgundy"],
    "blue": [r"blue", r"navy", r"denim", r"azure"],
    "yellow": [r"yellow", r"gold", r"golden"],
    "green": [r"green", r"olive", r"emerald"],
    "cyan": [r"cyan", r"teal", r"turquoise"],
    "purple": [r"purple", r"violet", r"lavender"],
    "pink": [r"pink", r"magenta", r"fuchsia"],
    "orange": [r"orange"],
    "brown": [r"brown", r"beige", r"tan", r"khaki"],
    "black": [r"black"],
    "gray": [r"gray", r"grey", r"charcoal"],
    "white": [r"white", r"cream", r"ivory", r"off[-\\s]?white"],
}

GARMENT_SYNONYMS = {
    "shirt": [r"shirt", r"button[-\\s]?down", r"dress shirt", r"oxford"],
    "t-shirt": [r"t[-\\s]?shirt", r"tee"],
    "hoodie": [r"hoodie", r"sweatshirt"],
    "jacket": [r"jacket", r"bomber", r"denim jacket"],
    "coat": [r"coat", r"overcoat", r"trench coat", r"parka"],
    "raincoat": [r"raincoat", r"rain coat"],
    "blazer": [r"blazer"],
    "suit": [r"suit", r"suit jacket"],
    "tie": [r"tie", r"necktie"],
    "pants": [r"pants", r"trousers", r"chinos"],
    "jeans": [r"jeans", r"denim"],
    "dress": [r"dress", r"gown"],
    "skirt": [r"skirt"],
    "sweater": [r"sweater", r"pullover", r"knitwear"],
}

ITEM_SYNONYMS = {
    "tie": [r"tie", r"necktie"],
    "handbag": [r"handbag", r"purse", r"bag"],
    "backpack": [r"backpack", r"rucksack"],
    "umbrella": [r"umbrella"],
    "suitcase": [r"suitcase", r"luggage"],
}

ENV_HINTS: List[Tuple[str, str]] = [
    ("office", "office"),
    ("corporate", "office"),
    ("conference", "office"),
    ("workspace", "office"),
    ("workplace", "office"),
    ("park", "park"),
    ("garden", "park"),
    ("bench", "park"),
    ("street", "street"),
    ("city", "street"),
    ("downtown", "street"),
    ("sidewalk", "street"),
    ("urban", "street"),
    ("home", "home"),
    ("living room", "home"),
    ("bedroom", "home"),
    ("kitchen", "home"),
    ("indoors", "home"),
]

STYLE_HINTS: List[Tuple[str, str]] = [
    ("formal", "formal"),
    ("professional", "formal"),
    ("business", "formal"),
    ("officewear", "formal"),
    ("smart", "formal"),
    ("casual", "casual"),
    ("weekend", "casual"),
    ("streetwear", "casual"),
]

WEATHER_HINTS: List[Tuple[str, str]] = [
    ("rain", "rainy"),
    ("raincoat", "rainy"),
    ("sunny", "sunny"),
    ("snow", "snowy"),
    ("snowy", "snowy"),
    ("cloudy", "cloudy"),
]


@dataclass
class ParsedQuery:
    phrases: List[str] = field(default_factory=list)
    colors: Set[str] = field(default_factory=set)
    garments: Set[str] = field(default_factory=set)
    items: Set[str] = field(default_factory=set)
    env: Optional[str] = None
    style: Optional[str] = None
    weather: Optional[str] = None
    brightness_hint: Optional[str] = None  # "bright" or "dark"


def parse_query(text: str) -> ParsedQuery:
    lower = text.lower()
    phrases: List[str] = []
    colors = _extract_terms(lower, COLOR_SYNONYMS)
    garments = _extract_terms(lower, GARMENT_SYNONYMS)
    items = _extract_terms(lower, ITEM_SYNONYMS)
    phrases = _extract_color_garment_phrases(lower, colors, garments)
    if not phrases:
        phrases = [p.strip() for p in re.split(r"and|,", text) if p.strip()]
    env = next((env for key, env in ENV_HINTS if re.search(rf"\b{re.escape(key)}\b", lower)), None)
    style = next((style for key, style in STYLE_HINTS if re.search(rf"\b{re.escape(key)}\b", lower)), None)
    weather = next((w for key, w in WEATHER_HINTS if re.search(rf"\b{re.escape(key)}\b", lower)), None)
    brightness_hint = _brightness_hint(lower)
    return ParsedQuery(
        phrases=phrases,
        colors=colors,
        garments=garments,
        items=items,
        env=env,
        style=style,
        weather=weather,
        brightness_hint=brightness_hint,
    )


def _extract_terms(text: str, vocab: dict) -> Set[str]:
    found: Set[str] = set()
    for canonical, patterns in vocab.items():
        for pattern in patterns:
            if re.search(rf"\b{pattern}\b", text):
                found.add(canonical)
                break
    return found


def _extract_color_garment_phrases(text: str, colors: Set[str], garments: Set[str]) -> List[str]:
    phrases: List[str] = []
    for color in colors:
        for garment in garments:
            for cpat in COLOR_SYNONYMS[color]:
                for gpat in GARMENT_SYNONYMS[garment]:
                    if re.search(rf"\b{cpat}\b\s+\b{gpat}\b", text):
                        phrase = f"{color} {garment}"
                        if phrase not in phrases:
                            phrases.append(phrase)
    return phrases


def _brightness_hint(text: str) -> Optional[str]:
    if re.search(r"\b(bright|vibrant|neon|light)\b", text):
        return "bright"
    if re.search(r"\b(dark|dim|black)\b", text):
        return "dark"
    return None
