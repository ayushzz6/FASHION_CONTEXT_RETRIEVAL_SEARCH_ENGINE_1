"""Disk helpers for patch embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np


def save_patches(image_id: str, patches: np.ndarray, coords: List[Tuple[int, int]], directory: str) -> str:
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{image_id}.npz"
    np.savez_compressed(path, patches=patches.astype(np.float16), coords=np.array(coords, dtype=np.int16))
    return str(path)


def load_patches(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["patches"], data["coords"]
