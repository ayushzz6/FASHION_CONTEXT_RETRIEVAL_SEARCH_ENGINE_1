"""Color palette + brightness extraction."""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import KMeans

COLOR_BINS = {
    "red": np.array([255, 0, 0]),
    "orange": np.array([255, 128, 0]),
    "yellow": np.array([255, 255, 0]),
    "green": np.array([0, 200, 0]),
    "cyan": np.array([0, 200, 200]),
    "blue": np.array([0, 0, 255]),
    "purple": np.array([128, 0, 200]),
    "pink": np.array([255, 105, 180]),
    "brown": np.array([150, 75, 0]),
    "black": np.array([0, 0, 0]),
    "gray": np.array([128, 128, 128]),
    "white": np.array([240, 240, 240]),
}

_CROP_BORDER_FRAC = 0.1
_MIN_SAT = 25
_MIN_VAL = 30
_MIN_PIXELS = 1000
_FG_MIN_FRAC = 0.05


def rgb_to_lab(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)


def dominant_colors(rgb_image: np.ndarray, k: int = 5) -> dict:
    """Return normalized color distribution and brightness score."""
    if rgb_image.size == 0:
        return {}, 0.0
    cropped = _center_crop(rgb_image, _CROP_BORDER_FRAC)
    fg_mask = _foreground_mask(cropped)
    lab = rgb_to_lab(cropped)
    hsv = cv2.cvtColor(cropped, cv2.COLOR_RGB2HSV)

    if fg_mask is None:
        fg_mask = np.ones(cropped.shape[:2], dtype=bool)

    if fg_mask.any():
        brightness = float(np.mean(lab[:, :, 0][fg_mask]) / 255.0)
    else:
        brightness = float(np.mean(lab[:, :, 0]) / 255.0)

    color_mask = fg_mask & (hsv[:, :, 1] >= _MIN_SAT) & (hsv[:, :, 2] >= _MIN_VAL)
    flat = lab.reshape(-1, 3)
    fg_pixels = lab[fg_mask] if fg_mask.any() else np.empty((0, 3), dtype=lab.dtype)
    color_pixels = lab[color_mask] if color_mask.any() else np.empty((0, 3), dtype=lab.dtype)

    if color_pixels.shape[0] >= _MIN_PIXELS:
        samples = color_pixels
    elif fg_pixels.shape[0] >= _MIN_PIXELS:
        samples = fg_pixels
    else:
        samples = flat

    if samples.shape[0] == 0:
        return {}, brightness

    k = min(k, len(samples))
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    km.fit(samples)
    counts = np.bincount(km.labels_)
    probs = counts / counts.sum()
    color_probs = {}
    for center, prob in zip(km.cluster_centers_, probs):
        rgb_center = cv2.cvtColor(center.astype("float32").reshape(1, 1, 3), cv2.COLOR_Lab2RGB).reshape(3)
        name = _closest_color_name(rgb_center)
        color_probs[name] = float(color_probs.get(name, 0.0) + prob)
    return color_probs, brightness


def _center_crop(image: np.ndarray, border_frac: float) -> np.ndarray:
    if image.size == 0:
        return image
    h, w = image.shape[:2]
    border = int(min(h, w) * border_frac)
    if border <= 0 or (h - 2 * border) <= 0 or (w - 2 * border) <= 0:
        return image
    return image[border : h - border, border : w - border]


def _foreground_mask(image: np.ndarray) -> np.ndarray | None:
    if image.size == 0:
        return None
    h, w = image.shape[:2]
    if h < 10 or w < 10:
        return None
    border = int(min(h, w) * 0.05)
    rect_w = w - 2 * border
    rect_h = h - 2 * border
    if rect_w <= 0 or rect_h <= 0:
        return None
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (border, border, rect_w, rect_h)
    try:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.grabCut(bgr, mask, rect, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    fg = (mask == 1) | (mask == 3)
    if fg.sum() < int(_FG_MIN_FRAC * h * w):
        return None
    return fg


def _closest_color_name(rgb: np.ndarray) -> str:
    min_dist = float("inf")
    best = "unknown"
    for name, ref in COLOR_BINS.items():
        dist = np.linalg.norm(rgb - ref)
        if dist < min_dist:
            min_dist = dist
            best = name
    return best
