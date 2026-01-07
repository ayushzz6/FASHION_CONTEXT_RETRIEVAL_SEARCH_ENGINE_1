"""Optional object detector for attribute metadata tags."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
except ImportError:  # pragma: no cover
    AutoImageProcessor = None
    AutoModelForObjectDetection = None

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None

from PIL import Image

DEFAULT_ALLOWED_LABELS = {"tie", "handbag", "backpack", "umbrella", "suitcase"}


class ObjectDetector:
    def __init__(
        self,
        backend: str = "detr",
        model_name: str = "facebook/detr-resnet-50",
        device: str = "cpu",
        min_score: float = 0.6,
        top_n: int = 8,
        allowed_labels: Iterable[str] | None = None,
        local_files_only: bool = True,
    ) -> None:
        self.backend = (backend or "none").lower()
        self.model_name = model_name
        self.device = device
        self.min_score = float(min_score)
        self.top_n = int(top_n) if top_n else 0
        self.local_files_only = bool(local_files_only)
        self._warned = False

        allowed_list = list(allowed_labels) if allowed_labels is not None else None
        if allowed_list is None:
            self.allowed_labels = set(DEFAULT_ALLOWED_LABELS)
        elif len(allowed_list) == 0:
            self.allowed_labels = None
        else:
            self.allowed_labels = {lab.lower() for lab in allowed_list}

        self.available = True
        self._disabled_reason = ""
        self._model = None
        self._processor = None

        if self.backend in {"none", "off", "disabled"}:
            self._disable("detector disabled")
            return
        if self.backend == "detr":
            self._init_detr()
        elif self.backend in {"yolo", "yolov8"}:
            self._init_yolo()
        else:
            self._disable(f"unknown backend '{self.backend}'")

    def detect(self, image_path: str) -> Dict[str, float]:
        if not self.available:
            self._warn_once()
            return {}
        if self.backend == "detr":
            return self._detect_detr(image_path)
        if self.backend in {"yolo", "yolov8"}:
            return self._detect_yolo(image_path)
        return {}

    def _init_detr(self) -> None:
        if AutoImageProcessor is None or AutoModelForObjectDetection is None or torch is None:
            self._disable("transformers/torch not available for DETR")
            return
        try:
            self._processor = AutoImageProcessor.from_pretrained(self.model_name, local_files_only=self.local_files_only)
            self._model = AutoModelForObjectDetection.from_pretrained(self.model_name, local_files_only=self.local_files_only)
            self._model.to(self.device)
            self._model.eval()
        except Exception as exc:  # pragma: no cover - depends on local cache
            self._disable(f"failed to load DETR model: {exc}")

    def _init_yolo(self) -> None:
        if YOLO is None:
            self._disable("ultralytics not installed for YOLOv8")
            return
        try:
            self._model = YOLO(self.model_name)
        except Exception as exc:  # pragma: no cover - depends on local cache
            self._disable(f"failed to load YOLOv8 model: {exc}")

    def _detect_detr(self, image_path: str) -> Dict[str, float]:
        if self._model is None or self._processor is None or torch is None:
            self._disable("DETR not initialized")
            self._warn_once()
            return {}
        image = Image.open(Path(image_path)).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        results = self._processor.post_process_object_detection(
            outputs,
            threshold=self.min_score,
            target_sizes=target_sizes,
        )[0]
        labels = [self._model.config.id2label[int(idx)].lower() for idx in results["labels"].tolist()]
        scores = results["scores"].tolist()
        return _filter_and_pack(labels, scores, self.allowed_labels, self.top_n)

    def _detect_yolo(self, image_path: str) -> Dict[str, float]:
        if self._model is None:
            self._disable("YOLO not initialized")
            self._warn_once()
            return {}
        results = self._model(image_path, conf=self.min_score, verbose=False)
        if not results:
            return {}
        result = results[0]
        names = getattr(self._model, "names", {}) or {}
        labels: List[str] = []
        scores: List[float] = []
        if hasattr(result, "boxes") and result.boxes is not None:
            classes = result.boxes.cls.tolist() if result.boxes.cls is not None else []
            confs = result.boxes.conf.tolist() if result.boxes.conf is not None else []
            for cls_idx, conf in zip(classes, confs):
                label = names.get(int(cls_idx), str(cls_idx)).lower()
                labels.append(label)
                scores.append(float(conf))
        return _filter_and_pack(labels, scores, self.allowed_labels, self.top_n)

    def _disable(self, reason: str) -> None:
        self.available = False
        self._disabled_reason = reason

    def _warn_once(self) -> None:
        if self._warned or not self._disabled_reason:
            return
        warnings.warn(f"ObjectDetector disabled: {self._disabled_reason}")
        self._warned = True


def _filter_and_pack(
    labels: List[str],
    scores: List[float],
    allowed: Optional[Iterable[str]],
    top_n: int,
) -> Dict[str, float]:
    merged: Dict[str, float] = {}
    allowed_set = set(lab.lower() for lab in allowed) if allowed else None
    for label, score in zip(labels, scores):
        if allowed_set is not None and label not in allowed_set:
            continue
        if label not in merged or score > merged[label]:
            merged[label] = float(score)
    if not merged:
        return {}
    items = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)
    if top_n:
        items = items[:top_n]
    return dict(items)
