"""
Grounding DINO open-vocabulary object detector.

Wraps the GroundingDINO model to accept an RGB image and a free-form
text prompt, returning bounding boxes with labels and confidence scores.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from soarm_gemini.config import (
    DETECTION_CONFIDENCE_THRESHOLD,
    DETECTION_NMS_THRESHOLD,
    GROUNDING_DINO_CONFIG,
    GROUNDING_DINO_WEIGHTS,
)

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single detected object."""
    label: str
    bbox_xyxy: List[float]  # [x1, y1, x2, y2] in pixel coords
    confidence: float
    centre_uv: Optional[List[float]] = field(default=None)

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox_xyxy
        self.centre_uv = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


class GroundingDINODetector:
    """Lazy-loaded Grounding DINO inference wrapper.

    The model weights are loaded on the first call to :meth:`detect` so that
    import time stays low and the GPU memory is only allocated when needed.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self._config_path = config_path or GROUNDING_DINO_CONFIG
        self._weights_path = weights_path or GROUNDING_DINO_WEIGHTS
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None

    # ── lazy loading ─────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load Grounding DINO model weights into GPU/CPU memory."""
        from groundingdino.util.inference import load_model

        logger.info(
            "Loading Grounding DINO  config=%s  weights=%s  device=%s",
            self._config_path,
            self._weights_path,
            self._device,
        )
        self._model = load_model(
            self._config_path,
            self._weights_path,
            device=self._device,
        )
        logger.info("Grounding DINO loaded successfully")

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    # ── inference ────────────────────────────────────────────────────────

    def detect(
        self,
        image_rgb: np.ndarray,
        text_prompt: str,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
    ) -> List[Detection]:
        """Run open-vocabulary detection on *image_rgb*.

        Args:
            image_rgb: (H, W, 3) uint8 RGB image.
            text_prompt: Free-form text prompt (e.g. "red cube . blue ball").
                         Separate categories with " . " for multi-class detection.
            box_threshold: Confidence threshold for box filtering.
            text_threshold: Confidence threshold for text–box matching.

        Returns:
            List of Detection dataclasses, sorted by descending confidence.
        """
        from groundingdino.util.inference import predict

        box_thr = box_threshold if box_threshold is not None else DETECTION_CONFIDENCE_THRESHOLD
        text_thr = text_threshold if text_threshold is not None else DETECTION_CONFIDENCE_THRESHOLD

        pil_image = Image.fromarray(image_rgb)

        boxes, logits, phrases = predict(
            model=self.model,
            image=pil_image,
            caption=text_prompt,
            box_threshold=box_thr,
            text_threshold=text_thr,
            device=self._device,
        )

        h, w = image_rgb.shape[:2]
        detections: List[Detection] = []
        for box_cxcywh, score, phrase in zip(boxes, logits, phrases):
            cx, cy, bw, bh = box_cxcywh.tolist()
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            detections.append(
                Detection(
                    label=phrase.strip(),
                    bbox_xyxy=[x1, y1, x2, y2],
                    confidence=float(score),
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)

        # simple class-aware NMS
        detections = self._nms(detections, iou_threshold=DETECTION_NMS_THRESHOLD)

        logger.info(
            "Grounding DINO detected %d objects for prompt '%s'",
            len(detections),
            text_prompt,
        )
        return detections

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _iou(a: List[float], b: List[float]) -> float:
        """Compute IoU between two [x1, y1, x2, y2] boxes."""
        xa = max(a[0], b[0])
        ya = max(a[1], b[1])
        xb = min(a[2], b[2])
        yb = min(a[3], b[3])
        inter = max(0.0, xb - xa) * max(0.0, yb - ya)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @classmethod
    def _nms(cls, dets: List[Detection], iou_threshold: float) -> List[Detection]:
        """Greedy NMS across all labels."""
        keep: List[Detection] = []
        for d in dets:
            if all(cls._iou(d.bbox_xyxy, k.bbox_xyxy) < iou_threshold for k in keep):
                keep.append(d)
        return keep

    @staticmethod
    def extract_nouns_simple(task: str) -> str:
        """Heuristic noun extraction to build a detection prompt from a task string.

        Falls back to returning the full task if no nouns are found.
        Common stop-words and verbs are stripped.
        """
        stop = {
            "a", "an", "the", "and", "or", "to", "from", "of", "in", "on",
            "at", "it", "is", "are", "was", "were", "be", "been", "being",
            "pick", "place", "put", "move", "grab", "drop", "push", "pull",
            "take", "bring", "lift", "lower", "rotate", "turn", "flip",
            "up", "down", "left", "right", "forward", "backward", "back",
            "into", "onto", "over", "under", "above", "below", "beside",
            "next", "near", "far", "away", "here", "there",
            "then", "now", "please", "can", "you", "i", "my", "me",
        }
        words = task.lower().replace(",", " ").replace(".", " ").split()
        nouns = [w for w in words if w not in stop and len(w) > 1]
        if not nouns:
            return task.lower()
        return " . ".join(nouns)
