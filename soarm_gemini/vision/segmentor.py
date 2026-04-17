"""
SAM 2 (Segment Anything Model 2) wrapper for pixel-level segmentation.

Given an RGB image and a bounding box (from Grounding DINO), produces a
binary mask and computes the physical area in cm² using depth information.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch

from soarm_gemini.config import (
    DEPTH_SCALE,
    SAM2_CHECKPOINT,
    SAM2_MODEL_CFG,
)

logger = logging.getLogger(__name__)


@dataclass
class SegmentationResult:
    """Segmentation output for one object."""
    mask: np.ndarray              # (H, W) bool
    area_pixels: int
    area_cm2: Optional[float]     # None if depth wasn't supplied
    bbox_xyxy: List[float]


class SAM2Segmentor:
    """Lazy-loaded SAM 2 inference wrapper.

    Loads the model on first call to :meth:`segment`.
    """

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        model_cfg: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self._checkpoint = checkpoint or SAM2_CHECKPOINT
        self._model_cfg = model_cfg or SAM2_MODEL_CFG
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._predictor = None

    # ── lazy loading ─────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Load SAM 2 checkpoint."""
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        logger.info(
            "Loading SAM 2  checkpoint=%s  config=%s  device=%s",
            self._checkpoint,
            self._model_cfg,
            self._device,
        )
        sam_model = build_sam2(self._model_cfg, self._checkpoint, device=self._device)
        self._predictor = SAM2ImagePredictor(sam_model)
        logger.info("SAM 2 loaded successfully")

    @property
    def predictor(self):
        if self._predictor is None:
            self._load_model()
        return self._predictor

    # ── inference ────────────────────────────────────────────────────────

    def segment(
        self,
        image_rgb: np.ndarray,
        bbox_xyxy: List[float],
        depth_metres: Optional[np.ndarray] = None,
    ) -> SegmentationResult:
        """Segment a single object inside *bbox_xyxy*.

        Args:
            image_rgb: (H, W, 3) uint8 RGB image.
            bbox_xyxy: [x1, y1, x2, y2] bounding box in pixel coords.
            depth_metres: Optional (H, W) depth map in metres. When provided
                          the physical area of the mask is estimated.

        Returns:
            SegmentationResult with binary mask and area.
        """
        pred = self.predictor
        pred.set_image(image_rgb)

        box_np = np.array(bbox_xyxy, dtype=np.float32)[None, :]  # (1, 4)
        masks, scores, _ = pred.predict(
            box=box_np,
            multimask_output=True,
        )

        # Take the mask with the highest predicted IoU
        best_idx = int(np.argmax(scores))
        mask = masks[best_idx].astype(bool)  # (H, W)
        area_pixels = int(mask.sum())

        area_cm2: Optional[float] = None
        if depth_metres is not None:
            area_cm2 = self._estimate_area_cm2(mask, depth_metres)

        logger.info(
            "SAM 2 segmented bbox=%s  area_px=%d  area_cm2=%s",
            bbox_xyxy,
            area_pixels,
            f"{area_cm2:.2f}" if area_cm2 is not None else "N/A",
        )
        return SegmentationResult(
            mask=mask,
            area_pixels=area_pixels,
            area_cm2=area_cm2,
            bbox_xyxy=bbox_xyxy,
        )

    def segment_batch(
        self,
        image_rgb: np.ndarray,
        bboxes: List[List[float]],
        depth_metres: Optional[np.ndarray] = None,
    ) -> List[SegmentationResult]:
        """Segment multiple bounding boxes in a single image.

        Args:
            image_rgb: (H, W, 3) uint8 RGB image.
            bboxes: List of [x1, y1, x2, y2] boxes.
            depth_metres: Optional depth map for area estimation.

        Returns:
            List of SegmentationResult, one per bbox.
        """
        return [
            self.segment(image_rgb, bbox, depth_metres)
            for bbox in bboxes
        ]

    # ── area estimation ──────────────────────────────────────────────────

    @staticmethod
    def _estimate_area_cm2(
        mask: np.ndarray,
        depth_metres: np.ndarray,
    ) -> Optional[float]:
        """Approximate the real-world area of a mask using the depth map.

        Each pixel's physical footprint is computed from its depth and the
        camera intrinsics (assuming locally planar surface), then summed.

        Returns area in cm², or None if insufficient depth data.
        """
        from soarm_gemini.config import DEPTH_FX, DEPTH_FY

        valid = mask & (depth_metres > 0.0)
        if valid.sum() < 10:
            return None

        depths = depth_metres[valid]
        # Each pixel covers (depth / fx) × (depth / fy) m² on the object surface
        pixel_areas_m2 = (depths / DEPTH_FX) * (depths / DEPTH_FY)
        total_m2 = float(pixel_areas_m2.sum())
        return total_m2 * 1e4  # m² → cm²
