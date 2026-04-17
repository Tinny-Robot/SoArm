"""
USB UVC wrist camera driver.

Provides frame capture and base64 encoding for the eye-in-hand camera
mounted on the SO-ARM101 end-effector.
"""

from __future__ import annotations

import base64
import logging
from typing import Optional

import cv2
import numpy as np

from soarm_gemini.config import (
    WRIST_CAMERA_FPS,
    WRIST_CAMERA_HEIGHT,
    WRIST_CAMERA_INDEX,
    WRIST_CAMERA_WIDTH,
)

logger = logging.getLogger(__name__)


class WristCamera:
    """Thin wrapper around an OpenCV VideoCapture for the wrist-mounted UVC camera."""

    def __init__(self, device_index: Optional[int] = None) -> None:
        self._index: int = device_index if device_index is not None else WRIST_CAMERA_INDEX
        self._cap: Optional[cv2.VideoCapture] = None

    # ── lifecycle ────────────────────────────────────────────────────────

    def open(self) -> None:
        """Open the camera device and apply capture settings."""
        self._cap = cv2.VideoCapture(self._index, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f"Wrist camera at index {self._index} failed to open")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, WRIST_CAMERA_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WRIST_CAMERA_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS, WRIST_CAMERA_FPS)
        logger.info(
            "Wrist camera opened: index=%d  %dx%d @%dfps",
            self._index,
            WRIST_CAMERA_WIDTH,
            WRIST_CAMERA_HEIGHT,
            WRIST_CAMERA_FPS,
        )

    def close(self) -> None:
        """Release the camera device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Wrist camera closed")

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    # ── frame acquisition ────────────────────────────────────────────────

    def grab_frame(self) -> np.ndarray:
        """Return a BGR frame from the wrist camera.

        Raises:
            RuntimeError: If the camera is not open or the read fails.
        """
        if not self.is_open:
            raise RuntimeError("Wrist camera is not open — call .open() first")
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise RuntimeError("Wrist camera frame grab failed")
        return frame

    def grab_rgb(self) -> np.ndarray:
        """Return an RGB frame (convenience for vision models that expect RGB)."""
        return cv2.cvtColor(self.grab_frame(), cv2.COLOR_BGR2RGB)

    # ── encoding helpers ─────────────────────────────────────────────────

    @staticmethod
    def encode_frame_base64(frame: np.ndarray, fmt: str = ".jpg") -> str:
        """Encode a BGR/RGB frame as a base64 JPEG (or PNG) string.

        Args:
            frame: HxWx3 uint8 image.
            fmt: OpenCV imencode format string ('.jpg' or '.png').

        Returns:
            Base64-encoded image string.
        """
        success, buf = cv2.imencode(fmt, frame)
        if not success:
            raise RuntimeError("Failed to encode frame")
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    # ── context manager ──────────────────────────────────────────────────

    def __enter__(self) -> "WristCamera":
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
