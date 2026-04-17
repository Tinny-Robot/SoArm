"""
Orbbec Astra Pro HD overhead camera driver.

Uses OpenCV (V4L2) for reliable RGB capture and optionally OpenNI2 for
the depth stream.  If the Orbbec OpenNI2 driver is not installed the
camera falls back to RGB-only mode with estimated depth (a fixed table
height is assumed for world-XYZ back-projection).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from soarm_gemini.config import (
    DEPTH_CX,
    DEPTH_CY,
    DEPTH_FX,
    DEPTH_FY,
    DEPTH_SCALE,
    OPENNI2_LIB_DIR,
    OVERHEAD_CAMERA_FPS,
    OVERHEAD_CAMERA_HEIGHT,
    OVERHEAD_CAMERA_INDEX,
    OVERHEAD_CAMERA_WIDTH,
    OVERHEAD_R_CAM_TO_BASE,
    OVERHEAD_T_CAM_TO_BASE,
)

logger = logging.getLogger(__name__)

# Assumed camera-to-table distance when depth is unavailable (metres)
_DEFAULT_TABLE_DEPTH_M: float = 0.45


@dataclass
class DepthFrame:
    """Container for a synchronised RGB + depth pair."""
    rgb: np.ndarray          # (H, W, 3) uint8, RGB order
    depth_raw: np.ndarray    # (H, W)    uint16, raw sensor values (zeros if no depth)
    depth_metres: np.ndarray # (H, W)    float64, metric depth in metres


class OverheadCamera:
    """Orbbec Astra Pro driver: OpenCV for RGB, OpenNI2 for depth (if available)."""

    def __init__(self) -> None:
        self._cap: Optional[cv2.VideoCapture] = None
        self._depth_available: bool = False
        self._openni_ctx_init: bool = False
        self._depth_stream = None
        self._openni_device = None

    # ── lifecycle ────────────────────────────────────────────────────────

    def open(self) -> None:
        """Open the RGB stream (V4L2) and attempt to open the depth stream (OpenNI2)."""
        # RGB via OpenCV / V4L2
        self._cap = cv2.VideoCapture(OVERHEAD_CAMERA_INDEX, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Overhead camera at /dev/video{OVERHEAD_CAMERA_INDEX} failed to open"
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, OVERHEAD_CAMERA_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, OVERHEAD_CAMERA_HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS, OVERHEAD_CAMERA_FPS)
        logger.info(
            "Overhead RGB opened: /dev/video%d  %dx%d @%dfps",
            OVERHEAD_CAMERA_INDEX,
            OVERHEAD_CAMERA_WIDTH,
            OVERHEAD_CAMERA_HEIGHT,
            OVERHEAD_CAMERA_FPS,
        )

        # Depth via OpenNI2 (best-effort)
        self._try_open_depth()

    def _try_open_depth(self) -> None:
        """Attempt to initialise OpenNI2 depth. Logs a warning on failure."""
        try:
            from openni import openni2

            if not self._openni_ctx_init:
                openni2.initialize(OPENNI2_LIB_DIR)
                self._openni_ctx_init = True

            self._openni_device = openni2.Device.open_any()
            self._depth_stream = self._openni_device.create_depth_stream()
            self._depth_stream.set_video_mode(
                openni2.VideoMode(
                    pixelFormat=openni2.PIXEL_FORMAT_DEPTH_1_MM,
                    resolutionX=OVERHEAD_CAMERA_WIDTH,
                    resolutionY=OVERHEAD_CAMERA_HEIGHT,
                    fps=OVERHEAD_CAMERA_FPS,
                )
            )
            self._depth_stream.start()
            self._depth_available = True
            logger.info("Overhead depth stream opened via OpenNI2")
        except Exception as exc:
            self._depth_available = False
            logger.warning(
                "OpenNI2 depth unavailable (%s) — running in RGB-only mode "
                "with estimated depth at %.2f m",
                exc,
                _DEFAULT_TABLE_DEPTH_M,
            )

    def close(self) -> None:
        """Release all resources."""
        if self._depth_stream is not None:
            try:
                self._depth_stream.stop()
            except Exception:
                pass
            self._depth_stream = None
        if self._openni_device is not None:
            try:
                self._openni_device.close()
            except Exception:
                pass
            self._openni_device = None
        if self._openni_ctx_init:
            try:
                from openni import openni2
                openni2.unload()
            except Exception:
                pass
            self._openni_ctx_init = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("Overhead camera closed")

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def has_depth(self) -> bool:
        return self._depth_available

    # ── frame acquisition ────────────────────────────────────────────────

    def grab_frames(self) -> DepthFrame:
        """Return an aligned RGB + depth frame pair.

        When real depth is unavailable, a synthetic depth map filled with
        the assumed camera-to-table distance is returned instead.
        """
        if not self.is_open:
            raise RuntimeError("Overhead camera is not open — call .open() first")

        ret, bgr = self._cap.read()
        if not ret or bgr is None:
            raise RuntimeError("Overhead RGB frame grab failed")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if self._depth_available and self._depth_stream is not None:
            depth_data = self._depth_stream.read_frame()
            depth_raw = np.frombuffer(
                depth_data.get_buffer_as_uint16(), dtype=np.uint16
            ).reshape((OVERHEAD_CAMERA_HEIGHT, OVERHEAD_CAMERA_WIDTH))
            depth_metres = depth_raw.astype(np.float64) * DEPTH_SCALE
        else:
            h, w = rgb.shape[:2]
            depth_raw = np.full((h, w), int(_DEFAULT_TABLE_DEPTH_M / DEPTH_SCALE), dtype=np.uint16)
            depth_metres = np.full((h, w), _DEFAULT_TABLE_DEPTH_M, dtype=np.float64)

        return DepthFrame(rgb=rgb, depth_raw=depth_raw.copy(), depth_metres=depth_metres)

    def grab_rgb_bgr(self) -> np.ndarray:
        """Convenience: return only the overhead frame as BGR for OpenCV display."""
        if not self.is_open:
            raise RuntimeError("Overhead camera is not open — call .open() first")
        ret, bgr = self._cap.read()
        if not ret or bgr is None:
            raise RuntimeError("Overhead RGB frame grab failed")
        return bgr

    # ── 3-D helpers ──────────────────────────────────────────────────────

    @staticmethod
    def pixel_to_camera_xyz(u: float, v: float, depth_m: float) -> np.ndarray:
        """Back-project a single pixel (u, v) at *depth_m* into the camera frame."""
        x_cam = (u - DEPTH_CX) * depth_m / DEPTH_FX
        y_cam = (v - DEPTH_CY) * depth_m / DEPTH_FY
        z_cam = depth_m
        return np.array([x_cam, y_cam, z_cam], dtype=np.float64)

    @staticmethod
    def camera_xyz_to_world(point_cam: np.ndarray) -> np.ndarray:
        """Transform a point from the overhead camera frame into the robot base frame."""
        return OVERHEAD_R_CAM_TO_BASE @ point_cam + OVERHEAD_T_CAM_TO_BASE

    def world_xyz_from_pixel(
        self, u: float, v: float, depth_metres: np.ndarray
    ) -> Optional[np.ndarray]:
        """End-to-end: pixel (u,v) + depth map → world XYZ.

        Samples a 5x5 median patch around (u, v) to reduce depth noise.
        """
        h, w = depth_metres.shape
        ui, vi = int(round(u)), int(round(v))
        pad = 2
        u0 = max(0, ui - pad)
        u1 = min(w, ui + pad + 1)
        v0 = max(0, vi - pad)
        v1 = min(h, vi + pad + 1)

        patch = depth_metres[v0:v1, u0:u1]
        valid = patch[patch > 0.0]
        if valid.size == 0:
            return None
        depth_m = float(np.median(valid))

        point_cam = self.pixel_to_camera_xyz(u, v, depth_m)
        return self.camera_xyz_to_world(point_cam)

    def point_cloud(self, depth_metres: np.ndarray) -> np.ndarray:
        """Convert the full depth map to a world-frame point cloud."""
        h, w = depth_metres.shape
        us, vs = np.meshgrid(np.arange(w), np.arange(h))
        mask = depth_metres > 0.0

        d = depth_metres[mask]
        u_flat = us[mask].astype(np.float64)
        v_flat = vs[mask].astype(np.float64)

        x_cam = (u_flat - DEPTH_CX) * d / DEPTH_FX
        y_cam = (v_flat - DEPTH_CY) * d / DEPTH_FY
        z_cam = d

        pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
        pts_world = (OVERHEAD_R_CAM_TO_BASE @ pts_cam.T).T + OVERHEAD_T_CAM_TO_BASE
        return pts_world

    # ── context manager ──────────────────────────────────────────────────

    def __enter__(self) -> "OverheadCamera":
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
