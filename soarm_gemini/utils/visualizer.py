"""
OpenCV debug visualizer.

Overlays detected bounding boxes, world XYZ labels, the last Gemini action
list, and current arm joint state onto the overhead camera frame.
"""

from __future__ import annotations

import logging
import textwrap
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import cv2
import numpy as np

from soarm_gemini.config import (
    VIS_BBOX_THICKNESS,
    VIS_FONT_SCALE,
    VIS_SCALE,
    VIS_WINDOW_NAME,
)

if TYPE_CHECKING:
    from soarm_gemini.planner.gemini_planner import RobotAction
    from soarm_gemini.scene.scene_builder import SceneState
    from soarm_gemini.vision.detector import Detection

logger = logging.getLogger(__name__)

_COLORS: List[Tuple[int, int, int]] = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
]


class DebugVisualizer:
    """Renders a live debug overlay window using OpenCV highgui."""

    def __init__(self, window_name: Optional[str] = None) -> None:
        self._win = window_name or VIS_WINDOW_NAME
        self._open = False

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Create the named window."""
        cv2.namedWindow(self._win, cv2.WINDOW_NORMAL)
        self._open = True

    def stop(self) -> None:
        """Destroy the window."""
        if self._open:
            cv2.destroyWindow(self._win)
            self._open = False

    # ── rendering ────────────────────────────────────────────────────────

    def update(
        self,
        overhead_bgr: np.ndarray,
        detections: Optional[List[Detection]] = None,
        scene: Optional[SceneState] = None,
        actions: Optional[List[RobotAction]] = None,
        extra_text: Optional[str] = None,
    ) -> np.ndarray:
        """Draw all overlays and display the result.

        Args:
            overhead_bgr: The raw BGR overhead frame.
            detections: Grounding DINO detections to draw as bounding boxes.
            scene: Current SceneState (used for arm state display).
            actions: Last Gemini action list.
            extra_text: Arbitrary text to show in the bottom-left.

        Returns:
            The annotated BGR frame (useful for recording).
        """
        canvas = overhead_bgr.copy()

        if detections:
            self._draw_detections(canvas, detections)

        if scene:
            self._draw_arm_state(canvas, scene)
            self._draw_scene_objects(canvas, scene)

        if actions:
            self._draw_actions(canvas, actions)

        if extra_text:
            self._draw_text_block(canvas, extra_text, origin=(10, canvas.shape[0] - 30))

        display = cv2.resize(
            canvas,
            None,
            fx=VIS_SCALE,
            fy=VIS_SCALE,
            interpolation=cv2.INTER_AREA,
        )

        if self._open:
            cv2.imshow(self._win, display)
            cv2.waitKey(1)

        return canvas

    # ── drawing primitives ───────────────────────────────────────────────

    @staticmethod
    def _draw_detections(canvas: np.ndarray, dets: List[Detection]) -> None:
        """Draw bounding boxes and labels for each detection."""
        for idx, det in enumerate(dets):
            color = _COLORS[idx % len(_COLORS)]
            x1, y1, x2, y2 = [int(c) for c in det.bbox_xyxy]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, VIS_BBOX_THICKNESS)

            label = f"{det.label} {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, VIS_FONT_SCALE, 1
            )
            cv2.rectangle(
                canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1
            )
            cv2.putText(
                canvas,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                VIS_FONT_SCALE,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    @staticmethod
    def _draw_scene_objects(canvas: np.ndarray, scene: SceneState) -> None:
        """Annotate detected objects with their world XYZ near the bbox centre."""
        h, w = canvas.shape[:2]
        for obj in scene.objects:
            xyz = obj.world_xyz
            text = f"{obj.label}: ({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})"
            # Place at a heuristic position (top-right quadrant)
            cv2.putText(
                canvas,
                text,
                (w - 450, 30 + scene.objects.index(obj) * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                VIS_FONT_SCALE,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    @staticmethod
    def _draw_arm_state(canvas: np.ndarray, scene: SceneState) -> None:
        """Overlay the current joint state in the top-left corner."""
        arm = scene.arm_state
        lines = [
            f"Joints: {[round(j, 1) for j in arm.joints_deg]}",
            f"Gripper: {arm.gripper}",
            f"EE: ({arm.end_effector_xyz[0]:.3f}, "
            f"{arm.end_effector_xyz[1]:.3f}, "
            f"{arm.end_effector_xyz[2]:.3f})",
        ]
        for i, line in enumerate(lines):
            cv2.putText(
                canvas,
                line,
                (10, 25 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                VIS_FONT_SCALE,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    @staticmethod
    def _draw_actions(canvas: np.ndarray, actions: List[RobotAction]) -> None:
        """Draw the last Gemini action list in the bottom-right."""
        h, w = canvas.shape[:2]
        y = h - 20 * len(actions) - 10
        cv2.putText(
            canvas,
            "Gemini Plan:",
            (w - 420, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            VIS_FONT_SCALE,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )
        for i, act in enumerate(actions):
            summary = _action_summary(act)
            cv2.putText(
                canvas,
                f"  {i+1}. {summary}",
                (w - 420, y + 18 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                VIS_FONT_SCALE - 0.05,
                (0, 200, 255),
                1,
                cv2.LINE_AA,
            )

    @staticmethod
    def _draw_text_block(
        canvas: np.ndarray,
        text: str,
        origin: Tuple[int, int],
        color: Tuple[int, int, int] = (200, 200, 200),
    ) -> None:
        """Draw a multi-line text block."""
        x, y = origin
        for line in textwrap.wrap(text, width=60):
            cv2.putText(
                canvas,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                VIS_FONT_SCALE,
                color,
                1,
                cv2.LINE_AA,
            )
            y += 20

    # ── context manager ──────────────────────────────────────────────────

    def __enter__(self) -> "DebugVisualizer":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()


# ── helpers ──────────────────────────────────────────────────────────────────

def _action_summary(act: RobotAction) -> str:
    """One-line summary of a RobotAction for the overlay."""
    if act.action == "move" and act.target_xyz:
        xyz = act.target_xyz
        return f"move → ({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})"
    if act.action == "lift" and act.delta_z is not None:
        return f"lift Δz={act.delta_z:.3f}"
    if act.action == "grip":
        return "grip (close)"
    if act.action == "release":
        return "release (open)"
    if act.action == "home":
        return "home"
    if act.action == "abort":
        return f"ABORT: {act.reason or '?'}"
    return act.action
