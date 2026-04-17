"""
Scene builder: fuses overhead RGB + depth detections with the wrist frame
into a structured scene_state dictionary suitable for the Gemini planner.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from soarm_gemini.cameras.overhead_cam import DepthFrame, OverheadCamera
from soarm_gemini.cameras.wrist_cam import WristCamera
from soarm_gemini.vision.detector import Detection, GroundingDINODetector
from soarm_gemini.vision.segmentor import SAM2Segmentor, SegmentationResult

logger = logging.getLogger(__name__)


# ── Structured data ──────────────────────────────────────────────────────────

@dataclass
class SceneObject:
    """Represents one detected object in world coordinates."""
    label: str
    world_xyz: List[float]            # [x, y, z] in metres
    confidence: float
    mask_area_cm2: Optional[float]
    visible_in: List[str] = field(default_factory=list)  # ["overhead", "wrist"]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ArmState:
    """Current state of the robot arm."""
    joints_deg: List[float]
    gripper: str                      # "open" or "close"
    end_effector_xyz: List[float]     # [x, y, z] in metres

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SceneState:
    """Full scene description sent to the Gemini planner."""
    objects: List[SceneObject]
    arm_state: ArmState
    task: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objects": [o.to_dict() for o in self.objects],
            "arm_state": self.arm_state.to_dict(),
            "task": self.task,
        }


# ── Builder ──────────────────────────────────────────────────────────────────

class SceneBuilder:
    """Builds a SceneState by fusing overhead + wrist camera observations."""

    def __init__(
        self,
        overhead_cam: OverheadCamera,
        wrist_cam: WristCamera,
        detector: GroundingDINODetector,
        segmentor: SAM2Segmentor,
    ) -> None:
        self._overhead = overhead_cam
        self._wrist = wrist_cam
        self._detector = detector
        self._segmentor = segmentor

    def build(
        self,
        task: str,
        arm_joints_deg: List[float],
        gripper_state: str,
        ee_xyz: List[float],
        detection_prompt: str,
    ) -> SceneState:
        """Capture frames, run perception, and build the scene state.

        Args:
            task: Natural-language task string.
            arm_joints_deg: Current joint angles (degrees) read from the robot.
            gripper_state: "open" or "close".
            ee_xyz: Current end-effector world position [x, y, z] (metres).
            detection_prompt: Grounding DINO text prompt (noun phrases separated
                              by " . ").

        Returns:
            Populated SceneState ready for the Gemini planner.
        """
        # 1. Grab frames
        overhead_df = self._overhead.grab_frames()
        wrist_bgr = self._wrist.grab_frame()
        wrist_rgb = self._wrist.grab_rgb()

        # 2. Detect objects in overhead image
        overhead_dets = self._detector.detect(overhead_df.rgb, detection_prompt)
        logger.info("Overhead detections: %d", len(overhead_dets))

        # 3. Detect objects in wrist image (for visibility tagging)
        wrist_dets = self._detector.detect(wrist_rgb, detection_prompt)
        wrist_labels = {d.label.lower() for d in wrist_dets}
        logger.info("Wrist detections: %d", len(wrist_dets))

        # 4. For each overhead detection → world XYZ + SAM mask
        scene_objects: List[SceneObject] = []
        for det in overhead_dets:
            obj = self._process_detection(
                det,
                overhead_df,
                wrist_labels,
            )
            if obj is not None:
                scene_objects.append(obj)

        arm_state = ArmState(
            joints_deg=arm_joints_deg,
            gripper=gripper_state,
            end_effector_xyz=ee_xyz,
        )

        scene = SceneState(
            objects=scene_objects,
            arm_state=arm_state,
            task=task,
        )
        logger.info(
            "Scene built: %d objects, task='%s'",
            len(scene_objects),
            task,
        )
        return scene

    # ── internal ─────────────────────────────────────────────────────────

    def _process_detection(
        self,
        det: Detection,
        overhead_df: DepthFrame,
        wrist_labels: set,
    ) -> Optional[SceneObject]:
        """Convert a single overhead Detection into a SceneObject with world XYZ."""
        cx, cy = det.centre_uv

        # World XYZ from depth
        world_xyz = self._overhead.world_xyz_from_pixel(
            cx, cy, overhead_df.depth_metres
        )
        if world_xyz is None:
            logger.warning(
                "No valid depth for detection '%s' at uv=(%.0f, %.0f) — skipping",
                det.label,
                cx,
                cy,
            )
            return None

        # SAM2 mask + area
        seg: SegmentationResult = self._segmentor.segment(
            overhead_df.rgb,
            det.bbox_xyxy,
            overhead_df.depth_metres,
        )

        visible_in = ["overhead"]
        if det.label.lower() in wrist_labels:
            visible_in.append("wrist")

        return SceneObject(
            label=det.label,
            world_xyz=[round(float(c), 4) for c in world_xyz],
            confidence=round(det.confidence, 3),
            mask_area_cm2=round(seg.area_cm2, 2) if seg.area_cm2 is not None else None,
            visible_in=visible_in,
        )

    # ── convenience for re-checking success ──────────────────────────────

    def check_object_position(
        self,
        label: str,
        expected_xyz: List[float],
        tolerance_m: float = 0.03,
        detection_prompt: Optional[str] = None,
    ) -> bool:
        """Re-grab overhead frame and verify *label* is near *expected_xyz*.

        Args:
            label: Object label to look for.
            expected_xyz: Expected [x, y, z] in metres.
            tolerance_m: Euclidean distance tolerance.
            detection_prompt: Override detection prompt; defaults to the label.

        Returns:
            True if the object is found within tolerance.
        """
        prompt = detection_prompt or label
        overhead_df = self._overhead.grab_frames()
        dets = self._detector.detect(overhead_df.rgb, prompt)

        for det in dets:
            if det.label.lower() != label.lower():
                continue
            cx, cy = det.centre_uv
            xyz = self._overhead.world_xyz_from_pixel(cx, cy, overhead_df.depth_metres)
            if xyz is None:
                continue
            dist = float(np.linalg.norm(np.array(xyz) - np.array(expected_xyz)))
            if dist <= tolerance_m:
                logger.info(
                    "Object '%s' found at xyz=%s (expected %s, dist=%.4f m) — success",
                    label,
                    [round(float(c), 4) for c in xyz],
                    expected_xyz,
                    dist,
                )
                return True

        logger.info(
            "Object '%s' not found near expected xyz=%s within %.3f m",
            label,
            expected_xyz,
            tolerance_m,
        )
        return False
