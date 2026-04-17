"""
Analytic inverse-kinematics solver for the SO-ARM101 6-DOF arm.

Converts a desired end-effector world position [x, y, z] (metres) into
joint angles (degrees) using the DH parameters defined in config.py.

For poses that cannot be reached analytically the solver falls back to
ikpy's numerical IK.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import numpy as np

from soarm_gemini.config import (
    ARM_DH,
    HOME_POSITION_DEG,
    JOINT_LIMITS_DEG,
    NUM_JOINTS,
    WORKSPACE_MAX_XYZ,
    WORKSPACE_MIN_XYZ,
)

logger = logging.getLogger(__name__)

# Link lengths (metres) extracted from DH table for the simplified planar model
_L1 = ARM_DH[0].d       # base height
_L2 = ARM_DH[1].a       # upper arm
_L3 = ARM_DH[2].a       # forearm
_L4 = ARM_DH[4].d       # wrist-to-tip


class IKSolver:
    """Analytic + fallback numerical IK for the SO-ARM101."""

    def __init__(self) -> None:
        self._ikpy_chain = None  # lazy-loaded

    # ── public API ───────────────────────────────────────────────────────

    def solve(
        self,
        target_xyz: List[float],
        current_joints_deg: Optional[List[float]] = None,
    ) -> List[float]:
        """Compute joint angles (degrees) to reach *target_xyz*.

        Args:
            target_xyz: Desired end-effector position [x, y, z] in metres.
            current_joints_deg: Hint for the numerical fallback (seed).

        Returns:
            List of 5 joint angles (degrees) for joints 1–5.
            Joint 6 (gripper) is NOT included.

        Raises:
            ValueError: If the target is outside the reachable workspace.
        """
        x, y, z = target_xyz
        self._check_workspace(x, y, z)

        try:
            angles = self._analytic_ik(x, y, z)
            logger.info(
                "Analytic IK: xyz=%s → joints_deg=%s",
                [round(c, 4) for c in target_xyz],
                [round(a, 2) for a in angles],
            )
            return angles
        except Exception as exc:
            logger.warning("Analytic IK failed (%s), falling back to numerical", exc)

        return self._numerical_ik(target_xyz, current_joints_deg)

    def forward_kinematics(self, joints_deg: List[float]) -> np.ndarray:
        """Compute the end-effector position for a given set of joint angles.

        Args:
            joints_deg: 5 joint angles (degrees), joints 1–5.

        Returns:
            (3,) array [x, y, z] in metres.
        """
        chain = self._get_ikpy_chain()
        rads = [0.0] + [math.radians(a) for a in joints_deg] + [0.0]
        fk = chain.forward_kinematics(rads)
        return np.array(fk[:3, 3], dtype=np.float64)

    # ── analytic solver ──────────────────────────────────────────────────

    @staticmethod
    def _analytic_ik(x: float, y: float, z: float) -> List[float]:
        """3-DOF planar analytic IK (base rotation + 2R elbow).

        Joints 4 & 5 are set to keep the end-effector pointing downward.

        Returns:
            [theta1, theta2, theta3, theta4, theta5] in degrees.
        """
        # Joint 1: base rotation
        theta1 = math.atan2(y, x)

        # Horizontal distance in the arm plane
        r = math.sqrt(x * x + y * y)
        # Vertical offset from the shoulder joint
        z_eff = z - _L1
        # Distance from shoulder to wrist centre (subtract wrist length)
        r_w = r - 0.0  # wrist offset in plane is negligible for SO-ARM101
        z_w = z_eff - _L4

        dist_sq = r_w * r_w + z_w * z_w
        dist = math.sqrt(dist_sq)

        # Reachability check
        if dist > (_L2 + _L3) or dist < abs(_L2 - _L3):
            raise ValueError(
                f"Target (r={r:.4f}, z_w={z_w:.4f}) out of 2R reach "
                f"(L2={_L2}, L3={_L3}, dist={dist:.4f})"
            )

        # Elbow angle (joint 3) via cosine rule
        cos_theta3 = (dist_sq - _L2 * _L2 - _L3 * _L3) / (2.0 * _L2 * _L3)
        cos_theta3 = max(-1.0, min(1.0, cos_theta3))
        theta3 = -math.acos(cos_theta3)  # elbow-up convention

        # Shoulder angle (joint 2)
        alpha = math.atan2(z_w, r_w)
        beta = math.atan2(_L3 * math.sin(theta3), _L2 + _L3 * math.cos(theta3))
        theta2 = alpha + beta

        # Wrist pitch (joint 4): keep tool pointing straight down
        theta4 = -(theta2 + theta3)
        # Wrist roll (joint 5): keep level
        theta5 = 0.0

        angles_deg = [
            math.degrees(theta1),
            math.degrees(theta2),
            math.degrees(theta3),
            math.degrees(theta4),
            math.degrees(theta5),
        ]

        # Clamp to joint limits
        for i, (lo, hi) in enumerate(
            [JOINT_LIMITS_DEG[j] for j in range(1, 6)]
        ):
            angles_deg[i] = max(lo, min(hi, angles_deg[i]))

        return [round(a, 2) for a in angles_deg]

    # ── numerical fallback (ikpy) ────────────────────────────────────────

    def _numerical_ik(
        self,
        target_xyz: List[float],
        seed_deg: Optional[List[float]] = None,
    ) -> List[float]:
        """Use ikpy to solve IK numerically when the analytic solver fails."""
        chain = self._get_ikpy_chain()

        target = np.eye(4)
        target[:3, 3] = target_xyz
        # Orientation: end-effector pointing straight down
        target[:3, :3] = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ])

        if seed_deg is not None:
            seed = [0.0] + [math.radians(a) for a in seed_deg[:5]] + [0.0]
        else:
            seed = [0.0] + [math.radians(a) for a in HOME_POSITION_DEG[:5]] + [0.0]

        result = chain.inverse_kinematics(target, initial_position=seed)
        angles_deg = [round(math.degrees(result[i + 1]), 2) for i in range(5)]

        # Clamp
        for i, (lo, hi) in enumerate(
            [JOINT_LIMITS_DEG[j] for j in range(1, 6)]
        ):
            angles_deg[i] = max(lo, min(hi, angles_deg[i]))

        logger.info(
            "Numerical IK: xyz=%s → joints_deg=%s",
            [round(c, 4) for c in target_xyz],
            angles_deg,
        )
        return angles_deg

    def _get_ikpy_chain(self):
        """Lazy-build an ikpy chain from the DH parameters."""
        if self._ikpy_chain is not None:
            return self._ikpy_chain

        from ikpy.chain import Chain
        from ikpy.link import URDFLink

        _zero = np.array([0.0, 0.0, 0.0])

        links = [
            URDFLink(name="base",
                     origin_translation=_zero, origin_orientation=_zero,
                     joint_type="fixed"),
            URDFLink(name="joint_1",
                     bounds=tuple(math.radians(x) for x in JOINT_LIMITS_DEG[1]),
                     origin_translation=np.array([0, 0, _L1]),
                     origin_orientation=_zero,
                     rotation=np.array([0, 0, 1])),
            URDFLink(name="joint_2",
                     bounds=tuple(math.radians(x) for x in JOINT_LIMITS_DEG[2]),
                     origin_translation=np.array([_L2, 0, 0]),
                     origin_orientation=_zero,
                     rotation=np.array([0, 1, 0])),
            URDFLink(name="joint_3",
                     bounds=tuple(math.radians(x) for x in JOINT_LIMITS_DEG[3]),
                     origin_translation=np.array([_L3, 0, 0]),
                     origin_orientation=_zero,
                     rotation=np.array([0, 1, 0])),
            URDFLink(name="joint_4",
                     bounds=tuple(math.radians(x) for x in JOINT_LIMITS_DEG[4]),
                     origin_translation=_zero,
                     origin_orientation=_zero,
                     rotation=np.array([0, 1, 0])),
            URDFLink(name="joint_5",
                     bounds=tuple(math.radians(x) for x in JOINT_LIMITS_DEG[5]),
                     origin_translation=np.array([0, 0, _L4]),
                     origin_orientation=_zero,
                     rotation=np.array([0, 0, 1])),
            URDFLink(name="tip",
                     origin_translation=_zero, origin_orientation=_zero,
                     joint_type="fixed"),
        ]
        active_mask = [False] + [True] * 5 + [False]
        self._ikpy_chain = Chain(links, name="so_arm101",
                                 active_links_mask=active_mask)
        return self._ikpy_chain

    # ── validation ───────────────────────────────────────────────────────

    @staticmethod
    def _check_workspace(x: float, y: float, z: float) -> None:
        mn = WORKSPACE_MIN_XYZ
        mx = WORKSPACE_MAX_XYZ
        if not (mn[0] <= x <= mx[0] and mn[1] <= y <= mx[1] and mn[2] <= z <= mx[2]):
            raise ValueError(
                f"Target ({x:.4f}, {y:.4f}, {z:.4f}) outside workspace "
                f"[{mn}..{mx}]"
            )
