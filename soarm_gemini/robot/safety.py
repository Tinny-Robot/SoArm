"""
Safety layer for the SO-ARM101.

Provides joint-limit clamping, workspace boundary enforcement,
collision pre-checking, and emergency-stop logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from soarm_gemini.config import (
    JOINT_LIMITS_DEG,
    MAX_JOINT_VELOCITY_DEG_S,
    MIN_Z_METRES,
    NUM_JOINTS,
    POSITION_ERROR_THRESHOLD_DEG,
    WORKSPACE_MAX_XYZ,
    WORKSPACE_MIN_XYZ,
)
from soarm_gemini.planner.gemini_planner import RobotAction

logger = logging.getLogger(__name__)


@dataclass
class SafetyVerdict:
    """Result of a safety check on a proposed action or trajectory."""
    safe: bool
    reason: str = ""
    clamped_angles: Optional[List[float]] = None


class SafetyChecker:
    """Validates actions and joint commands before they reach the hardware."""

    # ── action-level checks ──────────────────────────────────────────────

    def validate_action(self, action: RobotAction) -> SafetyVerdict:
        """Check a single RobotAction for obvious violations.

        Returns:
            SafetyVerdict with ``safe=True`` if the action can proceed.
        """
        if action.action == "abort":
            return SafetyVerdict(safe=True, reason="abort is always allowed")

        if action.action == "home":
            return SafetyVerdict(safe=True, reason="home is always allowed")

        if action.action in ("grip", "release"):
            return SafetyVerdict(safe=True)

        if action.action == "move" and action.target_xyz is not None:
            return self._check_target_xyz(action.target_xyz)

        if action.action == "lift" and action.delta_z is not None:
            if action.delta_z < -0.15 or action.delta_z > 0.20:
                return SafetyVerdict(
                    safe=False,
                    reason=f"delta_z={action.delta_z} out of safe range [-0.15, 0.20]",
                )
            return SafetyVerdict(safe=True)

        return SafetyVerdict(safe=True)

    def validate_action_list(self, actions: List[RobotAction]) -> SafetyVerdict:
        """Validate an entire action sequence from the planner.

        Returns:
            SafetyVerdict that is ``safe=False`` on the first violating action.
        """
        for idx, action in enumerate(actions):
            v = self.validate_action(action)
            if not v.safe:
                msg = f"Action #{idx} ({action.action}) rejected: {v.reason}"
                logger.warning(msg)
                return SafetyVerdict(safe=False, reason=msg)
        return SafetyVerdict(safe=True, reason="all actions passed")

    # ── joint-level checks ───────────────────────────────────────────────

    @staticmethod
    def clamp_joints(angles_deg: List[float]) -> Tuple[List[float], bool]:
        """Clamp joint angles to their configured limits.

        Args:
            angles_deg: Proposed joint positions (degrees), length 5 or 6.

        Returns:
            (clamped_angles, was_clamped) — the clamped list and whether any
            value was modified.
        """
        clamped = list(angles_deg)
        was_clamped = False
        for i, a in enumerate(clamped):
            joint_id = i + 1
            lo, hi = JOINT_LIMITS_DEG.get(joint_id, (-180.0, 180.0))
            if a < lo:
                clamped[i] = lo
                was_clamped = True
            elif a > hi:
                clamped[i] = hi
                was_clamped = True
        if was_clamped:
            logger.warning(
                "Joint angles clamped: %s → %s",
                [round(a, 2) for a in angles_deg],
                [round(a, 2) for a in clamped],
            )
        return clamped, was_clamped

    @staticmethod
    def check_velocity(
        current_deg: List[float],
        target_deg: List[float],
        dt_s: float = 1.0,
    ) -> SafetyVerdict:
        """Reject a move that would exceed the max joint velocity.

        Args:
            current_deg: Current joint angles (degrees).
            target_deg: Target joint angles (degrees).
            dt_s: Expected execution time in seconds.

        Returns:
            SafetyVerdict.
        """
        for i, (c, t) in enumerate(zip(current_deg, target_deg)):
            vel = abs(t - c) / dt_s
            if vel > MAX_JOINT_VELOCITY_DEG_S:
                return SafetyVerdict(
                    safe=False,
                    reason=(
                        f"Joint {i+1} velocity {vel:.1f}°/s exceeds max "
                        f"{MAX_JOINT_VELOCITY_DEG_S}°/s"
                    ),
                )
        return SafetyVerdict(safe=True)

    def check_position_error(
        self,
        expected_deg: List[float],
        actual_deg: List[float],
    ) -> SafetyVerdict:
        """Compare expected vs. actual position after a move (e-stop trigger).

        Returns:
            SafetyVerdict; ``safe=False`` suggests a stall / collision.
        """
        for i, (e, a) in enumerate(zip(expected_deg, actual_deg)):
            err = abs(e - a)
            if err > POSITION_ERROR_THRESHOLD_DEG:
                msg = (
                    f"Joint {i+1} position error {err:.1f}° exceeds "
                    f"threshold {POSITION_ERROR_THRESHOLD_DEG}° — possible collision"
                )
                logger.error(msg)
                return SafetyVerdict(safe=False, reason=msg)
        return SafetyVerdict(safe=True)

    # ── workspace checks ─────────────────────────────────────────────────

    @staticmethod
    def _check_target_xyz(xyz: List[float]) -> SafetyVerdict:
        """Verify the target XYZ lies inside the allowed workspace."""
        x, y, z = xyz
        mn, mx = WORKSPACE_MIN_XYZ, WORKSPACE_MAX_XYZ
        if not (mn[0] <= x <= mx[0] and mn[1] <= y <= mx[1]):
            return SafetyVerdict(
                safe=False,
                reason=f"Target XY ({x:.3f}, {y:.3f}) outside workspace",
            )
        if z < MIN_Z_METRES:
            return SafetyVerdict(
                safe=False,
                reason=f"Target Z {z:.4f} m below table guard {MIN_Z_METRES} m",
            )
        if z > mx[2]:
            return SafetyVerdict(
                safe=False,
                reason=f"Target Z {z:.4f} m above workspace ceiling {mx[2]} m",
            )
        return SafetyVerdict(safe=True)

    # ── emergency stop ───────────────────────────────────────────────────

    @staticmethod
    def emergency_stop(arm_controller) -> None:
        """Command the arm to hold its current position (soft e-stop).

        Reads the current joint angles and re-writes them as the goal,
        effectively freezing the arm in place.
        """
        try:
            current = arm_controller.read_joint_positions_deg()
            arm_controller.send_joint_angles(current, speed=0.0, block=False)
            logger.critical("EMERGENCY STOP — arm frozen at %s", current)
        except Exception:
            logger.exception("Emergency stop failed — power-cycle the robot!")
