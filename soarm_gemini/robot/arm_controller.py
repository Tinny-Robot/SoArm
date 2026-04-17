"""
SO-ARM101 arm controller via lerobot's FeetechMotorsBus.

Provides connect / disconnect, joint-state reading, position-command sending,
and gripper open/close utilities.

Motor names and calibration match the lerobot SO-ARM101 follower arm config:
  shoulder_pan (id=1), shoulder_lift (id=2), elbow_flex (id=3),
  wrist_flex (id=4), wrist_roll (id=5), gripper (id=6).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from soarm_gemini.config import (
    DEFAULT_SPEED,
    GRIPPER_CLOSE_DEG,
    GRIPPER_OPEN_DEG,
    HOME_POSITION_DEG,
    NUM_JOINTS,
    ROBOT_PORT,
)

logger = logging.getLogger(__name__)

# Motor names matching the lerobot SO-ARM101 follower calibration
MOTOR_NAMES: List[str] = [
    "shoulder_pan",   # id 1
    "shoulder_lift",  # id 2
    "elbow_flex",     # id 3
    "wrist_flex",     # id 4
    "wrist_roll",     # id 5
    "gripper",        # id 6
]
MOTOR_IDS: List[int] = [1, 2, 3, 4, 5, 6]
GRIPPER_NAME: str = "gripper"
GRIPPER_IDX: int = MOTOR_NAMES.index(GRIPPER_NAME)

# Default calibration file cached by lerobot's calibration routine
DEFAULT_CALIBRATION_PATH: str = (
    str(Path.home() / ".cache/huggingface/lerobot/calibration"
        "/robots/so_follower/my_awesome_follower_arm.json")
)


class ArmController:
    """High-level wrapper around lerobot FeetechMotorsBus for the SO-ARM101 follower arm."""

    def __init__(
        self,
        port: Optional[str] = None,
        calibration_path: Optional[str] = None,
    ) -> None:
        self._port = port or ROBOT_PORT
        self._calibration_path = calibration_path or DEFAULT_CALIBRATION_PATH
        self._bus = None

    # ── lifecycle ────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Open the serial bus, load calibration, and ping all servos."""
        from lerobot.motors.feetech import FeetechMotorsBus
        from lerobot.motors.motors_bus import Motor, MotorCalibration, MotorNormMode

        motors: Dict[str, Motor] = {
            name: Motor(id=mid, model="sts3215", norm_mode=MotorNormMode.DEGREES)
            for name, mid in zip(MOTOR_NAMES, MOTOR_IDS)
        }

        self._bus = FeetechMotorsBus(port=self._port, motors=motors)
        self._bus.connect()

        # Load calibration from the cached JSON file
        calib = self._load_calibration()
        if calib is not None:
            self._bus.write_calibration(calib)
            logger.info("Calibration loaded from %s", self._calibration_path)
        else:
            logger.warning(
                "No calibration file found at %s — reading from servos",
                self._calibration_path,
            )
            calib = self._bus.read_calibration()
            self._bus.write_calibration(calib)
            logger.info("Calibration read from servos")

        logger.info(
            "ArmController connected on %s, motors=%s",
            self._port,
            MOTOR_NAMES,
        )

    def _load_calibration(self) -> Optional[dict]:
        """Load MotorCalibration dict from the JSON file."""
        from lerobot.motors.motors_bus import MotorCalibration

        path = Path(self._calibration_path)
        if not path.is_file():
            return None

        with open(path, "r") as f:
            raw = json.load(f)

        calib: Dict[str, MotorCalibration] = {}
        for name, data in raw.items():
            calib[name] = MotorCalibration(
                id=data["id"],
                drive_mode=data["drive_mode"],
                homing_offset=data["homing_offset"],
                range_min=data["range_min"],
                range_max=data["range_max"],
            )
        return calib

    def disconnect(self) -> None:
        """Disable torque and close the serial bus."""
        if self._bus is not None:
            try:
                self._bus.disconnect()
            except Exception:
                logger.exception("Error during bus disconnect")
            self._bus = None
            logger.info("ArmController disconnected")

    @property
    def is_connected(self) -> bool:
        return self._bus is not None and self._bus.is_connected

    # ── state reading ────────────────────────────────────────────────────

    def read_joint_positions_deg(self) -> List[float]:
        """Read current joint positions in degrees for all 6 joints.

        Returns:
            List of 6 floats (degrees), ordered by MOTOR_NAMES.
        """
        self._ensure_connected()
        pos_dict = self._bus.sync_read("Present_Position")
        return [round(float(pos_dict[name]), 2) for name in MOTOR_NAMES]

    def read_gripper_deg(self) -> float:
        """Return the current gripper joint position in degrees."""
        self._ensure_connected()
        try:
            return round(float(self._bus.read("Present_Position", GRIPPER_NAME)), 2)
        except RuntimeError:
            logger.warning("Failed to read gripper — falling back to sync_read")
            pos_dict = self._bus.sync_read("Present_Position")
            return round(float(pos_dict[GRIPPER_NAME]), 2)

    def get_gripper_state(self) -> str:
        """Return 'open' or 'close' based on current gripper position."""
        pos = self.read_gripper_deg()
        mid = (GRIPPER_OPEN_DEG + GRIPPER_CLOSE_DEG) / 2.0
        return "open" if pos >= mid else "close"

    # ── motion commands ──────────────────────────────────────────────────

    def send_joint_angles(
        self,
        angles_deg: List[float],
        speed: float = DEFAULT_SPEED,
        block: bool = True,
        timeout_s: float = 5.0,
    ) -> None:
        """Command all joints to the specified positions (degrees).

        Args:
            angles_deg: Target angles for joints 1-6 in degrees.
            speed: Normalised speed 0.0-1.0 (mapped to Goal_Velocity register).
            block: If True, wait until the arm reaches the target (within tolerance).
            timeout_s: Maximum seconds to wait when blocking.
        """
        self._ensure_connected()
        if len(angles_deg) != NUM_JOINTS:
            raise ValueError(f"Expected {NUM_JOINTS} angles, got {len(angles_deg)}")

        # Set velocity for all motors via Goal_Velocity register (STS3215, addr 46)
        velocity = max(1, int(speed * 600))
        self._bus.sync_write(
            "Goal_Velocity",
            {name: velocity for name in MOTOR_NAMES},
            normalize=False,
        )

        goal_dict = {
            name: float(angle)
            for name, angle in zip(MOTOR_NAMES, angles_deg)
        }
        self._bus.sync_write("Goal_Position", goal_dict)

        logger.info(
            "Commanded joints to %s (speed=%.2f, vel_reg=%d)",
            [round(a, 1) for a in angles_deg],
            speed,
            velocity,
        )

        if block:
            self._wait_until_reached(angles_deg, timeout_s=timeout_s)

    def go_home(self, speed: float = 0.4, max_delta_deg: float = 90.0) -> None:
        """Move all joints to the home position.

        Reads the current positions first. Any joint whose home target is
        more than *max_delta_deg* away is left at its current position to
        prevent stalls on startup.
        """
        try:
            current = self.read_joint_positions_deg()
        except Exception:
            logger.warning("Cannot read positions — skipping go_home")
            return

        target = list(HOME_POSITION_DEG)
        for i, (cur, home) in enumerate(zip(current, target)):
            if abs(cur - home) > max_delta_deg:
                logger.warning(
                    "Joint %s skipped in go_home (current=%.1f°, home=%.1f°, delta=%.1f° > max %.1f°)",
                    MOTOR_NAMES[i], cur, home, abs(cur - home), max_delta_deg,
                )
                target[i] = cur

        logger.info("Moving to home position: %s", [round(t, 1) for t in target])
        self.send_joint_angles(target, speed=speed)

    def open_gripper(self, speed: float = 0.5) -> None:
        """Open the gripper fully."""
        current = self.read_joint_positions_deg()
        current[GRIPPER_IDX] = GRIPPER_OPEN_DEG
        self.send_joint_angles(current, speed=speed)
        logger.info("Gripper opened")

    def close_gripper(self, speed: float = 0.5) -> None:
        """Close the gripper fully."""
        current = self.read_joint_positions_deg()
        current[GRIPPER_IDX] = GRIPPER_CLOSE_DEG
        self.send_joint_angles(current, speed=speed)
        logger.info("Gripper closed")

    # ── internal ─────────────────────────────────────────────────────────

    def _ensure_connected(self) -> None:
        if not self.is_connected:
            raise RuntimeError("ArmController is not connected — call .connect() first")

    def _wait_until_reached(
        self,
        target_deg: List[float],
        tolerance_deg: float = 3.0,
        timeout_s: float = 5.0,
    ) -> None:
        """Block until joints are within *tolerance_deg* of their targets."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            current = self.read_joint_positions_deg()
            errors = [abs(c - t) for c, t in zip(current, target_deg)]
            if all(e < tolerance_deg for e in errors):
                return
            time.sleep(0.05)
        logger.warning(
            "Timeout waiting for target %s (last read: %s)",
            target_deg,
            self.read_joint_positions_deg(),
        )

    # ── context manager ──────────────────────────────────────────────────

    def __enter__(self) -> "ArmController":
        self.connect()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.disconnect()
