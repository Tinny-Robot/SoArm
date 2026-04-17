"""
Central configuration for the SO-ARM101 Gemini robot control system.
All hardware ports, camera indices, joint limits, and tuning constants live here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ── Robot Hardware ───────────────────────────────────────────────────────────

ROBOT_PORT: str = "/dev/ttyACM0"
ROBOT_BAUDRATE: int = 1_000_000

JOINT_IDS: List[int] = [1, 2, 3, 4, 5, 6]
GRIPPER_JOINT_ID: int = 6
NUM_JOINTS: int = 6

# STS3215 position range mapped to degrees (0–4095 → 0–360°)
SERVO_POSITION_MIN: int = 0
SERVO_POSITION_MAX: int = 4095
SERVO_DEGREES_PER_TICK: float = 360.0 / 4096.0

# Joint limits in degrees [min, max] — conservative safe envelope
JOINT_LIMITS_DEG: Dict[int, Tuple[float, float]] = {
    1: (-150.0, 150.0),   # base rotation
    2: (-90.0,  90.0),    # shoulder
    3: (-120.0, 30.0),    # elbow
    4: (-90.0,  90.0),    # wrist pitch
    5: (-150.0, 150.0),   # wrist roll
    6: (0.0,    100.0),   # gripper (0 = closed, 100 = open)
}

# Home position in degrees for each joint
# Order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
# Set to None to preserve the current position on that joint at startup
HOME_POSITION_DEG: List[float] = [0.0, -30.0, 60.0, 0.0, 0.0, 50.0]

# Gripper thresholds (servo position units)
GRIPPER_OPEN_DEG: float = 80.0
GRIPPER_CLOSE_DEG: float = 5.0

# Movement speed default (0.0–1.0 → mapped to servo velocity register)
DEFAULT_SPEED: float = 0.5
MAX_SERVO_VELOCITY: int = 600


# ── SO-ARM101 Kinematics (DH parameters in metres) ──────────────────────────

@dataclass(frozen=True)
class DHParams:
    """Denavit–Hartenberg parameters for one link."""
    d: float       # link offset along previous z (m)
    a: float       # link length along rotated x (m)
    alpha: float   # twist angle (rad)

import math

ARM_DH: List[DHParams] = [
    DHParams(d=0.0555, a=0.0,    alpha=-math.pi / 2),  # joint 1 – base
    DHParams(d=0.0,    a=0.104,  alpha=0.0),            # joint 2 – shoulder
    DHParams(d=0.0,    a=0.089,  alpha=0.0),            # joint 3 – elbow
    DHParams(d=0.0,    a=0.0,    alpha=-math.pi / 2),   # joint 4 – wrist pitch
    DHParams(d=0.087,  a=0.0,    alpha=0.0),            # joint 5 – wrist roll
]

# Workspace bounding box (metres from arm base origin)
WORKSPACE_MIN_XYZ: Tuple[float, float, float] = (-0.30, -0.30, -0.01)
WORKSPACE_MAX_XYZ: Tuple[float, float, float] = (0.30, 0.30, 0.35)


# ── Cameras ──────────────────────────────────────────────────────────────────

WRIST_CAMERA_INDEX: int = 1          # /dev/video1 — EMEET SmartCam (wrist)
WRIST_CAMERA_WIDTH: int = 640
WRIST_CAMERA_HEIGHT: int = 480
WRIST_CAMERA_FPS: int = 30

OVERHEAD_CAMERA_INDEX: int = 5       # /dev/video5 — Astra Pro HD RGB via UVC
OVERHEAD_CAMERA_WIDTH: int = 1920
OVERHEAD_CAMERA_HEIGHT: int = 1080
OVERHEAD_CAMERA_FPS: int = 30

# Path to the OpenNI2 shared library directory (for depth, if driver available)
OPENNI2_LIB_DIR: str = "/usr/lib/x86_64-linux-gnu/"

# Orbbec Astra depth intrinsics (default factory calibration)
DEPTH_FX: float = 570.34
DEPTH_FY: float = 570.34
DEPTH_CX: float = 960.0
DEPTH_CY: float = 540.0
DEPTH_SCALE: float = 0.001           # raw depth unit → metres


# ── Overhead Camera Extrinsics (camera frame → arm base frame) ───────────────

import numpy as np

# Rotation matrix: camera optical frame → robot base frame
# Assumes camera is mounted directly above, looking straight down, with
# camera-X aligned to robot-X.  Adjust after calibration.
OVERHEAD_R_CAM_TO_BASE: np.ndarray = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
], dtype=np.float64)

# Translation vector: camera optical centre position in robot base frame (m)
OVERHEAD_T_CAM_TO_BASE: np.ndarray = np.array([0.0, 0.0, 0.45], dtype=np.float64)


# ── Vision Models ────────────────────────────────────────────────────────────

GROUNDING_DINO_CONFIG: str = os.path.join(
    os.path.dirname(__import__("groundingdino").__file__),
    "config", "GroundingDINO_SwinT_OGC.py",
)
GROUNDING_DINO_WEIGHTS: str = os.path.join("weights", "groundingdino_swint_ogc.pth")

SAM2_CHECKPOINT: str = "weights/sam2_hiera_large.pt"
SAM2_MODEL_CFG: str = "sam2_hiera_l.yaml"

DETECTION_CONFIDENCE_THRESHOLD: float = 0.35
DETECTION_NMS_THRESHOLD: float = 0.5
SAM2_POINTS_PER_SIDE: int = 32


# ── Gemini LLM ───────────────────────────────────────────────────────────────

GEMINI_MODEL: str = "gemini-robotics-er-1.6-preview"
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")

GEMINI_TEMPERATURE: float = 0.0
GEMINI_MAX_OUTPUT_TOKENS: int = 2048

# Maximum planning attempts before giving up on a task
MAX_REPLAN_ATTEMPTS: int = 3


# ── Safety ───────────────────────────────────────────────────────────────────

# Maximum joint velocity in deg/s for collision pre-check
MAX_JOINT_VELOCITY_DEG_S: float = 120.0

# Minimum height (m) the end-effector may reach (table surface guard)
MIN_Z_METRES: float = -0.005

# Emergency-stop: max deviation from expected position after move (deg)
POSITION_ERROR_THRESHOLD_DEG: float = 15.0


# ── Logging ──────────────────────────────────────────────────────────────────

LOG_DIR: str = "logs"
LOG_LEVEL: str = "INFO"
GEMINI_LOG_DIR: str = os.path.join(LOG_DIR, "gemini_calls")


# ── Visualizer ───────────────────────────────────────────────────────────────

VIS_WINDOW_NAME: str = "SO-ARM101 Debug"
VIS_SCALE: float = 0.6                 # scale-down factor for display
VIS_FONT_SCALE: float = 0.5
VIS_BBOX_THICKNESS: int = 2
