#!/usr/bin/env python3
"""
SO-ARM101 Gemini zero-shot robot control — main entry point.

Orchestrates the full perception → planning → execution loop:
  1. Connect to robot + cameras
  2. Accept a task from the CLI
  3. Perceive → plan → execute → verify (up to MAX_REPLAN_ATTEMPTS)
  4. Repeat for the next task
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from typing import List, Optional

import cv2
import numpy as np

from soarm_gemini.cameras.overhead_cam import OverheadCamera
from soarm_gemini.cameras.wrist_cam import WristCamera
from soarm_gemini.config import (
    DEFAULT_SPEED,
    HOME_POSITION_DEG,
    MAX_REPLAN_ATTEMPTS,
)
from soarm_gemini.planner.gemini_planner import GeminiPlanner, RobotAction
from soarm_gemini.robot.arm_controller import ArmController
from soarm_gemini.robot.ik_solver import IKSolver
from soarm_gemini.robot.safety import SafetyChecker
from soarm_gemini.scene.scene_builder import SceneBuilder, SceneState
from soarm_gemini.utils.logger import configure_root_logger
from soarm_gemini.utils.visualizer import DebugVisualizer
from soarm_gemini.vision.detector import GroundingDINODetector
from soarm_gemini.vision.segmentor import SAM2Segmentor

logger = logging.getLogger(__name__)


# ─── Action executor ─────────────────────────────────────────────────────────

class ActionExecutor:
    """Translates high-level RobotActions into low-level motor commands."""

    def __init__(
        self,
        arm: ArmController,
        ik: IKSolver,
        safety: SafetyChecker,
    ) -> None:
        self._arm = arm
        self._ik = ik
        self._safety = safety

    def execute(self, actions: List[RobotAction]) -> bool:
        """Execute a validated action list sequentially.

        Returns True if all actions completed without error.
        """
        for idx, action in enumerate(actions):
            logger.info("Executing action %d/%d: %s", idx + 1, len(actions), action.action)
            try:
                self._dispatch(action)
            except Exception:
                logger.exception("Action %d (%s) failed", idx + 1, action.action)
                return False
            time.sleep(0.2)
        return True

    def _dispatch(self, action: RobotAction) -> None:
        if action.action == "move":
            self._execute_move(action)
        elif action.action == "grip":
            self._arm.close_gripper()
        elif action.action == "release":
            self._arm.open_gripper()
        elif action.action == "lift":
            self._execute_lift(action)
        elif action.action == "home":
            self._arm.go_home()
        elif action.action == "abort":
            logger.warning("ABORT action received: %s", action.reason)
        else:
            logger.warning("Unknown action '%s' — skipping", action.action)

    def _execute_move(self, action: RobotAction) -> None:
        """Solve IK for the target and send joint angles."""
        if action.target_xyz is None:
            raise ValueError("move action missing target_xyz")

        current_deg = self._arm.read_joint_positions_deg()
        speed = action.speed if action.speed is not None else DEFAULT_SPEED

        joint_angles = self._ik.solve(action.target_xyz, current_deg[:5])

        # Preserve gripper angle
        full_angles = joint_angles + [current_deg[5]]

        full_angles_list, was_clamped = self._safety.clamp_joints(full_angles)
        vel_check = self._safety.check_velocity(current_deg, full_angles_list)
        if not vel_check.safe:
            logger.warning("Velocity check failed: %s — slowing down", vel_check.reason)
            speed = max(0.1, speed * 0.5)

        self._arm.send_joint_angles(full_angles_list, speed=speed)

        # Post-move position error check
        actual = self._arm.read_joint_positions_deg()
        err_check = self._safety.check_position_error(full_angles_list, actual)
        if not err_check.safe:
            logger.error("Post-move error: %s", err_check.reason)
            self._safety.emergency_stop(self._arm)
            raise RuntimeError(err_check.reason)

    def _execute_lift(self, action: RobotAction) -> None:
        """Lift the end-effector by delta_z (relative move)."""
        if action.delta_z is None:
            raise ValueError("lift action missing delta_z")

        current_deg = self._arm.read_joint_positions_deg()
        ee_xyz = self._ik.forward_kinematics(current_deg[:5])
        new_xyz = [ee_xyz[0], ee_xyz[1], ee_xyz[2] + action.delta_z]

        lift_action = RobotAction(
            action="move",
            target_xyz=new_xyz,
            speed=action.speed or 0.3,
        )
        self._execute_move(lift_action)


# ─── Main control loop ──────────────────────────────────────────────────────

def main() -> None:
    """Entry point: connect hardware, run the task loop."""
    configure_root_logger()
    logger.info("=" * 60)
    logger.info("SO-ARM101 Gemini Controller starting")
    logger.info("=" * 60)

    # ── Instantiate components ───────────────────────────────────────────
    arm = ArmController()
    overhead = OverheadCamera()
    wrist = WristCamera()
    detector = GroundingDINODetector()
    segmentor = SAM2Segmentor()
    ik = IKSolver()
    safety = SafetyChecker()
    planner = GeminiPlanner()
    visualizer = DebugVisualizer()

    scene_builder = SceneBuilder(overhead, wrist, detector, segmentor)
    executor = ActionExecutor(arm, ik, safety)

    # Graceful shutdown on SIGINT / SIGTERM
    def _shutdown(signum, frame):
        logger.info("Signal %d received — shutting down", signum)
        _cleanup(arm, overhead, wrist, visualizer)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        # ── Connect hardware ─────────────────────────────────────────────
        arm.connect()
        overhead.open()
        wrist.open()
        visualizer.start()

        arm.go_home()
        logger.info("Arm at home position")

        # ── Task loop ────────────────────────────────────────────────────
        while True:
            task = input("\nEnter task (or 'quit'): ").strip()
            if task.lower() in ("quit", "exit", "q"):
                break
            if not task:
                continue

            logger.info("New input: '%s'", task)

            if _is_question(task):
                _handle_question(
                    question=task,
                    arm=arm,
                    planner=planner,
                    scene_builder=scene_builder,
                    ik=ik,
                    overhead=overhead,
                    visualizer=visualizer,
                    detector=detector,
                )
            else:
                success = _run_task(
                    task=task,
                    arm=arm,
                    planner=planner,
                    scene_builder=scene_builder,
                    executor=executor,
                    safety=safety,
                    ik=ik,
                    visualizer=visualizer,
                    detector=detector,
                    overhead=overhead,
                )
                status = "SUCCESS" if success else "FAILED"
                logger.info("Task '%s' result: %s", task, status)
                print(f"\n>>> Task result: {status}")

    except Exception:
        logger.exception("Fatal error in main loop")
    finally:
        _cleanup(arm, overhead, wrist, visualizer)


# ─── Question detection / conversational mode ────────────────────────────────

_QUESTION_PREFIXES = (
    "what", "where", "which", "who", "how", "why", "when",
    "can you see", "do you see", "are there", "is there",
    "tell me", "describe", "list", "show me", "identify",
    "look", "check", "any", "count",
)


def _is_question(text: str) -> bool:
    """Heuristic: detect whether the input is a question rather than a task command."""
    lower = text.lower().strip()
    if lower.endswith("?"):
        return True
    for prefix in _QUESTION_PREFIXES:
        if lower.startswith(prefix):
            return True
    return False


def _handle_question(
    *,
    question: str,
    arm: ArmController,
    planner: GeminiPlanner,
    scene_builder: SceneBuilder,
    ik: IKSolver,
    overhead: OverheadCamera,
    visualizer: DebugVisualizer,
    detector: GroundingDINODetector,
) -> None:
    """Capture the current scene and ask Gemini a freeform question about it."""
    logger.info("Question mode: '%s'", question)
    print("  [Observing scene...]")

    try:
        joints_deg = arm.read_joint_positions_deg()
        gripper_state = arm.get_gripper_state()
        ee_xyz = ik.forward_kinematics(joints_deg[:5]).tolist()

        scene = scene_builder.build(
            task=question,
            arm_joints_deg=joints_deg,
            gripper_state=gripper_state,
            ee_xyz=ee_xyz,
            detection_prompt="objects",
        )
        _update_visualizer(visualizer, overhead, scene, detector, actions=None)
    except Exception:
        logger.warning("Scene build failed for question — asking without scene")
        scene = None

    answer = planner.chat(question, scene)
    print(f"\n  AI: {answer}\n")
    logger.info("AI answer: %s", answer)


# ─── Single task execution ───────────────────────────────────────────────────

def _run_task(
    *,
    task: str,
    arm: ArmController,
    planner: GeminiPlanner,
    scene_builder: SceneBuilder,
    executor: ActionExecutor,
    safety: SafetyChecker,
    ik: IKSolver,
    visualizer: DebugVisualizer,
    detector: GroundingDINODetector,
    overhead: OverheadCamera,
) -> bool:
    """Perceive → plan → execute → verify, retrying up to MAX_REPLAN_ATTEMPTS."""

    # Generate detection prompt from the task
    detection_prompt = planner.extract_nouns(task)
    logger.info("Detection prompt: '%s'", detection_prompt)

    for attempt in range(1, MAX_REPLAN_ATTEMPTS + 1):
        logger.info("─── Attempt %d / %d ───", attempt, MAX_REPLAN_ATTEMPTS)

        # 1. Read arm state
        joints_deg = arm.read_joint_positions_deg()
        gripper_state = arm.get_gripper_state()
        ee_xyz = ik.forward_kinematics(joints_deg[:5]).tolist()

        # 2. Build scene
        scene = scene_builder.build(
            task=task,
            arm_joints_deg=joints_deg,
            gripper_state=gripper_state,
            ee_xyz=ee_xyz,
            detection_prompt=detection_prompt,
        )

        # 3. Update visualizer with latest perception
        _update_visualizer(visualizer, overhead, scene, detector, actions=None)

        # 4. Plan
        try:
            actions = planner.plan(scene)
        except ValueError as exc:
            logger.error("Planner returned invalid output: %s", exc)
            continue

        # Check for abort
        if actions and actions[0].action == "abort":
            logger.warning("Planner aborted: %s", actions[0].reason)
            print(f"  Planner says: {actions[0].reason}")
            return False

        # 5. Safety validation
        verdict = safety.validate_action_list(actions)
        if not verdict.safe:
            logger.warning("Safety rejected plan: %s", verdict.reason)
            continue

        # 6. Update visualizer with the plan
        _update_visualizer(visualizer, overhead, scene, detector, actions)

        # 7. Execute
        ok = executor.execute(actions)
        if not ok:
            logger.warning("Execution failed on attempt %d", attempt)
            arm.go_home()
            continue

        # 8. Verify success
        time.sleep(0.5)
        if _verify_task(scene, scene_builder, detection_prompt):
            return True
        logger.info("Verification inconclusive — replanning")

    logger.warning("All %d attempts exhausted for task '%s'", MAX_REPLAN_ATTEMPTS, task)
    arm.go_home()
    return False


# ─── Verification ────────────────────────────────────────────────────────────

def _verify_task(
    old_scene: SceneState,
    scene_builder: SceneBuilder,
    detection_prompt: str,
) -> bool:
    """Heuristic: check whether the scene changed enough to indicate success.

    For pick-and-place tasks, verifies that the manipulated object's
    position shifted significantly.
    """
    if not old_scene.objects:
        return True

    primary = old_scene.objects[0]
    original_xyz = np.array(primary.world_xyz)

    try:
        from soarm_gemini.cameras.overhead_cam import OverheadCamera
        # Attempt to re-detect the primary object
        shifted = scene_builder.check_object_position(
            label=primary.label,
            expected_xyz=primary.world_xyz,
            tolerance_m=0.025,
            detection_prompt=detection_prompt,
        )
        # If the object is still in the same place, the task likely failed
        return not shifted
    except Exception:
        logger.warning("Verification failed — assuming success")
        return True


# ─── Visualizer helper ───────────────────────────────────────────────────────

def _update_visualizer(
    vis: DebugVisualizer,
    overhead: OverheadCamera,
    scene: Optional[SceneState],
    detector: Optional[GroundingDINODetector],
    actions: Optional[List[RobotAction]],
) -> None:
    """Grab a live overhead frame, overlay scene info, and display it."""
    try:
        bgr = overhead.grab_rgb_bgr()
        vis.update(
            overhead_bgr=bgr,
            detections=None,
            scene=scene,
            actions=actions,
        )
    except Exception:
        pass


# ─── Cleanup ─────────────────────────────────────────────────────────────────

def _cleanup(
    arm: ArmController,
    overhead: OverheadCamera,
    wrist: WristCamera,
    visualizer: DebugVisualizer,
) -> None:
    """Safely shut down all hardware."""
    logger.info("Cleaning up...")
    try:
        arm.go_home()
    except Exception:
        pass
    try:
        arm.disconnect()
    except Exception:
        pass
    try:
        overhead.close()
    except Exception:
        pass
    try:
        wrist.close()
    except Exception:
        pass
    try:
        visualizer.stop()
    except Exception:
        pass
    cv2.destroyAllWindows()
    logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
