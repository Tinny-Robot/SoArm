"""
Gemini LLM planner.

Sends the structured scene state and task instruction to the
gemini-robotics-er-1.6-preview model and parses the returned JSON action list.

Uses the google.genai SDK (successor to google.generativeai).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from soarm_gemini.config import (
    GEMINI_API_KEY,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_MODEL,
    GEMINI_TEMPERATURE,
)
from soarm_gemini.scene.scene_builder import SceneState
from soarm_gemini.utils.logger import GeminiLogger

logger = logging.getLogger(__name__)


# ── System prompt (used verbatim) ────────────────────────────────────────────

SYSTEM = """
You are the planning brain of a physical 6-DOF robot arm called SO-ARM101.
Your only job is to convert a task instruction and a scene description into 
a precise sequence of robot actions.

RULES:
1. Output ONLY valid JSON. No explanation, no markdown, no code blocks.
2. Every action must have: "action", "target_xyz" (metres), and optionally 
   "gripper" ("open" or "close") and "speed" (0.0–1.0, default 0.5).
3. Coordinate system: X = forward from arm base, Y = left, Z = up.
4. Always end with a "home" action to return the arm to rest position.
5. If the task is impossible given the scene, return: 
   [{"action": "abort", "reason": "<why>"}]

AVAILABLE ACTIONS:
- move:    {"action": "move", "target_xyz": [x, y, z], "speed": 0.5}
- grip:    {"action": "grip", "gripper": "close"}
- release: {"action": "release", "gripper": "open"}
- lift:    {"action": "lift", "delta_z": 0.05}
- home:    {"action": "home"}
- abort:   {"action": "abort", "reason": "..."}

EXAMPLE OUTPUT:
[
  {"action": "move",    "target_xyz": [0.18, 0.05, 0.02], "speed": 0.4},
  {"action": "grip",    "gripper": "close"},
  {"action": "lift",    "delta_z": 0.08},
  {"action": "move",    "target_xyz": [0.18, -0.12, 0.08], "speed": 0.5},
  {"action": "release", "gripper": "open"},
  {"action": "home"}
]
"""


# ── Data classes ─────────────────────────────────────────────────────────────

VALID_ACTIONS = {"move", "grip", "release", "lift", "home", "abort"}


@dataclass
class RobotAction:
    """Parsed single robot action from the Gemini response."""
    action: str
    target_xyz: Optional[List[float]] = None
    gripper: Optional[str] = None
    speed: Optional[float] = None
    delta_z: Optional[float] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"action": self.action}
        if self.target_xyz is not None:
            d["target_xyz"] = self.target_xyz
        if self.gripper is not None:
            d["gripper"] = self.gripper
        if self.speed is not None:
            d["speed"] = self.speed
        if self.delta_z is not None:
            d["delta_z"] = self.delta_z
        if self.reason is not None:
            d["reason"] = self.reason
        return d


# ── Planner ──────────────────────────────────────────────────────────────────

class GeminiPlanner:
    """Wraps the google.genai SDK to call Gemini for action planning."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        from google import genai
        from google.genai import types

        key = api_key or GEMINI_API_KEY
        if not key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Export it as an environment variable "
                "or pass it to GeminiPlanner(api_key=...)."
            )

        self._client = genai.Client(api_key=key)
        self._model_name = GEMINI_MODEL
        self._gen_config = types.GenerateContentConfig(
            system_instruction=SYSTEM,
            temperature=GEMINI_TEMPERATURE,
            max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
        )
        self._gemini_logger = GeminiLogger()
        logger.info("GeminiPlanner initialised with model=%s", GEMINI_MODEL)

    # ── public API ───────────────────────────────────────────────────────

    def plan(self, scene: SceneState) -> List[RobotAction]:
        """Send the scene state to Gemini and return a list of RobotActions.

        Args:
            scene: Fully populated SceneState.

        Returns:
            Ordered list of RobotAction to execute.

        Raises:
            ValueError: If Gemini returns unparseable or invalid JSON.
        """
        user_message = json.dumps(scene.to_dict(), indent=2)
        logger.info("Sending scene to Gemini:\n%s", user_message)

        response = self._client.models.generate_content(
            model=self._model_name,
            contents=user_message,
            config=self._gen_config,
        )
        raw_text = response.text.strip()

        self._gemini_logger.log(prompt=user_message, response=raw_text)
        logger.info("Gemini raw response:\n%s", raw_text)

        actions = self._parse_response(raw_text)
        logger.info("Parsed %d actions from Gemini", len(actions))
        return actions

    # ── parsing ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_response(raw: str) -> List[RobotAction]:
        """Parse the raw Gemini text into a list of RobotAction.

        Handles possible markdown fences and stray whitespace.
        """
        cleaned = raw.strip()

        fence_pattern = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
        match = fence_pattern.search(cleaned)
        if match:
            cleaned = match.group(1).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Gemini returned invalid JSON: {exc}\n---\n{raw}"
            ) from exc

        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list, got {type(data).__name__}")

        actions: List[RobotAction] = []
        for idx, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Action #{idx} is not a dict: {item}")
            action_name = item.get("action")
            if action_name not in VALID_ACTIONS:
                raise ValueError(
                    f"Action #{idx} has unknown action '{action_name}'. "
                    f"Valid: {VALID_ACTIONS}"
                )
            actions.append(
                RobotAction(
                    action=action_name,
                    target_xyz=item.get("target_xyz"),
                    gripper=item.get("gripper"),
                    speed=item.get("speed"),
                    delta_z=item.get("delta_z"),
                    reason=item.get("reason"),
                )
            )
        return actions

    # ── conversational / scene-description API ──────────────────────────

    def chat(self, question: str, scene: Optional[SceneState] = None) -> str:
        """Send a freeform question to Gemini about the scene.

        Unlike `plan()`, this does NOT use the action-planning system prompt.
        Gemini answers in natural language, describing what it "sees" in the
        structured scene, answering questions about objects, or giving advice.
        """
        from google.genai import types as gtypes

        parts: List[str] = []
        if scene is not None:
            parts.append(
                "Here is the current scene state observed by the robot's cameras:\n"
                + json.dumps(scene.to_dict(), indent=2)
            )
        parts.append(question)

        chat_config = gtypes.GenerateContentConfig(
            system_instruction=(
                "You are the AI brain of a 6-DOF robot arm called SO-ARM101. "
                "You can see the world through an overhead RGB camera and a wrist camera. "
                "The user may ask what you see, ask about objects, or ask general questions. "
                "Answer concisely and helpfully in natural language. "
                "If scene data is provided, describe the objects you detect, "
                "their approximate positions, and any relevant observations."
            ),
            temperature=0.3,
            max_output_tokens=1024,
        )

        try:
            resp = self._client.models.generate_content(
                model=self._model_name,
                contents="\n\n".join(parts),
                config=chat_config,
            )
            answer = resp.text.strip()
            self._gemini_logger.log(prompt="\n\n".join(parts), response=answer)
            return answer
        except Exception as exc:
            logger.exception("Gemini chat call failed")
            return f"Sorry, I couldn't process that: {exc}"

    # ── noun extraction helper (uses Gemini itself) ──────────────────────

    def extract_nouns(self, task: str) -> str:
        """Ask Gemini to pull out object nouns from a task string.

        Uses a separate config *without* the robot system prompt so Gemini
        doesn't confuse noun extraction with action planning.

        Returns a " . "-separated string suitable for Grounding DINO.
        Falls back to simple heuristic on failure.
        """
        from google.genai import types as gtypes

        prompt = (
            "Extract ONLY the physical object nouns from this robot task instruction. "
            "Return them as a comma-separated list with no other text.\n\n"
            f"Task: {task}"
        )
        try:
            noun_config = gtypes.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=256,
            )
            resp = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=noun_config,
            )
            raw = resp.text.strip().strip("`").strip()
            nouns = [n.strip() for n in raw.split(",") if n.strip()]
            if nouns:
                result = " . ".join(nouns)
                logger.info("Gemini noun extraction: '%s' → '%s'", task, result)
                return result
        except Exception:
            logger.warning("Gemini noun extraction failed — falling back to heuristic")

        from soarm_gemini.vision.detector import GroundingDINODetector
        return GroundingDINODetector.extract_nouns_simple(task)
