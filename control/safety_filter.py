from __future__ import annotations

import numpy as np

from deploy.common.messages import RobotState
from deploy.config import SafetyConfig


class SafetyViolation(RuntimeError):
    pass


def _wrap_angle(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


class SafetyFilter:
    def __init__(self, config: SafetyConfig) -> None:
        self.config = config
        self.workspace_min = np.asarray(config.workspace_min_m, dtype=np.float64)
        self.workspace_max = np.asarray(config.workspace_max_m, dtype=np.float64)

    def _check_workspace(self, position: np.ndarray, label: str) -> None:
        if np.any(position < self.workspace_min) or np.any(position > self.workspace_max):
            raise SafetyViolation(f"{label} outside workspace: {position.tolist()}")

    def apply(self, action: np.ndarray, state: RobotState) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64).copy()
        if action.shape != (7,):
            raise SafetyViolation(f"Expected 7D Euler action, got {action.shape}")
        if not np.isfinite(action).all():
            raise SafetyViolation("Action contains NaN or Inf")
        if self.config.enforce_workspace:
            self._check_workspace(action[:3], "Target position")

        current_position = np.asarray(state.position_m, dtype=np.float64)
        translation = action[:3] - current_position
        distance = float(np.linalg.norm(translation))
        if distance > self.config.max_translation_step_m:
            action[:3] = current_position + translation * (
                self.config.max_translation_step_m / distance
            )
        if self.config.enforce_workspace:
            self._check_workspace(action[:3], "Limited position")

        current_rotation = np.asarray(state.euler_xyz_rad, dtype=np.float64)
        rotation_delta = _wrap_angle(action[3:6] - current_rotation)
        angle = float(np.linalg.norm(rotation_delta))
        if angle > self.config.max_rotation_step_rad:
            rotation_delta *= self.config.max_rotation_step_rad / angle
        action[3:6] = current_rotation + rotation_delta
        action[6] = (
            self.config.gripper_max_m
            if action[6] >= self.config.gripper_open_threshold
            else self.config.gripper_min_m
        )
        return action.astype(np.float32)
