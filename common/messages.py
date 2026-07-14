from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class CameraFrame:
    camera: str
    serial: str
    frame_number: int
    device_timestamp_ms: float
    host_timestamp_ns: int
    rgb: np.ndarray
    depth: np.ndarray | None = None


@dataclass(frozen=True)
class RobotState:
    host_timestamp_ns: int
    joint_radians: np.ndarray
    position_m: np.ndarray
    euler_xyz_rad: np.ndarray
    gripper_m: float
    feedback_hz: float = 0.0
    ctrl_mode: int = -1
    arm_status: int = -1
    mode_feed: int = -1
    motion_status: int = -1
    err_code: int = 0
    joint_limit_flags: tuple[bool, bool, bool, bool, bool, bool] = (
        False,
        False,
        False,
        False,
        False,
        False,
    )

    def model_vector(self) -> np.ndarray:
        return np.concatenate(
            (
                np.asarray(self.position_m, dtype=np.float32),
                np.asarray(self.euler_xyz_rad, dtype=np.float32),
                np.asarray([self.gripper_m], dtype=np.float32),
            )
        )


@dataclass(frozen=True)
class PolicyObservation:
    episode_id: str
    index: int
    host_timestamp_ns: int
    images: Mapping[str, np.ndarray]
    states: np.ndarray
    task: str


@dataclass(frozen=True)
class ActionChunk:
    episode_id: str
    observation_index: int
    observation_timestamp_ns: int
    completed_timestamp_ns: int
    actions: np.ndarray
    inference_seconds: float
    source_state: np.ndarray | None = None


@dataclass(frozen=True)
class ScheduledAction:
    episode_id: str
    target_index: int
    source_observation_index: int
    action: np.ndarray
    source_state: np.ndarray | None = None
