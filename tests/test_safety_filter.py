import numpy as np
import pytest

from deploy.common.messages import RobotState
from deploy.config import SafetyConfig
from deploy.control.safety_filter import SafetyFilter, SafetyViolation


def state():
    return RobotState(
        host_timestamp_ns=0,
        joint_radians=np.zeros(6, dtype=np.float32),
        position_m=np.array([0.3, 0.0, 0.3], dtype=np.float32),
        euler_xyz_rad=np.zeros(3, dtype=np.float32),
        gripper_m=0.02,
    )


def test_limits_per_step_delta_and_gripper():
    safety = SafetyFilter(SafetyConfig())
    result = safety.apply(
        np.array([0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 1.0], dtype=np.float32),
        state(),
    )
    assert np.linalg.norm(result[:3] - state().position_m) <= 0.015001
    assert np.linalg.norm(result[3:6]) <= 0.120001
    assert result[6] == pytest.approx(0.07)


def test_rejects_workspace_violation():
    safety = SafetyFilter(SafetyConfig())
    with pytest.raises(SafetyViolation):
        safety.apply(np.array([1.0, 0, 0.3, 0, 0, 0, 0.02]), state())


def test_workspace_can_be_disabled_only_by_explicit_config():
    safety = SafetyFilter(
        SafetyConfig(enforce_workspace=False, max_translation_step_m=0.005)
    )
    current = state()
    action = np.array([0.0, 0.0, 0.2, 0, 0, 0, -1.0])
    limited = safety.apply(action, current)
    assert np.linalg.norm(limited[:3] - current.position_m) <= 0.005001


def test_rejects_non_finite_action():
    safety = SafetyFilter(SafetyConfig())
    with pytest.raises(SafetyViolation):
        safety.apply(np.array([0.3, 0, 0.3, np.nan, 0, 0, 0.02]), state())


def test_maps_binary_gripper_action_to_physical_stroke():
    safety = SafetyFilter(SafetyConfig())
    open_action = safety.apply(
        np.array([0.3, 0, 0.3, 0, 0, 0, 1.0]), state()
    )
    close_action = safety.apply(
        np.array([0.3, 0, 0.3, 0, 0, 0, -1.0]), state()
    )
    assert open_action[6] == pytest.approx(0.07)
    assert close_action[6] == pytest.approx(0.0)


def test_rejects_limited_step_that_remains_outside_workspace():
    safety = SafetyFilter(SafetyConfig())
    outside_state = state()
    outside_state = RobotState(
        host_timestamp_ns=outside_state.host_timestamp_ns,
        joint_radians=outside_state.joint_radians,
        position_m=np.array([0.05, 0.0, 0.2], dtype=np.float32),
        euler_xyz_rad=outside_state.euler_xyz_rad,
        gripper_m=outside_state.gripper_m,
    )
    with pytest.raises(SafetyViolation, match="Limited position"):
        safety.apply(
            np.array([0.2, 0.0, 0.2, 0, 0, 0, -1.0]), outside_state
        )
