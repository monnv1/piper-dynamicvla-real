from pathlib import Path

import pytest

from deploy.config import load_config

SEQUENTIAL_CONFIG = Path("deploy/configs/piper_sequential.yaml")


def test_sequential_config_only_retains_command_speed_limit() -> None:
    config = load_config(SEQUENTIAL_CONFIG)

    assert config.model.checkpoint == "/data/checkpoints/dynamicvla_merged"
    assert config.model.delta_action_ry_mean_scale == pytest.approx(1.0)
    assert not config.runtime.continuous_inference
    assert config.runtime.max_trusted_action_steps == 20
    assert config.runtime.action_execution_mode == "timed"
    assert config.runtime.action_hz == pytest.approx(40.0)
    assert config.runtime.control_hz == pytest.approx(40.0)
    assert config.runtime.action_completion_joint_tolerance_deg == pytest.approx(0.5)
    assert config.runtime.action_completion_settle_cycles == 3
    assert config.runtime.action_completion_timeout_s == pytest.approx(30.0)
    assert config.robot.command_speed_percent == 10
    assert config.robot.control_backend == "firmware_move_p"
    assert not config.safety.enforce_workspace
    assert config.safety.max_translation_step_m == pytest.approx(10.0)
    assert config.safety.max_rotation_step_rad == pytest.approx(10.0)
    assert config.safety.max_action_age_ms == pytest.approx(3_600_000.0)
    assert config.robot.ik_max_joint_step_deg == pytest.approx(360.0)
    assert config.robot.ik_min_joint_limit_margin_deg == pytest.approx(0.0)
    assert config.robot.command_gripper
    assert config.safety.gripper_open_threshold == pytest.approx(0.035)


def test_unknown_config_field_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("runtime:\n  control_hzz: 25\n", encoding="utf-8")

    with pytest.raises(ValueError, match="control_hzz"):
        load_config(path)


def test_future_history_index_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("runtime:\n  history_indices: [-2, 1, 0]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="future observations"):
        load_config(path)


def test_delta_action_ry_mean_scale_is_bounded(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "model:\n  delta_action_ry_mean_scale: 1.1\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="delta_action_ry_mean_scale"):
        load_config(path)


def test_action_hz_cannot_exceed_control_hz(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "runtime:\n"
        "  control_hz: 25\n"
        "  action_execution_mode: timed\n"
        "  action_hz: 40\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="action_hz"):
        load_config(path)
