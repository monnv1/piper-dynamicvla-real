from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CameraConfig:
    driver: str = ""
    enabled: bool = True
    serial: str = ""
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass
class ModelConfig:
    checkpoint: str = ""
    device: str = "cuda"
    task: str = "place the object into the container"
    rotation: str = "euler"
    # Deployment-only scale applied in memory to delta-action ry mean.
    # This never rewrites the checkpoint on disk.
    delta_action_ry_mean_scale: float = 1.0


@dataclass
class RobotConfig:
    can_interface: str = "can0"
    dh_is_offset: int = 1
    official_can_adapter: bool = True
    auto_enable: bool = False
    command_gripper: bool = False
    command_speed_percent: int = 10
    control_backend: str = "host_pink_ik_move_j"
    # Source for RobotState.model_vector(): SDK EndPose or SDK FK from joint feedback.
    feedback_pose_source: str = "endpose"
    ik_position_tolerance_m: float = 0.002
    ik_rotation_tolerance_rad: float = 0.035
    ik_max_joint_step_deg: float = 5.0
    ik_min_joint_limit_margin_deg: float = 0.2
    ik_max_nfev: int = 60
    ik_allow_pose_projection: bool = False
    ik_projection_joint_limit_margin_deg: float = 2.0
    ik_projection_max_position_error_m: float = 0.003
    ik_projection_max_rotation_error_rad: float = 0.08
    ik_projection_position_weight: float = 1.0
    ik_projection_rotation_weight: float = 0.25
    diff_ik_lambda: float = 0.01
    diff_ik_finite_difference_eps_rad: float = 1e-4
    diff_ik_position_gain: float = 1.0
    diff_ik_rotation_gain: float = 1.0
    pink_urdf_path: str = "simulations/robots/PIPER/piper_description.urdf"
    pink_frame_name: str = "model_tcp"
    pink_parent_frame_name: str = "gripper_base"
    pink_solver: str = "proxqp"
    pink_dt: float = 0.04
    pink_position_cost: float = 1.0
    pink_orientation_cost: float = 0.25
    pink_posture_cost: float = 0.01
    pink_lm_damping: float = 0.0001
    pink_qpsolver_damping: float = 1e-12
    # Fixed transform T_sdk_model_tcp. Piper SDK/FK reports link6, while the
    # DynamicVLA simulation tracks gripper_base + 0.1334 m on local Z.
    sdk_to_model_translation_m: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.1334]
    )
    sdk_to_model_euler_xyz_rad: list[float] = field(
        default_factory=lambda: [0.0, 0.0, 3.141592653589793]
    )


@dataclass
class RuntimeConfig:
    mode: str = "shadow"
    control_hz: float = 25.0
    camera_sync_tolerance_ms: float = 50.0
    sensor_timeout_ms: float = 250.0
    startup_timeout_s: float = 300.0
    output_dir: str = "deploy/runs"
    history_indices: list[int] = field(default_factory=lambda: [-2, 0])
    record_video: bool = True
    video_fps: float = 25.0
    # Continuous mode overlaps inference and execution using LAAS. Sequential
    # mode waits for the trusted prefix to finish before starting inference.
    continuous_inference: bool = True
    max_trusted_action_steps: int = 20
    action_completion_joint_tolerance_deg: float = 0.5
    action_completion_settle_cycles: int = 3
    action_completion_timeout_s: float = 30.0
    # Execute-only wall-clock cap. Zero means unlimited.
    max_execute_seconds: float = 0.0
    # A normal execute exit may return to DynamicVLA's training-start joints.
    # Exceptions and Ctrl+C never trigger automatic return motion.
    return_to_training_start_on_normal_exit: bool = False
    return_speed_percent: int = 3
    return_timeout_s: float = 45.0


@dataclass
class SafetyConfig:
    workspace_min_m: list[float] = field(default_factory=lambda: [0.15, -0.35, 0.05])
    workspace_max_m: list[float] = field(default_factory=lambda: [0.60, 0.35, 0.55])
    enforce_workspace: bool = True
    max_translation_step_m: float = 0.015
    max_rotation_step_rad: float = 0.12
    max_action_age_ms: float = 800.0
    stale_action_hold_ms: float = 200.0
    hold_on_stale_action: bool = False
    gripper_open_threshold: float = 0.0
    gripper_min_m: float = 0.0
    gripper_max_m: float = 0.07


@dataclass
class DeployConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "opst_cam": CameraConfig(driver="realsense"),
            "wrist_cam": CameraConfig(driver="orbbec"),
        }
    )


def _construct(cls: type, values: dict[str, Any]):
    fields = {item.name: item for item in dataclasses.fields(cls)}
    unknown = sorted(set(values) - set(fields))
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} field(s): {', '.join(unknown)}")
    kwargs = {key: value for key, value in values.items() if key in fields}
    return cls(**kwargs)


def load_config(path: str | Path) -> DeployConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}

    cameras = {
        name: _construct(CameraConfig, values)
        for name, values in raw.get("cameras", {}).items()
    }
    config = DeployConfig(
        model=_construct(ModelConfig, raw.get("model", {})),
        robot=_construct(RobotConfig, raw.get("robot", {})),
        runtime=_construct(RuntimeConfig, raw.get("runtime", {})),
        safety=_construct(SafetyConfig, raw.get("safety", {})),
        cameras=cameras or DeployConfig().cameras,
    )
    if config.runtime.mode not in {"shadow", "execute"}:
        raise ValueError("runtime.mode must be 'shadow' or 'execute'")
    if config.model.rotation != "euler":
        raise ValueError("The current Piper adapter requires model.rotation=euler")
    if not 0.0 <= config.model.delta_action_ry_mean_scale <= 1.0:
        raise ValueError("model.delta_action_ry_mean_scale must be in [0, 1]")
    if config.runtime.control_hz <= 0:
        raise ValueError("runtime.control_hz must be positive")
    if not config.runtime.history_indices or config.runtime.history_indices[-1] != 0:
        raise ValueError("runtime.history_indices must be non-empty and end in 0")
    if any(index > 0 for index in config.runtime.history_indices):
        raise ValueError("runtime.history_indices cannot contain future observations")
    if config.runtime.history_indices != sorted(set(config.runtime.history_indices)):
        raise ValueError("runtime.history_indices must be strictly increasing")
    if config.runtime.camera_sync_tolerance_ms <= 0:
        raise ValueError("runtime.camera_sync_tolerance_ms must be positive")
    if config.runtime.sensor_timeout_ms <= 0:
        raise ValueError("runtime.sensor_timeout_ms must be positive")
    if config.runtime.startup_timeout_s <= 0:
        raise ValueError("runtime.startup_timeout_s must be positive")
    if config.runtime.video_fps <= 0:
        raise ValueError("runtime.video_fps must be positive")
    if config.runtime.max_trusted_action_steps <= 0:
        raise ValueError("runtime.max_trusted_action_steps must be positive")
    if config.runtime.action_completion_joint_tolerance_deg <= 0:
        raise ValueError(
            "runtime.action_completion_joint_tolerance_deg must be positive"
        )
    if config.runtime.action_completion_settle_cycles <= 0:
        raise ValueError("runtime.action_completion_settle_cycles must be positive")
    if config.runtime.action_completion_timeout_s <= 0:
        raise ValueError("runtime.action_completion_timeout_s must be positive")
    if config.runtime.max_execute_seconds < 0:
        raise ValueError("runtime.max_execute_seconds must be non-negative")
    if not 1 <= config.runtime.return_speed_percent <= 10:
        raise ValueError("runtime.return_speed_percent must be in [1, 10]")
    if config.runtime.return_timeout_s <= 0:
        raise ValueError("runtime.return_timeout_s must be positive")
    if len(config.robot.sdk_to_model_translation_m) != 3:
        raise ValueError("robot.sdk_to_model_translation_m must have 3 values")
    if len(config.robot.sdk_to_model_euler_xyz_rad) != 3:
        raise ValueError("robot.sdk_to_model_euler_xyz_rad must have 3 values")
    if not 1 <= config.robot.command_speed_percent <= 100:
        raise ValueError("robot.command_speed_percent must be in [1, 100]")
    if config.robot.control_backend not in {
        "host_ik_move_j",
        "host_diff_ik_move_j",
        "host_pink_ik_move_j",
        "firmware_move_p",
    }:
        raise ValueError(
            "robot.control_backend must be 'host_ik_move_j', "
            "'host_diff_ik_move_j', 'host_pink_ik_move_j', or 'firmware_move_p'"
        )
    if config.robot.feedback_pose_source not in {"endpose", "fk"}:
        raise ValueError("robot.feedback_pose_source must be 'endpose' or 'fk'")
    if config.robot.ik_position_tolerance_m <= 0:
        raise ValueError("robot.ik_position_tolerance_m must be positive")
    if config.robot.ik_rotation_tolerance_rad <= 0:
        raise ValueError("robot.ik_rotation_tolerance_rad must be positive")
    if config.robot.ik_max_joint_step_deg <= 0:
        raise ValueError("robot.ik_max_joint_step_deg must be positive")
    if config.robot.ik_min_joint_limit_margin_deg < 0:
        raise ValueError("robot.ik_min_joint_limit_margin_deg must be non-negative")
    if config.robot.ik_max_nfev <= 0:
        raise ValueError("robot.ik_max_nfev must be positive")
    if (
        config.robot.ik_projection_joint_limit_margin_deg
        < config.robot.ik_min_joint_limit_margin_deg
    ):
        raise ValueError(
            "robot.ik_projection_joint_limit_margin_deg must be >= "
            "ik_min_joint_limit_margin_deg"
        )
    if config.robot.ik_projection_max_position_error_m <= 0:
        raise ValueError("robot.ik_projection_max_position_error_m must be positive")
    if config.robot.ik_projection_max_rotation_error_rad <= 0:
        raise ValueError("robot.ik_projection_max_rotation_error_rad must be positive")
    if config.robot.ik_projection_position_weight <= 0:
        raise ValueError("robot.ik_projection_position_weight must be positive")
    if config.robot.ik_projection_rotation_weight <= 0:
        raise ValueError("robot.ik_projection_rotation_weight must be positive")
    if config.robot.diff_ik_lambda <= 0:
        raise ValueError("robot.diff_ik_lambda must be positive")
    if config.robot.diff_ik_finite_difference_eps_rad <= 0:
        raise ValueError("robot.diff_ik_finite_difference_eps_rad must be positive")
    if config.robot.diff_ik_position_gain <= 0:
        raise ValueError("robot.diff_ik_position_gain must be positive")
    if config.robot.diff_ik_rotation_gain <= 0:
        raise ValueError("robot.diff_ik_rotation_gain must be positive")
    if config.robot.pink_dt <= 0:
        raise ValueError("robot.pink_dt must be positive")
    if config.robot.pink_position_cost <= 0:
        raise ValueError("robot.pink_position_cost must be positive")
    if config.robot.pink_orientation_cost < 0:
        raise ValueError("robot.pink_orientation_cost must be non-negative")
    if config.robot.pink_posture_cost < 0:
        raise ValueError("robot.pink_posture_cost must be non-negative")
    if config.robot.pink_lm_damping < 0:
        raise ValueError("robot.pink_lm_damping must be non-negative")
    if config.robot.pink_qpsolver_damping < 0:
        raise ValueError("robot.pink_qpsolver_damping must be non-negative")
    if (
        len(config.safety.workspace_min_m) != 3
        or len(config.safety.workspace_max_m) != 3
    ):
        raise ValueError("safety workspace bounds must each have 3 values")
    if any(
        low >= high
        for low, high in zip(
            config.safety.workspace_min_m, config.safety.workspace_max_m
        )
    ):
        raise ValueError("safety.workspace_min_m must be below workspace_max_m")
    if config.safety.max_translation_step_m <= 0:
        raise ValueError("safety.max_translation_step_m must be positive")
    if config.safety.max_rotation_step_rad <= 0:
        raise ValueError("safety.max_rotation_step_rad must be positive")
    if config.safety.max_action_age_ms <= 0:
        raise ValueError("safety.max_action_age_ms must be positive")
    if config.safety.stale_action_hold_ms <= 0:
        raise ValueError("safety.stale_action_hold_ms must be positive")
    if config.safety.gripper_min_m >= config.safety.gripper_max_m:
        raise ValueError("safety.gripper_min_m must be below gripper_max_m")
    if (
        not config.safety.gripper_min_m
        <= config.safety.gripper_open_threshold
        <= config.safety.gripper_max_m
    ):
        raise ValueError("safety.gripper_open_threshold must be within gripper limits")
    for name, camera in config.cameras.items():
        if camera.driver not in {"orbbec", "realsense"}:
            raise ValueError(f"Unsupported camera driver for {name}: {camera.driver}")
    return config
