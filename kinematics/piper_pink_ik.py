from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from deploy.kinematics.piper_ik import HostIKError, JOINT_LOWER_DEG, JOINT_UPPER_DEG


class PiperPinkIK:
    """Pink + Pinocchio differential IK for Piper model-TCP targets.

    This backend works in the DynamicVLA model TCP frame directly:
    ``base_link -> gripper_base -> local +Z tcp_offset_m``. It returns a joint
    position target for Piper JointCtrl, same command path as host IK.
    """

    def __init__(
        self,
        urdf_path: str | Path,
        frame_name: str = "model_tcp",
        parent_frame_name: str = "gripper_base",
        tcp_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.1334),
        solver: str = "proxqp",
        dt: float = 0.04,
        position_cost: float = 1.0,
        orientation_cost: float = 0.25,
        posture_cost: float = 0.01,
        lm_damping: float = 1e-4,
        qpsolver_damping: float = 1e-12,
        max_joint_step_deg: float = 5.0,
        min_joint_limit_margin_deg: float = 0.0,
    ) -> None:
        try:
            import pinocchio as pin
            import pink
            from pink.limits import ConfigurationLimit
            from pink.tasks import FrameTask, PostureTask
        except Exception as error:  # pragma: no cover - depends on optional install
            raise ImportError(
                "PiperPinkIK requires pin-pink, pinocchio, qpsolvers, and a QP "
                "solver such as proxsuite. Install with: "
                "python -m pip install pin pin-pink qpsolvers proxsuite osqp"
            ) from error

        self.pin = pin
        self.pink = pink
        self.ConfigurationLimit = ConfigurationLimit
        self.FrameTask = FrameTask
        self.PostureTask = PostureTask
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        self.frame_name = frame_name
        self.parent_frame_name = parent_frame_name
        self.tcp_offset_m = np.asarray(tcp_offset_m, dtype=np.float64)
        self.solver = solver
        self.dt = float(dt)
        self.position_cost = float(position_cost)
        self.orientation_cost = float(orientation_cost)
        self.posture_cost = float(posture_cost)
        self.lm_damping = float(lm_damping)
        self.qpsolver_damping = float(qpsolver_damping)
        self.max_joint_step_deg = float(max_joint_step_deg)
        self.min_joint_limit_margin_deg = float(min_joint_limit_margin_deg)

        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"Piper URDF not found: {self.urdf_path}")
        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()
        self._override_arm_limits()
        self._add_tcp_frame_if_needed()
        self.frame_task = FrameTask(
            self.frame_name,
            position_cost=self.position_cost,
            orientation_cost=self.orientation_cost,
            lm_damping=self.lm_damping,
        )
        self.posture_task = PostureTask(cost=self.posture_cost, lm_damping=self.lm_damping)
        self.limits = [ConfigurationLimit(self.model)]
        self.lower_rad = self.model.lowerPositionLimit[:6].copy()
        self.upper_rad = self.model.upperPositionLimit[:6].copy()

    def _override_arm_limits(self) -> None:
        margin_rad = np.radians(self.min_joint_limit_margin_deg)
        self.model.lowerPositionLimit[:6] = np.radians(JOINT_LOWER_DEG) + margin_rad
        self.model.upperPositionLimit[:6] = np.radians(JOINT_UPPER_DEG) - margin_rad
        # Keep gripper joints fixed at a neutral feasible value for IK.
        if self.model.nq >= 8:
            self.model.lowerPositionLimit[6:] = np.asarray([0.0, -0.035])
            self.model.upperPositionLimit[6:] = np.asarray([0.035, 0.0])

    def _add_tcp_frame_if_needed(self) -> None:
        if self.model.existFrame(self.frame_name):
            return
        pin = self.pin
        parent_frame_id = self.model.getFrameId(self.parent_frame_name)
        if parent_frame_id >= len(self.model.frames):
            raise ValueError(f"Pinocchio frame not found: {self.parent_frame_name}")
        parent_frame = self.model.frames[parent_frame_id]
        tcp_placement = parent_frame.placement * pin.SE3(
            np.eye(3), self.tcp_offset_m
        )
        self.model.addFrame(
            pin.Frame(
                self.frame_name,
                parent_frame.parentJoint,
                parent_frame_id,
                tcp_placement,
                pin.FrameType.OP_FRAME,
            )
        )
        self.data = self.model.createData()

    def _configuration(self, current_joint_rad: np.ndarray):
        q = np.zeros(self.model.nq, dtype=np.float64)
        q[:6] = np.clip(current_joint_rad, self.lower_rad, self.upper_rad)
        if self.model.nq >= 8:
            q[6:] = np.asarray([0.035, -0.035])
        return self.pink.Configuration(self.model, self.data, q)

    @staticmethod
    def _target_se3(position_m: np.ndarray, euler_xyz_rad: np.ndarray, pin):
        rotation = Rotation.from_euler("xyz", euler_xyz_rad).as_matrix()
        return pin.SE3(rotation, position_m)

    def solve(
        self,
        target_position_m: np.ndarray,
        target_euler_xyz_rad: np.ndarray,
        current_joint_rad: np.ndarray,
    ) -> dict[str, object]:
        started = time.monotonic()
        target_position_m = np.asarray(target_position_m, dtype=np.float64)
        target_euler_xyz_rad = np.asarray(target_euler_xyz_rad, dtype=np.float64)
        current_joint_rad = np.asarray(current_joint_rad, dtype=np.float64)
        if target_position_m.shape != (3,) or target_euler_xyz_rad.shape != (3,):
            raise ValueError("Pink IK target must contain 3D position and Euler rotation")
        if current_joint_rad.shape != (6,):
            raise ValueError("Current Piper joints must contain six angles")
        if not (
            np.isfinite(target_position_m).all()
            and np.isfinite(target_euler_xyz_rad).all()
            and np.isfinite(current_joint_rad).all()
        ):
            raise ValueError("Pink IK input contains NaN or Inf")

        configuration = self._configuration(current_joint_rad)
        current_transform = configuration.get_transform_frame_to_world(self.frame_name)
        target_transform = self._target_se3(
            target_position_m, target_euler_xyz_rad, self.pin
        )
        self.frame_task.set_target(target_transform)
        self.posture_task.set_target(configuration.q.copy())
        try:
            velocity = self.pink.solve_ik(
                configuration,
                tasks=[self.frame_task, self.posture_task],
                dt=self.dt,
                solver=self.solver,
                damping=self.qpsolver_damping,
                limits=self.limits,
                safety_break=False,
            )
        except Exception as error:
            diagnostics = {
                "solver": "pink_pinocchio_qp",
                "target_model_tcp": {
                    "position_m": target_position_m,
                    "euler_xyz_rad": target_euler_xyz_rad,
                    "euler_xyz_deg": np.degrees(target_euler_xyz_rad),
                },
                "current_joint_degrees": np.degrees(current_joint_rad),
                "current_position_m": current_transform.translation.copy(),
                "current_quat_xyzw": Rotation.from_matrix(
                    current_transform.rotation
                ).as_quat(),
                "solver_name": self.solver,
                "solve_seconds": time.monotonic() - started,
                "error": str(error),
            }
            raise HostIKError("Pink IK solve failed", diagnostics) from error

        q_next = configuration.integrate(velocity, self.dt)
        requested_delta_rad = q_next[:6] - configuration.q[:6]
        requested_max_step_deg = float(np.max(np.abs(np.degrees(requested_delta_rad))))
        if requested_max_step_deg > self.max_joint_step_deg:
            scale = self.max_joint_step_deg / requested_max_step_deg
        else:
            scale = 1.0
        q_cmd_arm = configuration.q[:6] + scale * requested_delta_rad
        unclipped_q_cmd_arm = q_cmd_arm.copy()
        q_cmd_arm = np.clip(q_cmd_arm, self.lower_rad, self.upper_rad)
        q_cmd = configuration.q.copy()
        q_cmd[:6] = q_cmd_arm
        command_configuration = self.pink.Configuration(self.model, self.data, q_cmd)
        command_transform = command_configuration.get_transform_frame_to_world(self.frame_name)
        position_error = target_position_m - current_transform.translation
        rotation_error = (
            Rotation.from_euler("xyz", target_euler_xyz_rad)
            * Rotation.from_matrix(current_transform.rotation).inv()
        ).as_rotvec()
        command_position_error = target_position_m - command_transform.translation
        command_rotation_error = (
            Rotation.from_euler("xyz", target_euler_xyz_rad)
            * Rotation.from_matrix(command_transform.rotation).inv()
        ).as_rotvec()
        command_deg = np.degrees(q_cmd_arm)
        current_deg = np.degrees(configuration.q[:6])
        clipped_by_limits = bool(np.max(np.abs(q_cmd_arm - unclipped_q_cmd_arm)) > 1e-12)
        command_limit_margin_deg = float(
            np.min(np.minimum(command_deg - JOINT_LOWER_DEG, JOINT_UPPER_DEG - command_deg))
        )
        return {
            "solver": "pink_pinocchio_qp",
            "target_model_tcp": {
                "position_m": target_position_m,
                "euler_xyz_rad": target_euler_xyz_rad,
                "euler_xyz_deg": np.degrees(target_euler_xyz_rad),
            },
            "current_joint_degrees": current_deg,
            "joint_lower_degrees": JOINT_LOWER_DEG,
            "joint_upper_degrees": JOINT_UPPER_DEG,
            "current_position_m": current_transform.translation.copy(),
            "current_quat_xyzw": Rotation.from_matrix(current_transform.rotation).as_quat(),
            "position_error_m": position_error,
            "rotation_error_axis_angle_rad": rotation_error,
            "pink_velocity": velocity,
            "selected_joint_degrees": command_deg,
            "selected_joint_radians": q_cmd_arm,
            "ik_solution_joint_degrees": np.degrees(q_next[:6]),
            "ik_solution_joint_radians": q_next[:6],
            "selected_score": float(
                np.linalg.norm(position_error) + self.orientation_cost * np.linalg.norm(rotation_error)
            ),
            "selected": {
                "joint_degrees": command_deg,
                "position_error_m": float(np.linalg.norm(command_position_error)),
                "rotation_error_rad": float(np.linalg.norm(command_rotation_error)),
                "minimum_limit_margin_deg": command_limit_margin_deg,
                "maximum_step_from_current_deg": float(np.max(np.abs(command_deg - current_deg))),
                "accepted": True,
                "rejection_reason": None,
            },
            "exact_solution_available": True,
            "pose_projected": False,
            "pose_projection_reason": None,
            "joint_step_limited": scale < 1.0,
            "joint_step_scale": float(scale),
            "requested_max_joint_step_deg": requested_max_step_deg,
            "commanded_max_joint_step_deg": float(np.max(np.abs(command_deg - current_deg))),
            "commanded_minimum_limit_margin_deg": command_limit_margin_deg,
            "commanded_position_error_m": float(np.linalg.norm(command_position_error)),
            "commanded_rotation_error_rad": float(np.linalg.norm(command_rotation_error)),
            "joint_limit_clipped": clipped_by_limits,
            "solver_name": self.solver,
            "dt": self.dt,
            "solve_seconds": time.monotonic() - started,
        }
