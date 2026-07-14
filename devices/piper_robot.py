from __future__ import annotations

import math
import threading
import time

import numpy as np

from deploy.common.latest import LatestValue
from deploy.common.messages import RobotState
from deploy.config import RobotConfig
from deploy.devices.piper_frames import PiperFrameTransform
from deploy.kinematics import PiperDifferentialIK, PiperHostIK, PiperPinkIK


class PiperRobot:
    """Piper feedback and command adapter.

    Construction and start only read feedback. Motion requires an explicit
    call to enable_motion() followed by command_action().
    """

    def __init__(self, config: RobotConfig) -> None:
        self.config = config
        self.states: LatestValue[RobotState] = LatestValue()
        self._piper = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._motion_enabled = False
        self._ik: PiperHostIK | PiperDifferentialIK | PiperPinkIK | None = None
        self.error: Exception | None = None
        self.frames = PiperFrameTransform(
            config.sdk_to_model_translation_m,
            config.sdk_to_model_euler_xyz_rad,
        )

    def start(self) -> None:
        from piper_sdk import C_PiperForwardKinematics, C_PiperInterface_V2  # type: ignore

        self._piper = C_PiperInterface_V2(
            can_name=self.config.can_interface,
            judge_flag=self.config.official_can_adapter,
            can_auto_init=True,
            dh_is_offset=self.config.dh_is_offset,
            start_sdk_joint_limit=True,
            start_sdk_gripper_limit=True,
        )
        self._piper.ConnectPort(piper_init=False, start_thread=True)
        if self.config.feedback_pose_source == "fk":
            self._piper.EnableFkCal()
        if self.config.control_backend == "host_ik_move_j":
            self._ik = PiperHostIK(
                C_PiperForwardKinematics(self.config.dh_is_offset),
                self.config.ik_position_tolerance_m,
                self.config.ik_rotation_tolerance_rad,
                self.config.ik_max_joint_step_deg,
                self.config.ik_min_joint_limit_margin_deg,
                self.config.ik_max_nfev,
                self.config.ik_allow_pose_projection,
                self.config.ik_projection_joint_limit_margin_deg,
                self.config.ik_projection_max_position_error_m,
                self.config.ik_projection_max_rotation_error_rad,
                self.config.ik_projection_position_weight,
                self.config.ik_projection_rotation_weight,
            )
        elif self.config.control_backend == "host_diff_ik_move_j":
            self._ik = PiperDifferentialIK(
                C_PiperForwardKinematics(self.config.dh_is_offset),
                self.config.ik_max_joint_step_deg,
                self.config.ik_min_joint_limit_margin_deg,
                self.config.diff_ik_lambda,
                self.config.diff_ik_finite_difference_eps_rad,
                self.config.diff_ik_position_gain,
                self.config.diff_ik_rotation_gain,
            )
        elif self.config.control_backend == "host_pink_ik_move_j":
            self._ik = PiperPinkIK(
                self.config.pink_urdf_path,
                self.config.pink_frame_name,
                self.config.pink_parent_frame_name,
                tuple(self.config.sdk_to_model_translation_m),
                self.config.pink_solver,
                self.config.pink_dt,
                self.config.pink_position_cost,
                self.config.pink_orientation_cost,
                self.config.pink_posture_cost,
                self.config.pink_lm_damping,
                self.config.pink_qpsolver_damping,
                self.config.ik_max_joint_step_deg,
                self.config.ik_min_joint_limit_margin_deg,
            )
        self._thread = threading.Thread(
            target=self._feedback_loop,
            name="piper-feedback",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._piper is not None:
            if self._motion_enabled:
                self._hold_current_position()
            if self.config.feedback_pose_source == "fk":
                try:
                    self._piper.DisableFkCal()
                except Exception:
                    pass
            self._piper.DisconnectPort()

    def latest_state(self) -> RobotState | None:
        return self.states.get()

    def enable_motion(self) -> None:
        if self._piper is None:
            raise RuntimeError("Piper is not connected")

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if self._piper.EnablePiper():
                break
            time.sleep(0.05)
        else:
            raise TimeoutError(
                "Timed out waiting for all Piper motors to enable: "
                f"{self._piper.GetArmEnableStatus()}"
            )

        if self.config.control_backend in {"host_ik_move_j", "host_diff_ik_move_j", "host_pink_ik_move_j"}:
            mode_deadline = time.monotonic() + 3.0
            status_code = ctrl_mode = mode_feed = -1
            while time.monotonic() < mode_deadline:
                self._hold_current_position()
                status_wrapper = self._piper.GetArmStatus()
                status = getattr(status_wrapper, "arm_status", status_wrapper)
                status_code = int(getattr(status, "arm_status", -1))
                ctrl_mode = int(getattr(status, "ctrl_mode", -1))
                mode_feed = int(getattr(status, "mode_feed", -1))
                if status_code == 0x00 and ctrl_mode == 0x01 and mode_feed == 0x01:
                    self._motion_enabled = True
                    return
                if status_code not in {-1, 0x00, 0x04}:
                    break
                time.sleep(0.05)
            raise TimeoutError(
                "Timed out entering CAN MOVE J host-IK control: "
                f"ctrl_mode=0x{ctrl_mode:02X}, mode_feed=0x{mode_feed:02X}, "
                f"arm_status=0x{status_code:02X}"
            )

        # EndPoseCtrl updates a persistent three-frame target register. If an
        # old process left a distant target there, switching MOVE J -> MOVE P
        # activates it immediately and can raise target_joint_limit before the
        # new action is sent. Seed the register with measured feedback first.
        mode_deadline = time.monotonic() + 3.0
        status_code = ctrl_mode = mode_feed = -1
        while time.monotonic() < mode_deadline:
            current_pose_units = self._read_sdk_pose_units()
            # Preload before mode switching to replace a stale target, then
            # resend after 0x151 because Piper's official examples stream both
            # the mode and pose commands instead of treating 0x151 as reliable.
            self._piper.EndPoseCtrl(*current_pose_units)
            self._piper.MotionCtrl_2(
                0x01,
                0x00,
                int(self.config.command_speed_percent),
                0x00,
            )
            self._piper.EndPoseCtrl(*current_pose_units)
            time.sleep(0.1)
            status_wrapper = self._piper.GetArmStatus()
            status = getattr(status_wrapper, "arm_status", status_wrapper)
            status_code = int(getattr(status, "arm_status", -1))
            ctrl_mode = int(getattr(status, "ctrl_mode", -1))
            mode_feed = int(getattr(status, "mode_feed", -1))
            if status_code == 0x00 and ctrl_mode == 0x01 and mode_feed == 0x00:
                break
            if status_code == 0x04:
                self._hold_current_position()
                time.sleep(0.2)
            elif status_code not in {-1, 0x00}:
                break
        else:
            self._hold_current_position()
            raise TimeoutError(
                "Timed out entering CAN MOVE P control: "
                f"ctrl_mode=0x{ctrl_mode:02X}, mode_feed=0x{mode_feed:02X}, "
                f"arm_status=0x{status_code:02X}"
            )
        if status_code != 0x00 or ctrl_mode != 0x01 or mode_feed != 0x00:
            self._hold_current_position()
            raise RuntimeError(
                "Piper rejected CAN MOVE P control: "
                f"ctrl_mode=0x{ctrl_mode:02X}, mode_feed=0x{mode_feed:02X}, "
                f"arm_status=0x{status_code:02X}"
            )
        self._motion_enabled = True

    def expected_mode_feed(self) -> int:
        return 0x01 if self.config.control_backend in {"host_ik_move_j", "host_diff_ik_move_j", "host_pink_ik_move_j"} else 0x00

    def command_action(self, action: np.ndarray, prepared=None) -> dict[str, object]:
        """Send [x,y,z,rx,ry,rz,gripper] in meters/radians/meters."""
        if not self._motion_enabled or self._piper is None:
            raise RuntimeError("Piper motion is not explicitly enabled")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (7,):
            raise ValueError(f"Expected a 7D Euler action, got {action.shape}")

        sdk_position, sdk_euler = self.frames.model_to_sdk_pose(
            action[:3], action[3:6]
        )
        position_units = np.rint(sdk_position * 1_000_000.0).astype(int)
        rotation_units = np.rint(np.degrees(sdk_euler) * 1_000.0).astype(int)
        gripper_units = int(round(float(action[6]) * 1_000_000.0))
        if self.config.control_backend in {"host_ik_move_j", "host_diff_ik_move_j", "host_pink_ik_move_j"}:
            diagnostics = prepared or self.preview_action(action)
            target_deg = np.asarray(
                diagnostics["selected_joint_degrees"], dtype=np.float64
            )
            target_units = np.rint(target_deg * 1000.0).astype(int)
            self._piper.MotionCtrl_2(
                0x01, 0x01, int(self.config.command_speed_percent), 0x00
            )
            self._piper.JointCtrl(*target_units.tolist())
            if self.config.command_gripper:
                self._piper.GripperCtrl(gripper_units, 1000, 0x01, 0)
            diagnostics["backend"] = self.config.control_backend
            return diagnostics

        self._piper.MotionCtrl_2(
            0x01,
            0x00,
            int(self.config.command_speed_percent),
            0x00,
        )
        self._piper.EndPoseCtrl(*position_units.tolist(), *rotation_units.tolist())
        if self.config.command_gripper:
            self._piper.GripperCtrl(gripper_units, 1000, 0x01, 0)
        return {"backend": "firmware_move_p"}

    def preview_action(self, action: np.ndarray) -> dict[str, object]:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (7,):
            raise ValueError(f"Expected a 7D Euler action, got {action.shape}")
        if self.config.control_backend not in {"host_ik_move_j", "host_diff_ik_move_j", "host_pink_ik_move_j"}:
            return {"backend": "firmware_move_p"}
        if self._ik is None:
            raise RuntimeError("Host IK was not initialized")
        state = self.latest_state()
        if state is None:
            raise RuntimeError("Piper state is unavailable for host IK")
        if self.config.control_backend == "host_pink_ik_move_j":
            diagnostics = self._ik.solve(
                action[:3],
                action[3:6],
                state.joint_radians,
            )
        else:
            sdk_position, sdk_euler = self.frames.model_to_sdk_pose(
                action[:3], action[3:6]
            )
            diagnostics = self._ik.solve(
                sdk_position,
                sdk_euler,
                state.joint_radians,
            )
        diagnostics["backend"] = self.config.control_backend
        return diagnostics

    def describe_action(self, action: np.ndarray) -> dict[str, object]:
        """Return exact known model- and SDK-frame Cartesian command values."""
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (7,):
            raise ValueError(f"Expected a 7D Euler action, got {action.shape}")
        sdk_position, sdk_euler = self.frames.model_to_sdk_pose(
            action[:3], action[3:6]
        )
        return {
            "control_backend": self.config.control_backend,
            "model_tcp": {
                "position_m": action[:3],
                "euler_xyz_rad": action[3:6],
                "gripper_m": float(action[6]),
            },
            "sdk_link6": {
                "position_m": sdk_position,
                "position_mm": sdk_position * 1000.0,
                "euler_xyz_rad": sdk_euler,
                "euler_xyz_deg": np.degrees(sdk_euler),
            },
            "firmware_ik_target_joint_degrees": None,
            "firmware_ik_target_available": False,
            "firmware_ik_target_note": (
                "Piper EndPoseCtrl does not publish its internal IK joint target; "
                "use following-cycle realized joint feedback."
            ),
        }

    def _fk_sdk_pose(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self._piper is None:
            return None
        fk = np.asarray(self._piper.GetFK("feedback"), dtype=np.float64)
        if fk.shape != (6, 6) or not np.isfinite(fk).all() or not np.any(fk):
            return None
        sdk_pose = fk[-1]
        return sdk_pose[:3] / 1000.0, np.radians(sdk_pose[3:])

    def _select_feedback_model_pose(
        self,
        endpose_sdk_position: np.ndarray,
        endpose_sdk_euler: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, str, tuple[np.ndarray, np.ndarray] | None]:
        fk_pose = self._fk_sdk_pose()
        if self.config.feedback_pose_source == "fk":
            if fk_pose is None:
                raise RuntimeError("Piper FK feedback is unavailable")
            sdk_position, sdk_euler = fk_pose
            source = "fk"
        else:
            sdk_position, sdk_euler = endpose_sdk_position, endpose_sdk_euler
            source = "endpose"
        model_position, model_euler = self.frames.sdk_to_model_pose(
            sdk_position, sdk_euler
        )
        return model_position, model_euler, source, fk_pose

    def read_feedback_snapshot(self) -> dict[str, object]:
        """Read current Piper SDK feedback and converted model-TCP feedback."""
        if self._piper is None:
            raise RuntimeError("Piper is not connected")
        joint_message = self._piper.GetArmJointMsgs()
        pose_message = self._piper.GetArmEndPoseMsgs()
        joint = joint_message.joint_state
        pose = pose_message.end_pose
        joint_degrees = np.asarray(
            [
                joint.joint_1,
                joint.joint_2,
                joint.joint_3,
                joint.joint_4,
                joint.joint_5,
                joint.joint_6,
            ],
            dtype=np.float64,
        ) / 1000.0
        sdk_position = np.asarray(
            [pose.X_axis, pose.Y_axis, pose.Z_axis], dtype=np.float64
        ) / 1_000_000.0
        sdk_euler = np.asarray(
            [pose.RX_axis, pose.RY_axis, pose.RZ_axis], dtype=np.float64
        ) * (math.pi / 180_000.0)
        model_position, model_euler, pose_source, fk_pose = self._select_feedback_model_pose(
            sdk_position, sdk_euler
        )
        endpose_model_position, endpose_model_euler = self.frames.sdk_to_model_pose(
            sdk_position, sdk_euler
        )
        if fk_pose is not None:
            fk_model_position, fk_model_euler = self.frames.sdk_to_model_pose(
                fk_pose[0], fk_pose[1]
            )
        else:
            fk_model_position = fk_model_euler = None
        status_wrapper = self._piper.GetArmStatus()
        status = getattr(status_wrapper, "arm_status", status_wrapper)
        return {
            "host_timestamp_ns": time.monotonic_ns(),
            "joint_degrees": joint_degrees,
            "joint_radians": np.radians(joint_degrees),
            "joint_feedback_hz": float(getattr(joint_message, "Hz", 0.0)),
            "sdk_link6": {
                "position_m": sdk_position,
                "position_mm": sdk_position * 1000.0,
                "euler_xyz_rad": sdk_euler,
                "euler_xyz_deg": np.degrees(sdk_euler),
            },
            "model_tcp": {
                "position_m": model_position,
                "euler_xyz_rad": model_euler,
                "source": pose_source,
            },
            "model_tcp_from_end_pose": {
                "position_m": endpose_model_position,
                "euler_xyz_rad": endpose_model_euler,
            },
            "model_tcp_from_fk": None
            if fk_model_position is None
            else {
                "position_m": fk_model_position,
                "euler_xyz_rad": fk_model_euler,
            },
            "status": {
                "feedback_hz": float(getattr(status_wrapper, "Hz", 0.0)),
                "ctrl_mode": int(getattr(status, "ctrl_mode", -1)),
                "arm_status": int(getattr(status, "arm_status", -1)),
                "mode_feed": int(getattr(status, "mode_feed", -1)),
                "motion_status": int(getattr(status, "motion_status", -1)),
                "err_code": int(getattr(status, "err_code", 0)),
            },
        }

    def move_to_joint_target(
        self,
        target_deg: np.ndarray,
        speed_percent: int,
        timeout_s: float,
        workspace_min_m: np.ndarray,
        workspace_max_m: np.ndarray,
        close_gripper: bool = False,
        tolerance_deg: float = 1.5,
    ) -> np.ndarray:
        """Move slowly in MOVE J and leave the final measured pose holding."""
        if not self._motion_enabled or self._piper is None:
            raise RuntimeError("Piper motion is not explicitly enabled")
        target = np.asarray(target_deg, dtype=np.float64)
        if target.shape != (6,) or not np.isfinite(target).all():
            raise ValueError("Joint target must contain six finite degrees")
        if not 1 <= speed_percent <= 10:
            raise ValueError("Joint return speed must be in [1, 10]")

        current = self._read_joint_degrees()
        self._validate_joint_path_workspace(
            current,
            target,
            np.asarray(workspace_min_m, dtype=np.float64),
            np.asarray(workspace_max_m, dtype=np.float64),
        )

        # A rejected MOVE P target leaves arm_status=0x04 latched even though
        # joint feedback and motors remain healthy. Recover only this known,
        # non-emergency Cartesian error by switching to MOVE J at the measured
        # joints. Other status codes still prohibit automatic return motion.
        status_wrapper = self._piper.GetArmStatus()
        status = getattr(status_wrapper, "arm_status", status_wrapper)
        initial_status_code = int(getattr(status, "arm_status", -1))
        if initial_status_code == 0x04:
            self._hold_current_position()
            time.sleep(0.3)
            status_wrapper = self._piper.GetArmStatus()
            status = getattr(status_wrapper, "arm_status", status_wrapper)
            recovered_status_code = int(getattr(status, "arm_status", -1))
            if recovered_status_code != 0x00:
                raise RuntimeError(
                    "MOVE J current-position hold did not clear Piper "
                    f"target_joint_limit: status=0x{recovered_status_code:02X}"
                )
        elif initial_status_code != 0x00:
            raise RuntimeError(
                "Automatic return is prohibited for Piper status "
                f"0x{initial_status_code:02X}"
            )

        if close_gripper:
            self._piper.GripperCtrl(0, 1000, 0x01, 0x00)
            time.sleep(0.2)

        target_units = np.rint(target * 1000.0).astype(int)
        deadline = time.monotonic() + timeout_s
        settled = 0
        while time.monotonic() < deadline:
            status_wrapper = self._piper.GetArmStatus()
            status = getattr(status_wrapper, "arm_status", status_wrapper)
            status_code = int(getattr(status, "arm_status", -1))
            if status_code != 0x00:
                raise RuntimeError(
                    f"Piper status 0x{status_code:02X} while returning to standby"
                )
            self._piper.MotionCtrl_2(0x01, 0x01, speed_percent, 0x00)
            self._piper.JointCtrl(*target_units.tolist())
            current = self._read_joint_degrees()
            if float(np.max(np.abs(target - current))) <= tolerance_deg:
                settled += 1
                if settled >= 5:
                    if close_gripper:
                        self._piper.GripperCtrl(0, 1000, 0x01, 0x00)
                    self._hold_current_position()
                    return current
            else:
                settled = 0
            time.sleep(0.1)
        raise TimeoutError(
            f"Piper did not reach the training start within {timeout_s:.1f}s"
        )

    def _hold_current_position(self) -> None:
        """Replace the active command with a measured-position hold."""
        message = self._piper.GetArmJointMsgs().joint_state
        joint_units = [
            int(message.joint_1),
            int(message.joint_2),
            int(message.joint_3),
            int(message.joint_4),
            int(message.joint_5),
            int(message.joint_6),
        ]
        self._piper.MotionCtrl_2(0x01, 0x01, 1, 0x00)
        self._piper.JointCtrl(*joint_units)
        time.sleep(0.2)

    def _resume_position_control(self) -> None:
        """Safely release a prior track termination at the measured pose."""
        message = self._piper.GetArmJointMsgs().joint_state
        joint_units = [
            int(message.joint_1),
            int(message.joint_2),
            int(message.joint_3),
            int(message.joint_4),
            int(message.joint_5),
            int(message.joint_6),
        ]
        self._piper.MotionCtrl_2(0x01, 0x01, 1, 0x00)
        self._piper.JointCtrl(*joint_units)
        self._piper.MotionCtrl_1(0x00, 0x02, 0x00)
        self._piper.JointCtrl(*joint_units)
        time.sleep(0.2)

    def _feedback_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                joint_message = self._piper.GetArmJointMsgs()
                pose_message = self._piper.GetArmEndPoseMsgs()
                joint = joint_message.joint_state
                pose = pose_message.end_pose
                joint_deg = np.asarray(
                    [
                        joint.joint_1,
                        joint.joint_2,
                        joint.joint_3,
                        joint.joint_4,
                        joint.joint_5,
                        joint.joint_6,
                    ],
                    dtype=np.float64,
                ) / 1_000.0
                sdk_position = np.asarray(
                    [pose.X_axis, pose.Y_axis, pose.Z_axis], dtype=np.float64
                ) / 1_000_000.0
                sdk_euler = np.asarray(
                    [pose.RX_axis, pose.RY_axis, pose.RZ_axis], dtype=np.float64
                ) * (math.pi / 180_000.0)
                try:
                    position, euler, _, _ = self._select_feedback_model_pose(
                        sdk_position, sdk_euler
                    )
                except RuntimeError:
                    time.sleep(0.005)
                    continue
                gripper_m = self._read_gripper_m()
                status_wrapper = self._piper.GetArmStatus()
                status = getattr(status_wrapper, "arm_status", status_wrapper)
                err_status = getattr(status, "err_status", None)
                joint_limit_flags = tuple(
                    bool(getattr(err_status, f"joint_{index}_angle_limit", False))
                    for index in range(1, 7)
                )
                self.states.publish(
                    RobotState(
                        host_timestamp_ns=time.monotonic_ns(),
                        joint_radians=np.radians(joint_deg).astype(np.float32),
                        position_m=position.astype(np.float32),
                        euler_xyz_rad=euler.astype(np.float32),
                        gripper_m=gripper_m,
                        feedback_hz=float(joint_message.Hz),
                        ctrl_mode=int(getattr(status, "ctrl_mode", -1)),
                        arm_status=int(getattr(status, "arm_status", -1)),
                        mode_feed=int(getattr(status, "mode_feed", -1)),
                        motion_status=int(getattr(status, "motion_status", -1)),
                        err_code=int(getattr(status, "err_code", 0)),
                        joint_limit_flags=joint_limit_flags,
                    )
                )
                time.sleep(0.005)
        except Exception as error:
            self.error = error
            self._stop_event.set()

    def _read_gripper_m(self) -> float:
        try:
            message = self._piper.GetArmGripperMsgs()
            state = message.gripper_state
            return float(state.grippers_angle) / 1_000_000.0
        except Exception:
            return 0.0

    def _read_sdk_pose_units(self) -> list[int]:
        pose = self._piper.GetArmEndPoseMsgs().end_pose
        return [
            int(pose.X_axis),
            int(pose.Y_axis),
            int(pose.Z_axis),
            int(pose.RX_axis),
            int(pose.RY_axis),
            int(pose.RZ_axis),
        ]

    def _read_joint_degrees(self) -> np.ndarray:
        joint = self._piper.GetArmJointMsgs().joint_state
        return np.asarray(
            [
                joint.joint_1,
                joint.joint_2,
                joint.joint_3,
                joint.joint_4,
                joint.joint_5,
                joint.joint_6,
            ],
            dtype=np.float64,
        ) / 1000.0

    def _validate_joint_path_workspace(
        self,
        start_deg: np.ndarray,
        target_deg: np.ndarray,
        workspace_min_m: np.ndarray,
        workspace_max_m: np.ndarray,
    ) -> None:
        from piper_sdk import C_PiperForwardKinematics  # type: ignore

        fk = C_PiperForwardKinematics(self.config.dh_is_offset)
        start_rad = np.radians(start_deg)
        target_rad = np.radians(target_deg)
        for index, fraction in enumerate(np.linspace(0.0, 1.0, 51)):
            joints = start_rad + (target_rad - start_rad) * fraction
            sdk_pose = np.asarray(fk.CalFK(joints)[-1], dtype=np.float64)
            model_position, _ = self.frames.sdk_to_model_pose(
                sdk_pose[:3] / 1000.0,
                np.radians(sdk_pose[3:]),
            )
            if np.any(model_position < workspace_min_m) or np.any(
                model_position > workspace_max_m
            ):
                raise RuntimeError(
                    "Training-start MOVE J path leaves the configured workspace "
                    f"at sample {index}: {model_position.tolist()}"
                )
