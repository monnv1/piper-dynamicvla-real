from __future__ import annotations

import argparse
import dataclasses
import logging
import time
import uuid
from pathlib import Path

import numpy as np

from deploy.common.event_log import EventLog
from deploy.common.latest import FrameBuffer
from deploy.common.video_recorder import AsyncVideoWriter, RecordingFrameBuffer
from deploy.config import DeployConfig, load_config
from deploy.control.safety_filter import SafetyFilter, SafetyViolation
from deploy.devices.factory import create_camera
from deploy.kinematics import HostIKError
from deploy.devices.piper_robot import PiperRobot
from deploy.policy.action_scheduler import ActionScheduler, FixedRateGate
from deploy.policy.inference_worker import DynamicVLAWorker
from deploy.policy.observation_builder import ObservationBuilder
from deploy.targets import TRAINING_START_DEG, TRAINING_START_GRIPPER_M


class DeploymentRuntime:
    def __init__(self, config: DeployConfig, confirm_motion: bool) -> None:
        self.config = config
        self.confirm_motion = confirm_motion
        self.episode_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.log = EventLog(config.runtime.output_dir, self.episode_id)
        self.video_recorders = {}
        self.buffers = {}
        for name, camera_config in config.cameras.items():
            if not camera_config.enabled:
                continue
            if config.runtime.record_video:
                recorder = AsyncVideoWriter(
                    name,
                    self.log.directory / "videos" / f"{name}.mp4",
                    config.runtime.video_fps,
                )
                self.video_recorders[name] = recorder
                self.buffers[name] = RecordingFrameBuffer(recorder)
            else:
                self.buffers[name] = FrameBuffer()
        self.cameras = [
            create_camera(name, config.cameras[name], buffer)
            for name, buffer in self.buffers.items()
        ]
        self.robot = PiperRobot(config.robot)
        self.worker = DynamicVLAWorker(config.model)
        self.builder = ObservationBuilder(
            self.buffers,
            config.runtime.history_indices,
            config.runtime.camera_sync_tolerance_ms,
        )
        self.scheduler = ActionScheduler(20, config.safety.max_action_age_ms)
        self.safety = SafetyFilter(config.safety)

    def run(self) -> None:
        execute = self.config.runtime.mode == "execute"
        if execute and not (self.confirm_motion and self.config.robot.auto_enable):
            raise RuntimeError(
                "execute mode requires robot.auto_enable=true and --confirm-motion"
            )
        self.log.write("runtime_start", config=dataclasses.asdict(self.config))
        try:
            for recorder in self.video_recorders.values():
                recorder.start()
            if self.video_recorders:
                self.log.write(
                    "video_recording_start",
                    videos={
                        name: str(recorder.path)
                        for name, recorder in self.video_recorders.items()
                    },
                    fps=self.config.runtime.video_fps,
                )
            self.robot.start()
            for camera in self.cameras:
                camera.start()
            self.worker.start()
            self._wait_until_ready(self.config.runtime.startup_timeout_s)
            self.scheduler.reset(self.episode_id)
            if execute:
                self.robot.enable_motion()
                self.log.write("motion_enabled")
            self._control_loop(execute)
            if execute and self.config.runtime.return_to_training_start_on_normal_exit:
                self.log.write(
                    "return_to_training_start_begin",
                    target_deg=TRAINING_START_DEG,
                    speed_percent=self.config.runtime.return_speed_percent,
                )
                reached = self.robot.move_to_joint_target(
                    TRAINING_START_DEG,
                    speed_percent=self.config.runtime.return_speed_percent,
                    timeout_s=self.config.runtime.return_timeout_s,
                    workspace_min_m=self.config.safety.workspace_min_m,
                    workspace_max_m=self.config.safety.workspace_max_m,
                    gripper_target_m=TRAINING_START_GRIPPER_M,
                )
                self.log.write(
                    "return_to_training_start_complete",
                    reached_deg=reached,
                    gripper_target_m=TRAINING_START_GRIPPER_M,
                )
        finally:
            self.worker.stop()
            for camera in self.cameras:
                camera.stop()
            video_stats = {
                name: dataclasses.asdict(recorder.stop())
                for name, recorder in self.video_recorders.items()
            }
            for name, recorder in self.video_recorders.items():
                if recorder.error is not None:
                    self.log.write(
                        "video_recording_error",
                        camera=name,
                        error=str(recorder.error),
                    )
            if video_stats:
                self.log.write("video_recording_stop", videos=video_stats)
            self.robot.stop()
            self.log.write(
                "runtime_stop", scheduler=dataclasses.asdict(self.scheduler.stats)
            )
            self.log.close()

    def _wait_until_ready(self, timeout_seconds: float = 300.0) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            self._raise_worker_errors()
            cameras_ready = all(
                buffer.latest() is not None for buffer in self.buffers.values()
            )
            robot_ready = self.robot.latest_state() is not None
            if cameras_ready and robot_ready and self.worker.ready.is_set():
                self.log.write("devices_ready")
                return
            time.sleep(0.05)
        raise TimeoutError("Timed out waiting for cameras, Piper, or model")

    def _raise_worker_errors(self) -> None:
        if self.worker.error is not None:
            raise RuntimeError("Inference worker failed") from self.worker.error
        if self.robot.error is not None:
            raise RuntimeError("Piper feedback worker failed") from self.robot.error
        for camera in self.cameras:
            if camera.error is not None:
                raise RuntimeError(f"Camera {camera.name} failed") from camera.error

    def _control_loop(self, execute: bool) -> None:
        period = 1.0 / self.config.runtime.control_hz
        sensor_timeout_ns = int(self.config.runtime.sensor_timeout_ms * 1_000_000)
        index = 0
        next_tick = time.monotonic()
        last_executed_action_ns = None
        previous_command = None
        stale_hold_ns = int(self.config.safety.stale_action_hold_ms * 1_000_000)
        stale_action_hold_logged = False
        execute_deadline_ns = None
        sequential_inference = not self.config.runtime.continuous_inference
        point_to_point_execution = (
            self.config.runtime.action_execution_mode == "point_to_point"
        )
        timed_action_gate = FixedRateGate(self.config.runtime.action_hz)
        inference_pending = False
        awaiting_action_completion = False
        action_completion_settle_count = 0
        if execute and self.config.runtime.max_execute_seconds > 0:
            execute_deadline_ns = time.monotonic_ns() + int(
                self.config.runtime.max_execute_seconds * 1_000_000_000
            )
        while True:
            self._raise_worker_errors()
            now_ns = time.monotonic_ns()
            if execute_deadline_ns is not None and now_ns >= execute_deadline_ns:
                self.log.write(
                    "execute_time_limit",
                    index=index,
                    seconds=self.config.runtime.max_execute_seconds,
                )
                return
            state = self.robot.latest_state()
            if state is None or now_ns - state.host_timestamp_ns > sensor_timeout_ns:
                raise RuntimeError("Piper feedback is stale")
            expected_mode_feed = self.robot.expected_mode_feed()
            if execute and (
                state.ctrl_mode != 0x01
                or state.mode_feed != expected_mode_feed
                or state.arm_status != 0x00
            ):
                self.log.write(
                    "robot_status_fault",
                    index=index,
                    ctrl_mode=state.ctrl_mode,
                    mode_feed=state.mode_feed,
                    arm_status=state.arm_status,
                    motion_status=state.motion_status,
                    err_code=state.err_code,
                    joint_limit_flags=state.joint_limit_flags,
                    last_command=previous_command,
                    robot_current={
                        "joint_degrees": np.degrees(state.joint_radians),
                        "joint_radians": state.joint_radians,
                        "tcp_model_position_m": state.position_m,
                        "tcp_model_euler_xyz_rad": state.euler_xyz_rad,
                        "gripper_m": state.gripper_m,
                    },
                    limit_joints=[
                        index + 1
                        for index, exceeded in enumerate(state.joint_limit_flags)
                        if exceeded
                    ],
                    joint_degrees=np.degrees(state.joint_radians),
                    state=state.model_vector(),
                )
                raise RuntimeError(
                    "Piper left expected CAN control mode: "
                    f"ctrl_mode=0x{state.ctrl_mode:02X}, "
                    f"mode_feed=0x{state.mode_feed:02X}, "
                    f"arm_status=0x{state.arm_status:02X}"
                )

            previous_command_feedback = None
            if previous_command is not None:
                previous_command_feedback = {
                    "command_index": previous_command["index"],
                    "elapsed_ms": (now_ns - previous_command["timestamp_ns"])
                    / 1_000_000.0,
                    "planned_tcp_model": previous_command["planned_tcp_model"],
                    "ik_realized_joint_degrees": np.degrees(state.joint_radians),
                    "realized_tcp_model": {
                        "position_m": state.position_m,
                        "euler_xyz_rad": state.euler_xyz_rad,
                        "gripper_m": state.gripper_m,
                    },
                    "robot_status": {
                        "ctrl_mode": state.ctrl_mode,
                        "mode_feed": state.mode_feed,
                        "arm_status": state.arm_status,
                        "motion_status": state.motion_status,
                        "err_code": state.err_code,
                        "joint_limit_flags": state.joint_limit_flags,
                    },
                }

            if (
                sequential_inference
                and point_to_point_execution
                and execute
                and awaiting_action_completion
            ):
                selected_joint_degrees = previous_command["selected_joint_degrees"]
                current_joint_degrees = np.degrees(state.joint_radians)
                if selected_joint_degrees is None:
                    target_joint_degrees = None
                    maximum_joint_error_deg = None
                    action_completed = state.motion_status == 0x00
                else:
                    target_joint_degrees = np.asarray(
                        selected_joint_degrees, dtype=np.float64
                    )
                    maximum_joint_error_deg = float(
                        np.max(np.abs(target_joint_degrees - current_joint_degrees))
                    )
                    action_completed = (
                        state.motion_status == 0x00
                        and maximum_joint_error_deg
                        <= self.config.runtime.action_completion_joint_tolerance_deg
                    )
                if action_completed:
                    action_completion_settle_count += 1
                else:
                    action_completion_settle_count = 0
                elapsed_s = (
                    now_ns - previous_command["timestamp_ns"]
                ) / 1_000_000_000.0
                if (
                    action_completion_settle_count
                    >= self.config.runtime.action_completion_settle_cycles
                ):
                    self.log.write(
                        "action_completed",
                        index=index,
                        command_index=previous_command["index"],
                        elapsed_s=elapsed_s,
                        motion_status=state.motion_status,
                        maximum_joint_error_deg=maximum_joint_error_deg,
                        target_joint_degrees=target_joint_degrees,
                        actual_joint_degrees=current_joint_degrees,
                    )
                    awaiting_action_completion = False
                    action_completion_settle_count = 0
                elif elapsed_s > self.config.runtime.action_completion_timeout_s:
                    self.log.write(
                        "action_completion_timeout",
                        index=index,
                        command_index=previous_command["index"],
                        elapsed_s=elapsed_s,
                        motion_status=state.motion_status,
                        maximum_joint_error_deg=maximum_joint_error_deg,
                        target_joint_degrees=target_joint_degrees,
                        actual_joint_degrees=current_joint_degrees,
                    )
                    raise RuntimeError(
                        "Piper did not complete the sequential action within "
                        f"{self.config.runtime.action_completion_timeout_s:.1f}s"
                    )

            should_submit_observation = not sequential_inference or (
                not inference_pending
                and not awaiting_action_completion
                and not self.scheduler.has_pending_actions()
            )
            if should_submit_observation:
                try:
                    observation = self.builder.build(
                        self.episode_id,
                        index,
                        state,
                        self.config.model.task,
                    )
                except RuntimeError as error:
                    self.log.write(
                        "observation_skipped", index=index, reason=str(error)
                    )
                else:
                    self.worker.submit(observation)
                    if sequential_inference:
                        inference_pending = True
                        self.log.write("inference_requested", index=index)

            chunk = self.worker.get_result()
            if chunk is not None:
                if sequential_inference:
                    inference_pending = False
                accepted = self.scheduler.submit(
                    chunk,
                    index,
                    now_ns,
                    reanchor=sequential_inference,
                    max_actions=(
                        self.config.runtime.max_trusted_action_steps
                        if sequential_inference
                        else None
                    ),
                )
                if accepted and sequential_inference and not point_to_point_execution:
                    timed_action_gate.arm(now_ns)
                self.log.write(
                    "action_chunk",
                    index=index,
                    source_index=chunk.observation_index,
                    inference_seconds=chunk.inference_seconds,
                    accepted=accepted,
                    actions=chunk.actions,
                    source_state=chunk.source_state,
                    raw_delta_to_source_state=(
                        None
                        if chunk.source_state is None
                        else chunk.actions[
                            :,
                            : min(
                                chunk.actions.shape[-1] - 1,
                                chunk.source_state.shape[-1],
                            ),
                        ]
                        - chunk.source_state[
                            : min(
                                chunk.actions.shape[-1] - 1,
                                chunk.source_state.shape[-1],
                            )
                        ]
                    ),
                )

            dispatch_timing = None
            if sequential_inference and point_to_point_execution:
                scheduled = (
                    self.scheduler.pop_next()
                    if not awaiting_action_completion
                    else None
                )
            elif sequential_inference:
                if timed_action_gate.ready(now_ns):
                    scheduled = self.scheduler.pop_next()
                    if scheduled is not None:
                        dispatch_timing = timed_action_gate.consume(now_ns)
                else:
                    scheduled = None
            else:
                scheduled = self.scheduler.pop(index)
            if scheduled is not None:
                stale_action_hold_logged = False
                command_details = self.robot.describe_action(scheduled.action)
                try:
                    safe_action = self.safety.apply(scheduled.action, state)
                except SafetyViolation as error:
                    self.log.write(
                        "safety_reject",
                        index=index,
                        reason=str(error),
                        raw_action=scheduled.action,
                        planned_command=command_details,
                        robot_current={
                            "joint_degrees": np.degrees(state.joint_radians),
                            "joint_radians": state.joint_radians,
                            "tcp_model_position_m": state.position_m,
                            "tcp_model_euler_xyz_rad": state.euler_xyz_rad,
                            "gripper_m": state.gripper_m,
                        },
                        previous_command_feedback=previous_command_feedback,
                    )
                    if execute:
                        raise
                else:
                    command_details = self.robot.describe_action(safe_action)
                    ik_diagnostics = None
                    try:
                        ik_diagnostics = self.robot.preview_action(safe_action)
                    except HostIKError as error:
                        self.log.write(
                            "host_ik_reject",
                            index=index,
                            source_index=scheduled.source_observation_index,
                            executed=execute,
                            diagnostics=error.diagnostics,
                        )
                        if execute:
                            raise
                    if ik_diagnostics is not None and ik_diagnostics.get(
                        "pose_projected"
                    ):
                        self.log.write(
                            "pose_projected",
                            index=index,
                            source_index=scheduled.source_observation_index,
                            executed=execute,
                            exact_solution_available=ik_diagnostics.get(
                                "exact_solution_available"
                            ),
                            selected_joint_degrees=ik_diagnostics[
                                "selected_joint_degrees"
                            ],
                            selected=ik_diagnostics["selected"],
                            commanded_position_error_m=ik_diagnostics[
                                "commanded_position_error_m"
                            ],
                            commanded_rotation_error_rad=ik_diagnostics[
                                "commanded_rotation_error_rad"
                            ],
                        )
                    if ik_diagnostics is not None and ik_diagnostics.get(
                        "joint_step_limited"
                    ):
                        self.log.write(
                            "joint_step_limited",
                            index=index,
                            source_index=scheduled.source_observation_index,
                            executed=execute,
                            scale=ik_diagnostics["joint_step_scale"],
                            requested_max_joint_step_deg=ik_diagnostics[
                                "requested_max_joint_step_deg"
                            ],
                            commanded_max_joint_step_deg=ik_diagnostics[
                                "commanded_max_joint_step_deg"
                            ],
                            ik_solution_joint_degrees=ik_diagnostics[
                                "ik_solution_joint_degrees"
                            ],
                            commanded_joint_degrees=ik_diagnostics[
                                "selected_joint_degrees"
                            ],
                            commanded_position_error_m=ik_diagnostics[
                                "commanded_position_error_m"
                            ],
                            commanded_rotation_error_rad=ik_diagnostics[
                                "commanded_rotation_error_rad"
                            ],
                        )
                    self.log.write(
                        "action",
                        index=index,
                        source_index=scheduled.source_observation_index,
                        raw=scheduled.action,
                        safe=safe_action,
                        source_state=scheduled.source_state,
                        raw_delta_to_source_state=(
                            None
                            if scheduled.source_state is None
                            else scheduled.action[
                                : min(
                                    scheduled.action.shape[-1] - 1,
                                    scheduled.source_state.shape[-1],
                                )
                            ]
                            - scheduled.source_state[
                                : min(
                                    scheduled.action.shape[-1] - 1,
                                    scheduled.source_state.shape[-1],
                                )
                            ]
                        ),
                        executed=execute,
                        state=state.model_vector(),
                        planned_command=command_details,
                        host_ik=ik_diagnostics,
                        robot_current={
                            "joint_degrees": np.degrees(state.joint_radians),
                            "joint_radians": state.joint_radians,
                            "tcp_model_position_m": state.position_m,
                            "tcp_model_euler_xyz_rad": state.euler_xyz_rad,
                            "gripper_m": state.gripper_m,
                        },
                        previous_command_feedback=previous_command_feedback,
                        robot_status={
                            "ctrl_mode": state.ctrl_mode,
                            "mode_feed": state.mode_feed,
                            "arm_status": state.arm_status,
                            "motion_status": state.motion_status,
                        },
                        action_execution_mode=self.config.runtime.action_execution_mode,
                        dispatch_timing=(
                            None
                            if dispatch_timing is None
                            else {
                                "target_ns": dispatch_timing.target_ns,
                                "actual_ns": dispatch_timing.actual_ns,
                                "lateness_ms": dispatch_timing.lateness_ns
                                / 1_000_000.0,
                                "skipped_intervals": dispatch_timing.skipped_intervals,
                            }
                        ),
                    )
                    if execute:
                        command_result = self.robot.command_action(
                            safe_action, prepared=ik_diagnostics
                        )
                        self.log.write(
                            "host_ik_solution",
                            index=index,
                            diagnostics=command_result,
                        )
                        last_executed_action_ns = time.monotonic_ns()
                        previous_command = {
                            "index": index,
                            "timestamp_ns": last_executed_action_ns,
                            "selected_joint_degrees": command_result.get(
                                "selected_joint_degrees"
                            ),
                            "planned_tcp_model": command_details["model_tcp"],
                        }
                        if sequential_inference and point_to_point_execution:
                            awaiting_action_completion = True
                            action_completion_settle_count = 0
            elif (
                execute
                and last_executed_action_ns is not None
                and now_ns - last_executed_action_ns > stale_hold_ns
            ):
                if self.config.safety.hold_on_stale_action:
                    if not stale_action_hold_logged:
                        self.log.write(
                            "action_stream_stale_hold",
                            index=index,
                            elapsed_ms=(now_ns - last_executed_action_ns) / 1_000_000.0,
                            last_command=previous_command,
                            robot_current={
                                "joint_degrees": np.degrees(state.joint_radians),
                                "joint_radians": state.joint_radians,
                                "tcp_model_position_m": state.position_m,
                                "tcp_model_euler_xyz_rad": state.euler_xyz_rad,
                                "gripper_m": state.gripper_m,
                            },
                        )
                        stale_action_hold_logged = True
                else:
                    raise RuntimeError("Action stream is stale; holding Piper position")

            index += 1
            next_tick += period
            sleep_seconds = next_tick - time.monotonic()
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            else:
                self.log.write("control_overrun", index=index, seconds=-sleep_seconds)
                next_tick = time.monotonic()


def main() -> None:
    parser = argparse.ArgumentParser(description="DynamicVLA real-robot runtime")
    parser.add_argument(
        "--config",
        default="deploy/configs/piper_gemini_d435i.yaml",
    )
    parser.add_argument("--mode", choices=["shadow", "execute"])
    parser.add_argument("--checkpoint")
    parser.add_argument("--task")
    parser.add_argument(
        "--action-execution-mode", choices=["point_to_point", "timed"]
    )
    parser.add_argument("--action-hz", type=float)
    parser.add_argument("--confirm-motion", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.mode:
        config.runtime.mode = args.mode
    if args.checkpoint:
        config.model.checkpoint = args.checkpoint
    if args.task:
        config.model.task = args.task
    if args.action_execution_mode:
        config.runtime.action_execution_mode = args.action_execution_mode
    if args.action_hz is not None:
        if args.action_hz <= 0:
            parser.error("--action-hz must be positive")
        config.runtime.action_hz = args.action_hz
    if (
        config.runtime.action_execution_mode == "timed"
        and config.runtime.action_hz > config.runtime.control_hz
    ):
        parser.error("timed action_hz must be <= runtime.control_hz")
    if not Path(config.model.checkpoint).expanduser().is_dir():
        raise FileNotFoundError(f"Invalid checkpoint: {config.model.checkpoint}")

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    runtime = DeploymentRuntime(config, confirm_motion=args.confirm_motion)
    try:
        runtime.run()
    except KeyboardInterrupt:
        logging.info("Stopped by user")


if __name__ == "__main__":
    main()
