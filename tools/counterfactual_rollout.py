from __future__ import annotations

import argparse
import dataclasses
import logging
import time
import uuid
from collections import deque
from pathlib import Path

import numpy as np

from deploy.common.event_log import EventLog
from deploy.common.latest import FrameBuffer
from deploy.common.messages import PolicyObservation, RobotState
from deploy.config import DeployConfig, load_config
from deploy.devices.factory import create_camera
from deploy.devices.piper_robot import PiperRobot
from deploy.policy.inference_worker import DynamicVLAWorker


def select_state_history(
    states: deque[np.ndarray], history_indices: list[int]
) -> np.ndarray:
    """Select a clamped state history using ObservationBuilder's semantics."""
    if not states:
        raise ValueError("states must contain at least one item")
    if not history_indices or history_indices[-1] != 0:
        raise ValueError("history_indices must end in 0")
    last_index = len(states) - 1
    selected = [
        states[max(0, min(last_index, last_index + relative_index))]
        for relative_index in history_indices
    ]
    return np.stack(selected, axis=0).astype(np.float32, copy=False)


def select_rollout_action(actions: np.ndarray, action_index: int) -> np.ndarray:
    """Return one absolute action from a model action chunk."""
    actions = np.asarray(actions)
    if actions.ndim != 2:
        raise ValueError(f"actions must have shape [T, D], got {actions.shape}")
    if action_index < 0 or action_index >= actions.shape[0]:
        raise IndexError(
            f"action_index {action_index} out of range for chunk length {actions.shape[0]}"
        )
    return actions[action_index].astype(np.float32, copy=True)


class CounterfactualRollout:
    """Run model-only closed-loop rollout with virtual state feedback.

    This tool intentionally does not call PiperRobot.enable_motion(),
    PiperRobot.command_action(), JointCtrl, or EndPoseCtrl. It reads the initial
    robot state and camera frames, freezes the images, then feeds the previous
    target action back as the next observation.state so we can inspect what the
    policy wants if every target were reached perfectly.
    """

    def __init__(
        self,
        config: DeployConfig,
        steps: int,
        action_index: int,
        inference_timeout_s: float,
    ) -> None:
        if steps <= 0:
            raise ValueError("--steps must be positive")
        if action_index < 0:
            raise ValueError("--action-index must be non-negative")
        if inference_timeout_s <= 0:
            raise ValueError("--inference-timeout-s must be positive")
        self.config = config
        self.steps = steps
        self.action_index = action_index
        self.inference_timeout_s = inference_timeout_s
        self.episode_id = (
            "counterfactual-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        )
        self.log = EventLog(config.runtime.output_dir, self.episode_id)
        self.buffers = {
            name: FrameBuffer()
            for name, camera_config in config.cameras.items()
            if camera_config.enabled
        }
        self.cameras = [
            create_camera(name, config.cameras[name], buffer)
            for name, buffer in self.buffers.items()
        ]
        self.robot = PiperRobot(config.robot)
        self.worker = DynamicVLAWorker(config.model)

    def run(self) -> None:
        self.log.write(
            "counterfactual_start",
            config=dataclasses.asdict(self.config),
            steps=self.steps,
            action_index=self.action_index,
            inference_timeout_s=self.inference_timeout_s,
            read_only=True,
            note=(
                "No robot motion is enabled or commanded. Images are frozen after "
                "startup; observation.state is replaced by the previous selected "
                "target action."
            ),
        )
        try:
            self.robot.start()
            for camera in self.cameras:
                camera.start()
            self.worker.start()
            self._wait_until_ready(self.config.runtime.startup_timeout_s)
            frozen_images = self._freeze_images()
            initial_state = self._latest_robot_state()
            self._rollout(frozen_images, initial_state)
        finally:
            self.worker.stop()
            for camera in self.cameras:
                camera.stop()
            self.robot.stop()
            self.log.write("counterfactual_stop")
            self.log.close()

    def _wait_until_ready(self, timeout_seconds: float) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            self._raise_background_errors()
            cameras_ready = all(buffer.latest() is not None for buffer in self.buffers.values())
            robot_ready = self.robot.latest_state() is not None
            if cameras_ready and robot_ready and self.worker.ready.is_set():
                self.log.write("devices_ready")
                return
            time.sleep(0.05)
        raise TimeoutError("Timed out waiting for cameras, Piper feedback, or model")

    def _raise_background_errors(self) -> None:
        if self.worker.error is not None:
            raise RuntimeError("Inference worker failed") from self.worker.error
        if self.robot.error is not None:
            raise RuntimeError("Piper feedback worker failed") from self.robot.error
        for camera in self.cameras:
            if camera.error is not None:
                raise RuntimeError(f"Camera {camera.name} failed") from camera.error

    def _freeze_images(self) -> dict[str, np.ndarray]:
        images: dict[str, np.ndarray] = {}
        for camera_name, buffer in self.buffers.items():
            frame = buffer.latest()
            if frame is None:
                raise RuntimeError(f"No frame available for {camera_name}")
            images[f"observation.images.{camera_name}"] = np.asarray(frame.rgb).copy()
        self.log.write(
            "images_frozen",
            cameras={
                key: {
                    "shape": value.shape,
                    "dtype": str(value.dtype),
                }
                for key, value in images.items()
            },
        )
        return images

    def _latest_robot_state(self) -> RobotState:
        state = self.robot.latest_state()
        if state is None:
            raise RuntimeError("Piper feedback is unavailable")
        self.log.write(
            "initial_robot_state",
            joint_degrees=np.degrees(state.joint_radians),
            state=state.model_vector(),
            tcp_model_position_m=state.position_m,
            tcp_model_euler_xyz_rad=state.euler_xyz_rad,
            gripper_m=state.gripper_m,
            robot_status={
                "ctrl_mode": state.ctrl_mode,
                "mode_feed": state.mode_feed,
                "arm_status": state.arm_status,
                "motion_status": state.motion_status,
                "err_code": state.err_code,
                "joint_limit_flags": state.joint_limit_flags,
            },
        )
        return state

    def _build_observation(
        self,
        index: int,
        frozen_images: dict[str, np.ndarray],
        virtual_states: deque[np.ndarray],
    ) -> PolicyObservation:
        state_history = select_state_history(
            virtual_states, self.config.runtime.history_indices
        )
        return PolicyObservation(
            episode_id=self.episode_id,
            index=index,
            host_timestamp_ns=time.monotonic_ns(),
            images={
                key: np.stack([image for _ in self.config.runtime.history_indices], axis=0)
                for key, image in frozen_images.items()
            },
            states=state_history,
            task=self.config.model.task,
        )

    def _rollout(
        self, frozen_images: dict[str, np.ndarray], initial_state: RobotState
    ) -> None:
        max_history = max(32, abs(min(self.config.runtime.history_indices)) + 8)
        virtual_states: deque[np.ndarray] = deque(maxlen=max_history)
        virtual_states.append(initial_state.model_vector().astype(np.float32, copy=True))

        for index in range(self.steps):
            self._raise_background_errors()
            observation = self._build_observation(index, frozen_images, virtual_states)
            self.worker.submit(observation)
            chunk = self._wait_for_chunk(index)
            action = select_rollout_action(chunk.actions, self.action_index)
            previous_state = virtual_states[-1]
            virtual_states.append(action)
            action_dimensions = min(previous_state.shape[0], action.shape[0])
            self.log.write(
                "counterfactual_step",
                index=index,
                source_index=chunk.observation_index,
                inference_seconds=chunk.inference_seconds,
                action_index=self.action_index,
                virtual_state_in=previous_state,
                selected_action=action,
                selected_delta=action[:action_dimensions]
                - previous_state[:action_dimensions],
                action_chunk=chunk.actions,
            )

    def _wait_for_chunk(self, observation_index: int):
        deadline = time.monotonic() + self.inference_timeout_s
        while time.monotonic() < deadline:
            self._raise_background_errors()
            chunk = self.worker.get_result()
            if chunk is not None and chunk.observation_index == observation_index:
                return chunk
            time.sleep(0.01)
        raise TimeoutError(
            f"Timed out waiting for action chunk for observation {observation_index}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a read-only counterfactual DynamicVLA rollout. The previous "
            "target action is fed back as the next observation.state."
        )
    )
    parser.add_argument(
        "--config",
        default="deploy/configs/piper_sequential.yaml",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--task")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--action-index", type=int, default=0)
    parser.add_argument(
        "--inference-timeout-s",
        type=float,
        default=None,
        help="Per-step model inference timeout. Defaults to runtime.startup_timeout_s.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.checkpoint:
        config.model.checkpoint = args.checkpoint
    if args.task:
        config.model.task = args.task
    if not Path(config.model.checkpoint).expanduser().is_dir():
        raise FileNotFoundError(f"Invalid checkpoint: {config.model.checkpoint}")

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    rollout = CounterfactualRollout(
        config=config,
        steps=args.steps,
        action_index=args.action_index,
        inference_timeout_s=(
            args.inference_timeout_s
            if args.inference_timeout_s is not None
            else config.runtime.startup_timeout_s
        ),
    )
    try:
        rollout.run()
        logging.info("Counterfactual log: %s", rollout.log.path)
    except KeyboardInterrupt:
        logging.info("Stopped by user")


if __name__ == "__main__":
    main()
