from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path

import numpy as np

from deploy.common.messages import ActionChunk, PolicyObservation
from deploy.config import ModelConfig


LOGGER = logging.getLogger(__name__)


DOM_CHUNK000_DELTA_ACTION_MEAN = np.asarray(
    [0.001576, 0.000079, -0.007462, 0.001280, 0.027347, -0.014355, 0.226501],
    dtype=np.float32,
)
DOM_CHUNK000_DELTA_ACTION_STD = np.asarray(
    [0.035303, 0.053547, 0.045835, 0.030943, 0.080895, 1.894453, 0.974011],
    dtype=np.float32,
)


def fit_feature_dimension(values: np.ndarray, expected_dimension: int) -> np.ndarray:
    """Trim or zero-pad the last axis to the checkpoint feature dimension."""
    current_dimension = values.shape[-1]
    if current_dimension == expected_dimension:
        return values
    if current_dimension > expected_dimension:
        return values[..., :expected_dimension]
    padding = [(0, 0)] * values.ndim
    padding[-1] = (0, expected_dimension - current_dimension)
    return np.pad(values, padding, mode="constant")


def override_action_normalization(policy, torch_module) -> None:
    """Use measured DOM delta-action stats for deployment output scaling.

    The official checkpoint stores action buffers that match absolute-action
    statistics. Deployment consumes delta actions before adding the latest
    state, so use the measured delta-action distribution for action
    unnormalization in this runtime.
    """

    action_dim = int(policy.config.output_features["action"].shape[0])
    if action_dim != len(DOM_CHUNK000_DELTA_ACTION_MEAN):
        raise ValueError(
            "Delta-action normalization override expects a 7D action, "
            f"got {action_dim}D"
        )
    device = next(policy.parameters()).device
    mean = torch_module.as_tensor(
        DOM_CHUNK000_DELTA_ACTION_MEAN,
        dtype=torch_module.float32,
        device=device,
    )
    std = torch_module.as_tensor(
        DOM_CHUNK000_DELTA_ACTION_STD,
        dtype=torch_module.float32,
        device=device,
    )
    params = dict(policy.named_parameters())
    for prefix in ("normalize_targets", "unnormalize_outputs"):
        params[f"{prefix}.buffer_action.mean"].data.copy_(mean)
        params[f"{prefix}.buffer_action.std"].data.copy_(std)


def scale_runtime_delta_action_ry_mean(policy, scale: float) -> tuple[float, float]:
    """Scale only the in-memory output ry mean and return (before, after)."""

    key = "unnormalize_outputs.buffer_action.mean"
    tensors = dict(policy.named_parameters())
    tensors.update(dict(policy.named_buffers()))
    if key not in tensors:
        raise KeyError(f"Checkpoint is missing normalization tensor: {key}")
    mean = tensors[key]
    if mean.numel() <= 4:
        raise ValueError(
            f"Delta-action ry mean requires at least 5 action dimensions, got {mean.numel()}"
        )
    original = float(mean[4].item())
    mean.data[4].mul_(scale)
    return original, float(mean[4].item())


def _replace_queue_item(target: queue.Queue, item) -> None:
    try:
        target.put_nowait(item)
    except queue.Full:
        try:
            target.get_nowait()
        except queue.Empty:
            pass
        target.put_nowait(item)


class DynamicVLAWorker:
    """Single-slot asynchronous inference worker.

    The producer never waits for inference. While the model is busy, newer
    observations replace older pending observations.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.ready = threading.Event()
        self.error: Exception | None = None
        self.chunk_size = 20
        self._input: queue.Queue[PolicyObservation] = queue.Queue(maxsize=1)
        self._output: queue.Queue[ActionChunk] = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run_guarded,
            name="dynamicvla-inference",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def submit(self, observation: PolicyObservation) -> None:
        _replace_queue_item(self._input, observation)

    def get_result(self) -> ActionChunk | None:
        latest = None
        while True:
            try:
                latest = self._output.get_nowait()
            except queue.Empty:
                return latest

    def _run_guarded(self) -> None:
        try:
            self._run()
        except Exception as error:
            self.error = error
            self.ready.set()

    def _run(self) -> None:
        import torch
        import torch.nn.functional as torch_functional

        import utils.helpers

        checkpoint = Path(self.config.checkpoint).expanduser().resolve()
        config_path = checkpoint / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Checkpoint config not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as stream:
            checkpoint_config = json.load(stream)

        policy_config = utils.helpers.get_policy_cfg(cfg_file=config_path)
        policy_class = utils.helpers.get_policy_class(checkpoint_config["type"])
        policy = policy_class.from_pretrained(checkpoint, config=policy_config)
        policy.eval()
        device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        policy = policy.to(device)
        # override_action_normalization(policy, torch)  # disabled: safetensors already contains delta-action stats
        ry_mean_before, ry_mean_after = scale_runtime_delta_action_ry_mean(
            policy, self.config.delta_action_ry_mean_scale
        )
        LOGGER.warning(
            "Deployment-only delta-action ry mean scale=%.3f: %.8f -> %.8f rad "
            "(checkpoint file unchanged)",
            self.config.delta_action_ry_mean_scale,
            ry_mean_before,
            ry_mean_after,
        )
        self.chunk_size = int(policy.config.n_action_steps)
        self.ready.set()

        current_episode = None
        while not self._stop_event.is_set():
            try:
                observation = self._input.get(timeout=0.1)
            except queue.Empty:
                continue
            if observation.episode_id != current_episode:
                policy.reset()
                current_episode = observation.episode_id

            batch = {}
            for key, images in observation.images.items():
                tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
                expected_height, expected_width = policy.config.input_features[key].shape[-2:]
                if tensor.shape[-2:] != (expected_height, expected_width):
                    tensor = torch_functional.interpolate(
                        tensor,
                        size=(expected_height, expected_width),
                        mode="bilinear",
                        align_corners=False,
                    )
                # DynamicVLAPolicy.prepare_images() uses view() to merge the
                # observation and channel axes, which requires contiguous input.
                batch[key] = tensor.contiguous().unsqueeze(0).to(device)
            expected_state_dimension = policy.config.input_features[
                "observation.state"
            ].shape[0]
            fitted_states = fit_feature_dimension(
                observation.states, expected_state_dimension
            )
            raw_state = torch.from_numpy(fitted_states).float().unsqueeze(0).to(device)
            batch["observation.state"] = raw_state
            batch["task"] = [observation.task]

            tick = time.perf_counter()
            with torch.inference_mode():
                actions = policy.predict_action_chunk(batch)
                if policy.config.use_delta_action:
                    action_dimensions = actions.shape[-1] - 1
                    actions[..., :action_dimensions] += raw_state[:, -1:, :action_dimensions]
                    # Wrap rx (index 3) and rz (index 5) to [0, 2π) to match
                    # the Euler-angle preprocessing used during training.
                    actions[..., 3] = actions[..., 3] % (2 * torch.pi)
                    actions[..., 5] = actions[..., 5] % (2 * torch.pi)
            inference_seconds = time.perf_counter() - tick
            result = ActionChunk(
                episode_id=observation.episode_id,
                observation_index=observation.index,
                observation_timestamp_ns=observation.host_timestamp_ns,
                completed_timestamp_ns=time.monotonic_ns(),
                actions=actions[0].detach().cpu().float().numpy(),
                inference_seconds=inference_seconds,
                source_state=fitted_states[-1].astype(np.float32).copy(),
            )
            _replace_queue_item(self._output, result)
