from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from deploy.common.latest import FrameBuffer
from deploy.common.messages import PolicyObservation, RobotState


@dataclass(frozen=True)
class _Sample:
    images: dict[str, np.ndarray]
    state: np.ndarray


class ObservationBuilder:
    def __init__(
        self,
        camera_buffers: dict[str, FrameBuffer],
        history_indices: list[int],
        sync_tolerance_ms: float,
    ) -> None:
        if not history_indices or history_indices[-1] != 0:
            raise ValueError("history_indices must end in 0")
        self.camera_buffers = camera_buffers
        self.history_indices = history_indices
        self.tolerance_ns = int(sync_tolerance_ms * 1_000_000)
        self.samples: deque[_Sample] = deque(maxlen=max(32, abs(min(history_indices)) + 8))

    def build(
        self,
        episode_id: str,
        index: int,
        robot_state: RobotState,
        task: str,
    ) -> PolicyObservation:
        timestamp_ns = time.monotonic_ns()
        images: dict[str, np.ndarray] = {}
        for camera_name, buffer in self.camera_buffers.items():
            frame = buffer.nearest(timestamp_ns, self.tolerance_ns)
            if frame is None:
                raise RuntimeError(f"No synchronized frame for {camera_name}")
            images[f"observation.images.{camera_name}"] = frame.rgb

        self.samples.append(_Sample(images=images, state=robot_state.model_vector()))
        last_index = len(self.samples) - 1
        selected = [
            self.samples[max(0, min(last_index, last_index + relative_index))]
            for relative_index in self.history_indices
        ]
        return PolicyObservation(
            episode_id=episode_id,
            index=index,
            host_timestamp_ns=timestamp_ns,
            images={
                key: np.stack([sample.images[key] for sample in selected], axis=0)
                for key in images
            },
            states=np.stack([sample.state for sample in selected], axis=0),
            task=task,
        )

