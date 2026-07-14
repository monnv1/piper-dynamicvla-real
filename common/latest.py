from __future__ import annotations

import threading
from collections import deque
from typing import Generic, TypeVar

from .messages import CameraFrame

T = TypeVar("T")


class LatestValue(Generic[T]):
    """Thread-safe single-slot value; publishers overwrite stale values."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._value: T | None = None
        self._sequence = 0

    def publish(self, value: T) -> int:
        with self._condition:
            self._value = value
            self._sequence += 1
            self._condition.notify_all()
            return self._sequence

    def get(self) -> T | None:
        with self._condition:
            return self._value

    def wait_newer(self, sequence: int, timeout: float | None = None) -> tuple[int, T | None]:
        with self._condition:
            self._condition.wait_for(lambda: self._sequence > sequence, timeout=timeout)
            return self._sequence, self._value


class FrameBuffer:
    """Small timestamp-indexed frame buffer for software synchronization."""

    def __init__(self, maxlen: int = 120) -> None:
        self._frames: deque[CameraFrame] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, frame: CameraFrame) -> None:
        with self._lock:
            self._frames.append(frame)

    def latest(self) -> CameraFrame | None:
        with self._lock:
            return self._frames[-1] if self._frames else None

    def nearest(self, timestamp_ns: int, tolerance_ns: int) -> CameraFrame | None:
        with self._lock:
            if not self._frames:
                return None
            frame = min(
                self._frames,
                key=lambda candidate: abs(candidate.host_timestamp_ns - timestamp_ns),
            )
            if abs(frame.host_timestamp_ns - timestamp_ns) > tolerance_ns:
                return None
            return frame

