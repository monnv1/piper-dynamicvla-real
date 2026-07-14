from __future__ import annotations

import abc
import threading

from deploy.common.latest import FrameBuffer


class CameraDevice(abc.ABC):
    def __init__(self, name: str, buffer: FrameBuffer) -> None:
        self.name = name
        self.buffer = buffer
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: Exception | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError(f"Camera {self.name} is already started")
        self._thread = threading.Thread(
            target=self._run_guarded,
            name=f"camera-{self.name}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _run_guarded(self) -> None:
        try:
            self.capture_loop()
        except Exception as error:  # surfaced to the runtime watchdog
            self.error = error
            self._stop_event.set()

    @abc.abstractmethod
    def capture_loop(self) -> None:
        raise NotImplementedError

