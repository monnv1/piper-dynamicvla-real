from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from deploy.common.latest import FrameBuffer
from deploy.common.messages import CameraFrame


@dataclass
class VideoRecorderStats:
    camera: str
    path: str
    frames_written: int = 0
    frames_dropped: int = 0


class AsyncVideoWriter:
    """Asynchronous RGB-to-MP4 writer for one camera stream."""

    def __init__(self, camera: str, path: str | Path, fps: float, queue_size: int = 120) -> None:
        if fps <= 0:
            raise ValueError("video fps must be positive")
        self.camera = camera
        self.path = Path(path)
        self.fps = float(fps)
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        self._writer = None
        self.frames_written = 0
        self.frames_dropped = 0
        self.error: Exception | None = None

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._run_guarded,
            name=f"video-writer-{self.camera}",
            daemon=True,
        )
        self._thread.start()

    def write(self, rgb: np.ndarray) -> None:
        frame = np.asarray(rgb)
        if frame.ndim != 3 or frame.shape[2] != 3:
            self.frames_dropped += 1
            return
        try:
            self._queue.put_nowait(frame.copy())
        except queue.Full:
            self.frames_dropped += 1

    def stop(self) -> VideoRecorderStats:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            # Make room for the sentinel. Dropping one queued frame is preferable
            # to hanging shutdown while the robot is being stopped.
            try:
                self._queue.get_nowait()
                self.frames_dropped += 1
            except queue.Empty:
                pass
            self._queue.put_nowait(None)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        return VideoRecorderStats(
            camera=self.camera,
            path=str(self.path),
            frames_written=self.frames_written,
            frames_dropped=self.frames_dropped,
        )

    def _run_guarded(self) -> None:
        try:
            self._run()
        except Exception as error:
            self.error = error

    def _run(self) -> None:
        while True:
            frame = self._queue.get()
            if frame is None:
                return
            rgb = np.ascontiguousarray(frame)
            height, width = rgb.shape[:2]
            if self._writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self._writer = cv2.VideoWriter(
                    str(self.path), fourcc, self.fps, (width, height)
                )
                if not self._writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer: {self.path}")
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            self._writer.write(bgr)
            self.frames_written += 1


class RecordingFrameBuffer(FrameBuffer):
    """FrameBuffer that mirrors frames into an AsyncVideoWriter."""

    def __init__(self, recorder: AsyncVideoWriter, maxlen: int = 120) -> None:
        super().__init__(maxlen=maxlen)
        self.recorder = recorder

    def append(self, frame: CameraFrame) -> None:
        super().append(frame)
        self.recorder.write(frame.rgb)
