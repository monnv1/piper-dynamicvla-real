from __future__ import annotations

import time

import numpy as np

from deploy.common.latest import FrameBuffer
from deploy.common.messages import CameraFrame
from deploy.config import CameraConfig
from deploy.devices.camera_base import CameraDevice


class RealSenseCamera(CameraDevice):
    """D435i RGB adapter using librealsense/pyrealsense2."""

    def __init__(self, name: str, config: CameraConfig, buffer: FrameBuffer) -> None:
        super().__init__(name, buffer)
        self.config = config

    def capture_loop(self) -> None:
        import pyrealsense2 as rs  # type: ignore

        pipeline = rs.pipeline()
        pipeline_config = rs.config()
        if self.config.serial:
            pipeline_config.enable_device(self.config.serial)
        pipeline_config.enable_stream(
            rs.stream.color,
            self.config.width,
            self.config.height,
            rs.format.rgb8,
            self.config.fps,
        )
        profile = pipeline.start(pipeline_config)
        device = profile.get_device()
        serial = device.get_info(rs.camera_info.serial_number)

        try:
            while not self._stop_event.is_set():
                frames = pipeline.wait_for_frames(1000)
                frame = frames.get_color_frame()
                if not frame:
                    continue
                rgb = np.asanyarray(frame.get_data()).copy()
                self.buffer.append(
                    CameraFrame(
                        camera=self.name,
                        serial=serial,
                        frame_number=int(frame.get_frame_number()),
                        device_timestamp_ms=float(frame.get_timestamp()),
                        host_timestamp_ns=time.monotonic_ns(),
                        rgb=rgb,
                    )
                )
        finally:
            pipeline.stop()

