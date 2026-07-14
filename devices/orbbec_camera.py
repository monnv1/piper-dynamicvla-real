from __future__ import annotations

import time

import numpy as np

from deploy.common.latest import FrameBuffer
from deploy.common.messages import CameraFrame
from deploy.config import CameraConfig
from deploy.devices.camera_base import CameraDevice


class OrbbecCamera(CameraDevice):
    """Gemini 305 RGB adapter using Orbbec SDK v2."""

    def __init__(self, name: str, config: CameraConfig, buffer: FrameBuffer) -> None:
        super().__init__(name, buffer)
        self.config = config

    def capture_loop(self) -> None:
        from pyorbbecsdk import (  # type: ignore
            Config,
            Context,
            FormatConvertFilter,
            OBConvertFormat,
            OBFormat,
            OBSensorType,
            Pipeline,
        )

        context = Context()
        devices = context.query_devices()
        if devices.get_count() == 0:
            raise RuntimeError("No Orbbec camera found")

        device = None
        for index in range(devices.get_count()):
            candidate = devices.get_device_by_index(index)
            serial = candidate.get_device_info().get_serial_number()
            if not self.config.serial or serial == self.config.serial:
                device = candidate
                break
        if device is None:
            raise RuntimeError(f"Orbbec serial not found: {self.config.serial}")

        info = device.get_device_info()
        serial = info.get_serial_number()
        pipeline = Pipeline(device)
        pipeline_config = Config()
        profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        try:
            profile = profiles.get_video_stream_profile(
                self.config.width,
                self.config.height,
                OBFormat.RGB,
                self.config.fps,
            )
        except Exception:
            profile = profiles.get_default_video_stream_profile()
        pipeline_config.enable_stream(profile)
        pipeline.start(pipeline_config)

        converter = None
        convert_formats = {
            OBFormat.I420: OBConvertFormat.I420_TO_RGB888,
            OBFormat.MJPG: OBConvertFormat.MJPG_TO_RGB888,
            OBFormat.YUYV: OBConvertFormat.YUYV_TO_RGB888,
            OBFormat.NV21: OBConvertFormat.NV21_TO_RGB888,
            OBFormat.NV12: OBConvertFormat.NV12_TO_RGB888,
            OBFormat.UYVY: OBConvertFormat.UYVY_TO_RGB888,
        }
        try:
            while not self._stop_event.is_set():
                frames = pipeline.wait_for_frames(1000)
                if frames is None:
                    continue
                frame = frames.get_color_frame()
                if frame is None:
                    continue
                if frame.get_format() != OBFormat.RGB:
                    convert_format = convert_formats.get(frame.get_format())
                    if convert_format is None:
                        raise RuntimeError(
                            f"Unsupported Orbbec color format: {frame.get_format()}"
                        )
                    if converter is None:
                        converter = FormatConvertFilter()
                        converter.set_format_convert_format(convert_format)
                    frame = converter.process(frame)
                    if frame is None:
                        raise RuntimeError("Orbbec RGB conversion failed")
                width, height = frame.get_width(), frame.get_height()
                rgb = np.frombuffer(frame.get_data(), dtype=np.uint8).copy()
                rgb = rgb.reshape(height, width, 3)
                self.buffer.append(
                    CameraFrame(
                        camera=self.name,
                        serial=serial,
                        frame_number=int(frame.get_index()),
                        device_timestamp_ms=float(frame.get_timestamp()),
                        host_timestamp_ns=time.monotonic_ns(),
                        rgb=rgb,
                    )
                )
        finally:
            pipeline.stop()
