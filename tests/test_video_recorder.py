import time

import numpy as np

from deploy.common.messages import CameraFrame
from deploy.common.video_recorder import AsyncVideoWriter, RecordingFrameBuffer


def test_recording_frame_buffer_writes_mp4(tmp_path):
    path = tmp_path / "camera.mp4"
    recorder = AsyncVideoWriter("test_cam", path, fps=10.0, queue_size=4)
    buffer = RecordingFrameBuffer(recorder)

    recorder.start()
    for index in range(3):
        rgb = np.zeros((16, 20, 3), dtype=np.uint8)
        rgb[:, :, index % 3] = 80 + index
        buffer.append(
            CameraFrame(
                camera="test_cam",
                serial="serial",
                frame_number=index,
                device_timestamp_ms=float(index),
                host_timestamp_ns=time.monotonic_ns(),
                rgb=rgb,
            )
        )
    stats = recorder.stop()

    assert path.is_file()
    assert path.stat().st_size > 0
    assert stats.frames_written == 3
    assert stats.camera == "test_cam"
    assert buffer.latest().frame_number == 2
