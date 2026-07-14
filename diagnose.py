from __future__ import annotations

import argparse
import time

from deploy.common.latest import FrameBuffer
from deploy.config import load_config
from deploy.devices.factory import create_camera
from deploy.devices.piper_robot import PiperRobot


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only hardware diagnosis")
    parser.add_argument(
        "--config", default="deploy/configs/piper_gemini_d435i.yaml"
    )
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()
    config = load_config(args.config)
    buffers = {
        name: FrameBuffer()
        for name, camera_config in config.cameras.items()
        if camera_config.enabled
    }
    cameras = [
        create_camera(name, config.cameras[name], buffer)
        for name, buffer in buffers.items()
    ]
    robot = PiperRobot(config.robot)
    try:
        robot.start()
        for camera in cameras:
            camera.start()
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            errors = [camera.error for camera in cameras if camera.error is not None]
            if robot.error is not None:
                errors.append(robot.error)
            if errors:
                raise RuntimeError("Hardware diagnosis failed") from errors[0]
            state = robot.latest_state()
            camera_status = {
                name: (
                    None
                    if buffer.latest() is None
                    else {
                        "serial": buffer.latest().serial,
                        "frame": buffer.latest().frame_number,
                        "shape": buffer.latest().rgb.shape,
                    }
                )
                for name, buffer in buffers.items()
            }
            print(
                {
                    "cameras": camera_status,
                    "piper_hz": None if state is None else state.feedback_hz,
                    "joints": None if state is None else state.joint_radians.tolist(),
                    "position": None if state is None else state.position_m.tolist(),
                }
            )
            time.sleep(1.0)
    finally:
        for camera in cameras:
            camera.stop()
        robot.stop()


if __name__ == "__main__":
    main()

