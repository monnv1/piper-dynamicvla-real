from deploy.common.latest import FrameBuffer
from deploy.config import CameraConfig
from deploy.devices.orbbec_camera import OrbbecCamera
from deploy.devices.realsense_camera import RealSenseCamera


def create_camera(name: str, config: CameraConfig, buffer: FrameBuffer):
    if config.driver == "orbbec":
        return OrbbecCamera(name, config, buffer)
    if config.driver == "realsense":
        return RealSenseCamera(name, config, buffer)
    raise ValueError(f"Unsupported camera driver: {config.driver}")

