import numpy as np
from scipy.spatial.transform import Rotation

from deploy.devices.piper_frames import PiperFrameTransform


def make_transform() -> PiperFrameTransform:
    return PiperFrameTransform([0.0, 0.0, 0.1334], [0.0, 0.0, np.pi])


def test_sdk_identity_maps_to_simulated_tcp() -> None:
    position, euler = make_transform().sdk_to_model_pose(
        np.zeros(3), np.zeros(3)
    )
    np.testing.assert_allclose(position, [0.0, 0.0, 0.1334], atol=1e-10)
    np.testing.assert_allclose(
        Rotation.from_euler("xyz", euler).as_matrix(),
        Rotation.from_euler("z", np.pi).as_matrix(),
        atol=1e-10,
    )


def test_pose_conversion_round_trip() -> None:
    frames = make_transform()
    sdk_position = np.asarray([0.31, -0.12, 0.27])
    sdk_euler = np.asarray([2.8, 0.4, -2.6])
    model_position, model_euler = frames.sdk_to_model_pose(
        sdk_position, sdk_euler
    )
    recovered_position, recovered_euler = frames.model_to_sdk_pose(
        model_position, model_euler
    )
    np.testing.assert_allclose(recovered_position, sdk_position, atol=1e-10)
    np.testing.assert_allclose(
        Rotation.from_euler("xyz", recovered_euler).as_matrix(),
        Rotation.from_euler("xyz", sdk_euler).as_matrix(),
        atol=1e-10,
    )
