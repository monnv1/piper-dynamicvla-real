import json

import numpy as np
from scipy.spatial.transform import Rotation

from deploy.tools.move_to_model_tcp_pose import jsonable, quat_wxyz_to_euler_xyz


def test_default_quat_matches_expected_model_tcp_orientation() -> None:
    quat_wxyz = [0.0, 0.9739, 0.0, 0.227]

    euler = quat_wxyz_to_euler_xyz(quat_wxyz)

    expected = Rotation.from_euler("xyz", [np.pi, -0.458, 0.0])
    actual = Rotation.from_euler("xyz", euler)
    assert (expected.inv() * actual).magnitude() < 0.002


def test_jsonable_converts_numpy_values() -> None:
    payload = {
        "array": np.asarray([1.0, 2.0]),
        "float": np.float64(3.0),
        "int": np.int64(4),
        "bool": np.bool_(True),
    }

    encoded = json.dumps(jsonable(payload), allow_nan=False)

    assert json.loads(encoded) == {
        "array": [1.0, 2.0],
        "float": 3.0,
        "int": 4,
        "bool": True,
    }
