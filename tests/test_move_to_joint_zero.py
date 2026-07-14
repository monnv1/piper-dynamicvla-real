import numpy as np

from deploy.tools.move_to_joint_zero import JOINT_ZERO_DEG
from deploy.tools.move_to_training_start import validate_joint_values


def test_joint_zero_is_six_valid_zero_angles() -> None:
    np.testing.assert_array_equal(JOINT_ZERO_DEG, np.zeros(6))
    validate_joint_values(JOINT_ZERO_DEG, "joint zero")
