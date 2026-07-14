import numpy as np
import pytest

from deploy.tools.move_to_training_start import (
    TRAINING_START_DEG,
    TRAINING_START_RAD,
    validate_joint_values,
    validate_tcp_path,
)


def test_training_target_matches_calibrated_physical_configuration() -> None:
    np.testing.assert_allclose(
        TRAINING_START_DEG,
        [0.0, 89.913, -80.913, 0.0, 58.398, 0.0],
        atol=1e-9,
    )
    np.testing.assert_allclose(TRAINING_START_RAD, np.radians(TRAINING_START_DEG))
    validate_joint_values(TRAINING_START_DEG, "target", margin_deg=0.5)


def test_joint_limit_validation_rejects_out_of_range_target() -> None:
    invalid = TRAINING_START_DEG.copy()
    invalid[4] = 75.0
    with pytest.raises(RuntimeError, match="J5"):
        validate_joint_values(invalid, "target")


def test_tcp_path_validation_rejects_workspace_exit() -> None:
    path = np.asarray([[0.3, 0.0, 0.2], [0.1, 0.0, 0.2]])
    with pytest.raises(RuntimeError, match="sample 1"):
        validate_tcp_path(
            path,
            np.asarray([0.15, -0.35, 0.05]),
            np.asarray([0.60, 0.35, 0.55]),
        )
