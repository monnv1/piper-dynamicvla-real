import numpy as np

import pytest

from deploy.tools.jog_piper_joint import make_jog_target, validate_jog_target


def test_jog_changes_only_selected_joint() -> None:
    start = np.asarray([1.0, 2.0, -3.0, 4.0, 5.0, 6.0])
    target = make_jog_target(start, joint_number=3, delta_deg=-1.0)
    np.testing.assert_allclose(target, [1.0, 2.0, -4.0, 4.0, 5.0, 6.0])
    np.testing.assert_allclose(start, [1.0, 2.0, -3.0, 4.0, 5.0, 6.0])


def test_other_joints_may_remain_on_hard_limits() -> None:
    target = np.asarray([1.0, 0.0, 0.0, 0.0, 28.0, 0.0])
    validate_jog_target(target, joint_number=1)


def test_moved_joint_must_leave_its_soft_limit() -> None:
    target = np.asarray([0.0, 0.1, 0.0, 0.0, 28.0, 0.0])
    with pytest.raises(RuntimeError, match="soft range"):
        validate_jog_target(target, joint_number=2)
