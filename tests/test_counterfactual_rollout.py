from collections import deque

import numpy as np
import pytest

from deploy.tools.counterfactual_rollout import (
    select_rollout_action,
    select_state_history,
)


def test_select_state_history_clamps_missing_past_to_first_state():
    states = deque([np.array([1, 2, 3], dtype=np.float32)])

    selected = select_state_history(states, [-2, 0])

    assert selected.shape == (2, 3)
    np.testing.assert_array_equal(selected[0], states[0])
    np.testing.assert_array_equal(selected[1], states[0])


def test_select_state_history_uses_relative_indices():
    states = deque(
        [
            np.array([0], dtype=np.float32),
            np.array([1], dtype=np.float32),
            np.array([2], dtype=np.float32),
            np.array([3], dtype=np.float32),
        ]
    )

    selected = select_state_history(states, [-2, 0])

    np.testing.assert_array_equal(selected[:, 0], np.array([1, 3], dtype=np.float32))


def test_select_rollout_action_returns_requested_row_copy():
    actions = np.arange(21, dtype=np.float32).reshape(3, 7)

    selected = select_rollout_action(actions, 1)

    np.testing.assert_array_equal(selected, actions[1])
    selected[0] = -1
    assert actions[1, 0] != -1


def test_select_rollout_action_rejects_out_of_range_index():
    actions = np.zeros((2, 7), dtype=np.float32)

    with pytest.raises(IndexError):
        select_rollout_action(actions, 2)
