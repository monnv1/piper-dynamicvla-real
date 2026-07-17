import numpy as np
import pytest
import torch

from deploy.policy.inference_worker import (
    fit_feature_dimension,
    scale_runtime_delta_action_ry_mean,
)


def test_trims_state_to_checkpoint_dimension():
    states = np.arange(14, dtype=np.float32).reshape(2, 7)
    fitted = fit_feature_dimension(states, 6)
    assert fitted.shape == (2, 6)
    np.testing.assert_array_equal(fitted, states[:, :6])


def test_pads_state_to_checkpoint_dimension():
    states = np.ones((2, 6), dtype=np.float32)
    fitted = fit_feature_dimension(states, 8)
    assert fitted.shape == (2, 8)
    np.testing.assert_array_equal(fitted[:, :6], states)
    np.testing.assert_array_equal(fitted[:, 6:], 0)


def test_scales_only_runtime_delta_action_ry_mean():
    class FakePolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.unnormalize_outputs = torch.nn.Module()
            self.unnormalize_outputs.buffer_action = torch.nn.Module()
            self.unnormalize_outputs.buffer_action.register_parameter(
                "mean", torch.nn.Parameter(torch.arange(7, dtype=torch.float32))
            )

    policy = FakePolicy()
    before_tensor = policy.unnormalize_outputs.buffer_action.mean.detach().clone()

    before, after = scale_runtime_delta_action_ry_mean(policy, 0.5)

    assert before == pytest.approx(4.0)
    assert after == pytest.approx(2.0)
    expected = before_tensor.clone()
    expected[4] *= 0.5
    torch.testing.assert_close(
        policy.unnormalize_outputs.buffer_action.mean.detach(), expected
    )
