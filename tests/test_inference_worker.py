import numpy as np

from deploy.policy.inference_worker import fit_feature_dimension


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
