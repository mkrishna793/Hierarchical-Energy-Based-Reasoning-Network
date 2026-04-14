"""Tests for Contrastive Divergence learning."""

import numpy as np
import pytest


def test_cd_positive_phase_clamps_s1(simple_config, small_weights, numpy_backend):
    """In the positive phase, s1 should remain clamped to the input data."""
    from herblib.learning.cd import ContrastiveDivergence
    from herblib._backend import array, to_numpy

    cd = ContrastiveDivergence(simple_config)
    X = array(np.random.randn(2, simple_config.layer_sizes[0]))
    X_np = np.array(X).copy()

    pos_state = cd.positive_phase(X, small_weights)

    # s1 should still match input data
    np.testing.assert_allclose(np.array(pos_state.s[0]), X_np, atol=1e-5,
                                err_msg="s1 was not clamped to input data")


def test_cd_weight_update_shapes(simple_config, small_weights, numpy_backend):
    """CD weight updates should have the correct shapes."""
    from herblib.learning.cd import ContrastiveDivergence
    from herblib._backend import array

    cd = ContrastiveDivergence(simple_config)
    X = array(np.random.randn(4, simple_config.layer_sizes[0]))

    pos_state = cd.positive_phase(X, small_weights)
    neg_state = cd.negative_phase(small_weights, n_samples=4)
    dW, db, dc = cd.compute_updates(pos_state, neg_state)

    assert len(dW) == 3
    assert len(db) == 3
    assert len(dc) == 3

    d1, d2, d3, d4 = simple_config.layer_sizes
    assert np.array(dW[0]).shape == (d1, d2)
    assert np.array(dW[1]).shape == (d2, d3)
    assert np.array(dW[2]).shape == (d3, d4)


def test_cd_step_updates_weights(simple_config, numpy_backend):
    """A CD step should produce weights that differ from the original."""
    from herblib.learning.cd import ContrastiveDivergence
    from herblib.utils.init import initialize_weights
    from herblib._backend import array, to_numpy

    weights = initialize_weights(simple_config)
    cd = ContrastiveDivergence(simple_config)
    X = array(np.random.randn(4, simple_config.layer_sizes[0]))

    W_before = np.array(weights.W[0]).copy()
    new_weights, _ = cd.step(X, weights)
    W_after = np.array(new_weights.W[0])

    # Weights should have changed
    assert not np.allclose(W_before, W_after, atol=1e-10), \
        "CD step did not update weights"