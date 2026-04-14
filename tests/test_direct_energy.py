"""Tests for Direct Energy Gradient learning."""

import numpy as np
import pytest


def test_direct_energy_step_updates_weights(simple_config, numpy_backend):
    """A Direct Energy step should produce weights that differ from the original."""
    from herblib.learning.direct_energy import DirectEnergyLearning
    from herblib.utils.init import initialize_weights
    from herblib._backend import array

    weights = initialize_weights(simple_config)
    learner = DirectEnergyLearning(simple_config)
    X = array(np.random.randn(4, simple_config.layer_sizes[0]))

    W_before = np.array(weights.W[0]).copy()
    new_weights, _ = learner.step(X, weights)
    W_after = np.array(new_weights.W[0])

    assert not np.allclose(W_before, W_after, atol=1e-10), \
        "Direct Energy step did not update weights"


def test_direct_energy_clamps_s1(simple_config, small_weights, numpy_backend):
    """In the positive phase, s1 should remain clamped to the input data."""
    from herblib.learning.direct_energy import DirectEnergyLearning
    from herblib._backend import array

    learner = DirectEnergyLearning(simple_config)
    X = array(np.random.randn(2, simple_config.layer_sizes[0]))
    X_np = np.array(X).copy()

    pos_state = learner.positive_phase(X, small_weights)

    np.testing.assert_allclose(np.array(pos_state.s[0]), X_np, atol=1e-5,
                                err_msg="s1 was not clamped to input data")


def test_direct_energy_weight_shapes(simple_config, small_weights, numpy_backend):
    """Direct Energy weight updates should have the correct shapes."""
    from herblib.learning.direct_energy import DirectEnergyLearning
    from herblib._backend import array

    learner = DirectEnergyLearning(simple_config)
    X = array(np.random.randn(4, simple_config.layer_sizes[0]))

    pos_state = learner.positive_phase(X, small_weights)
    dW, db, dc = learner.compute_updates(pos_state)

    d1, d2, d3, d4 = simple_config.layer_sizes
    assert len(dW) == 3
    assert np.array(dW[0]).shape == (d1, d2)
    assert np.array(dW[1]).shape == (d2, d3)
    assert np.array(dW[2]).shape == (d3, d4)


def test_herb_with_direct_energy(simple_config, numpy_backend):
    """HERB with learning_method='direct' should train without errors."""
    from herblib import HERB
    from herblib._types import HERBConfig

    config = HERBConfig(
        layer_sizes=[4, 3, 2, 2],
        learning_method="direct",
        leapfrog_steps=3,
    )
    model = HERB(config, n_out=2)
    X = np.random.randn(10, 4)
    model.fit(X, epochs=3, batch_size=5, verbose=False)

    assert len(model._energy_log) == 3
    output = model.infer(X[:2])
    assert np.array(output).shape == (2, 2)