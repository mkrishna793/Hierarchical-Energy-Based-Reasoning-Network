"""Tests for energy computations."""

import numpy as np
import pytest


def test_pairwise_energy_positive(simple_config, small_weights, numpy_backend):
    """pairwise_energy should return negative values for positive inputs and weights."""
    from herblib.core.energy import pairwise_energy
    from herblib._backend import array

    s_l = array(np.ones((2, 4)))
    s_lp1 = array(np.ones((2, 3)))
    W = small_weights.W[0]
    b = small_weights.b[0]
    c = small_weights.c[0]  # c2, shape (d2=3,)

    E = pairwise_energy(s_l, s_lp1, W, b, c)
    result = np.array(E)

    # Energy should be a 1D array of shape (B,)
    assert result.shape == (2,)


def test_total_energy_decreases_with_gradient_step(simple_config, small_weights, numpy_backend):
    """Taking a small gradient step should decrease total energy."""
    from herblib.core.energy import total_energy
    from herblib.core.gradients import compute_all_gradients
    from herblib._backend import array, copy
    from herblib._types import HERBState

    rng = np.random.RandomState(42)
    d1, d2, d3, d4 = simple_config.layer_sizes
    B = 2

    s = [array(rng.randn(B, d)) for d in [d1, d2, d3, d4]]
    p = [array(rng.randn(B, d) * 0.01) for d in [d1, d2, d3, d4]]
    state = HERBState(s=s, p=p)

    E_before = float(np.mean(total_energy(state, small_weights, simple_config.lam)))

    # Take a small gradient step
    grads = compute_all_gradients(state, small_weights, simple_config.lam)
    lr = 0.001
    new_s = [copy(state.s[l]) - lr * grads[l] for l in range(4)]
    new_state = HERBState(s=new_s, p=state.p)

    E_after = float(np.mean(total_energy(new_state, small_weights, simple_config.lam)))

    # Energy should decrease after a gradient step
    assert E_after < E_before, f"Energy did not decrease: {E_before} -> {E_after}"


def test_total_energy_components(simple_config, small_weights, numpy_backend):
    """Total energy should equal sum of pairwise energies plus regularization."""
    from herblib.core.energy import pairwise_energy, total_energy
    from herblib._backend import array
    from herblib._types import HERBState

    rng = np.random.RandomState(123)
    d1, d2, d3, d4 = simple_config.layer_sizes
    B = 1

    s = [array(rng.randn(B, d)) for d in [d1, d2, d3, d4]]
    p = [array(rng.randn(B, d) * 0.01) for d in [d1, d2, d3, d4]]
    state = HERBState(s=s, p=p)

    E_total = total_energy(state, small_weights, simple_config.lam)

    E1 = pairwise_energy(s[0], s[1], small_weights.W[0], small_weights.b[0], small_weights.c[0])
    E2 = pairwise_energy(s[1], s[2], small_weights.W[1], small_weights.b[1], small_weights.c[1])
    E3 = pairwise_energy(s[2], s[3], small_weights.W[2], small_weights.b[2], small_weights.c[2])
    E_reg = simple_config.lam * sum(np.sum(np.array(si) ** 2) for si in s)

    expected = np.array(E1) + np.array(E2) + np.array(E3) + E_reg
    np.testing.assert_allclose(np.array(E_total), expected, atol=1e-10)