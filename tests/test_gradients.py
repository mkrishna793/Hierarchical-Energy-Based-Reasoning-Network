"""Tests for gradient computations — the most critical tests.

Each gradient function is verified against numerical finite differences.
"""

import numpy as np
import pytest
from herblib._backend import use


def _finite_diff_energy(state, weights, lam, layer_idx, eps=1e-5):
    """Compute numerical gradient of total_energy w.r.t. state.s[layer_idx] via finite differences.

    Computes per-sample gradients by using per-sample energy values.
    """
    from herblib.core.energy import total_energy
    from herblib._backend import array, copy
    from herblib._types import HERBState

    B, d = np.array(state.s[layer_idx]).shape
    grad = np.zeros((B, d))

    for b_idx in range(B):
        for i in range(d):
            # E(s + eps) — per-sample energy
            s_plus = [copy(si) for si in state.s]
            s_plus[layer_idx] = copy(state.s[layer_idx])
            arr = np.array(s_plus[layer_idx])
            arr[b_idx, i] += eps
            s_plus[layer_idx] = array(arr)
            state_plus = HERBState(s=s_plus, p=state.p)
            E_plus = np.array(total_energy(state_plus, weights, lam))
            e_plus = E_plus[b_idx]  # per-sample

            # E(s - eps)
            s_minus = [copy(si) for si in state.s]
            s_minus[layer_idx] = copy(state.s[layer_idx])
            arr2 = np.array(s_minus[layer_idx])
            arr2[b_idx, i] -= eps
            s_minus[layer_idx] = array(arr2)
            state_minus = HERBState(s=s_minus, p=state.p)
            E_minus = np.array(total_energy(state_minus, weights, lam))
            e_minus = E_minus[b_idx]

            grad[b_idx, i] = (e_plus - e_minus) / (2 * eps)

    return grad


def test_grad_s1_finite_diff(simple_config, small_weights, numpy_backend):
    """Verify grad_s1 against numerical finite differences."""
    from herblib.core.gradients import grad_s1
    from herblib._backend import array
    from herblib._types import HERBState

    rng = np.random.RandomState(42)
    d1, d2, d3, d4 = simple_config.layer_sizes
    B = 2

    s = [array(rng.randn(B, d)) for d in [d1, d2, d3, d4]]
    p = [array(rng.randn(B, d) * 0.01) for d in [d1, d2, d3, d4]]
    state = HERBState(s=s, p=p)

    g_analytical = grad_s1(s[0], s[1], small_weights.W[0], small_weights.b[0], simple_config.lam)
    g_numerical = _finite_diff_energy(state, small_weights, simple_config.lam, 0)

    np.testing.assert_allclose(np.array(g_analytical), g_numerical, atol=1e-4)


def test_grad_s2_finite_diff(simple_config, small_weights, numpy_backend):
    """Verify grad_s2 against numerical finite differences."""
    from herblib.core.gradients import grad_s2
    from herblib._backend import array
    from herblib._types import HERBState

    rng = np.random.RandomState(42)
    d1, d2, d3, d4 = simple_config.layer_sizes
    B = 2

    s = [array(rng.randn(B, d)) for d in [d1, d2, d3, d4]]
    p = [array(rng.randn(B, d) * 0.01) for d in [d1, d2, d3, d4]]
    state = HERBState(s=s, p=p)

    g_analytical = grad_s2(s[0], s[1], s[2], small_weights.W[0], small_weights.W[1],
                           small_weights.b[1], small_weights.c[0], simple_config.lam)
    g_numerical = _finite_diff_energy(state, small_weights, simple_config.lam, 1)

    np.testing.assert_allclose(np.array(g_analytical), g_numerical, atol=1e-4)


def test_grad_s3_finite_diff(simple_config, small_weights, numpy_backend):
    """Verify grad_s3 against numerical finite differences."""
    from herblib.core.gradients import grad_s3
    from herblib._backend import array
    from herblib._types import HERBState

    rng = np.random.RandomState(42)
    d1, d2, d3, d4 = simple_config.layer_sizes
    B = 2

    s = [array(rng.randn(B, d)) for d in [d1, d2, d3, d4]]
    p = [array(rng.randn(B, d) * 0.01) for d in [d1, d2, d3, d4]]
    state = HERBState(s=s, p=p)

    g_analytical = grad_s3(s[1], s[2], s[3], small_weights.W[1], small_weights.W[2],
                           small_weights.b[2], small_weights.c[1], simple_config.lam)
    g_numerical = _finite_diff_energy(state, small_weights, simple_config.lam, 2)

    np.testing.assert_allclose(np.array(g_analytical), g_numerical, atol=1e-4)


def test_grad_s4_finite_diff(simple_config, small_weights, numpy_backend):
    """Verify grad_s4 against numerical finite differences."""
    from herblib.core.gradients import grad_s4
    from herblib._backend import array
    from herblib._types import HERBState

    rng = np.random.RandomState(42)
    d1, d2, d3, d4 = simple_config.layer_sizes
    B = 2

    s = [array(rng.randn(B, d)) for d in [d1, d2, d3, d4]]
    p = [array(rng.randn(B, d) * 0.01) for d in [d1, d2, d3, d4]]
    state = HERBState(s=s, p=p)

    g_analytical = grad_s4(s[2], s[3], small_weights.W[2], small_weights.c[2], simple_config.lam)
    g_numerical = _finite_diff_energy(state, small_weights, simple_config.lam, 3)

    np.testing.assert_allclose(np.array(g_analytical), g_numerical, atol=1e-4)


def test_compute_all_gradients_shapes(simple_config, small_weights, numpy_backend):
    """All gradients should have shapes matching their layer states."""
    from herblib.core.gradients import compute_all_gradients
    from herblib._backend import array
    from herblib._types import HERBState

    rng = np.random.RandomState(99)
    d1, d2, d3, d4 = simple_config.layer_sizes
    B = 3

    s = [array(rng.randn(B, d)) for d in [d1, d2, d3, d4]]
    p = [array(rng.randn(B, d) * 0.01) for d in [d1, d2, d3, d4]]
    state = HERBState(s=s, p=p)

    grads = compute_all_gradients(state, small_weights, simple_config.lam)

    assert len(grads) == 4
    assert np.array(grads[0]).shape == (B, d1)
    assert np.array(grads[1]).shape == (B, d2)
    assert np.array(grads[2]).shape == (B, d3)
    assert np.array(grads[3]).shape == (B, d4)