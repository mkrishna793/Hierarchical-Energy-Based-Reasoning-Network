"""Tests for the leapfrog integrator and energy conservation."""

import numpy as np
import pytest


def test_leapfrog_step_shapes(simple_config, small_weights, small_state, numpy_backend):
    """A leapfrog step should produce states with the same shapes."""
    from herblib.dynamics.leapfrog import leapfrog_step

    new_state = leapfrog_step(small_state, small_weights, simple_config)

    d1, d2, d3, d4 = simple_config.layer_sizes
    for l, d in enumerate([d1, d2, d3, d4]):
        assert np.array(new_state.s[l]).shape == (2, d), f"Layer {l} state shape mismatch"
        assert np.array(new_state.p[l]).shape == (2, d), f"Layer {l} momentum shape mismatch"


def test_leapfrog_hamiltonian_conservation(simple_config, numpy_backend):
    """The leapfrog integrator should approximately conserve the Hamiltonian (KE + PE).

    Uses a small step size and small state values for stability.
    """
    from herblib.core.energy import total_energy
    from herblib.dynamics.leapfrog import leapfrog_step
    from herblib.utils.init import initialize_weights
    from herblib._types import HERBConfig
    from herblib._backend import array
    from herblib._types import HERBState

    # Use a config with small epsilon for stable integration
    stable_config = HERBConfig(
        layer_sizes=[4, 3, 2, 2],
        lr=0.01, lam=0.001, epsilon=0.001, mass=1.0,
        leapfrog_steps=5, convergence_tol=1e-6,
    )
    weights = initialize_weights(stable_config)

    rng = np.random.RandomState(42)
    d1, d2, d3, d4 = stable_config.layer_sizes
    B = 2

    s = [array(rng.randn(B, d) * 0.1) for d in [d1, d2, d3, d4]]
    p = [array(rng.randn(B, d) * 0.01) for d in [d1, d2, d3, d4]]
    state = HERBState(s=s, p=p)

    def compute_hamiltonian(st, wts, cfg):
        """H = KE + PE per sample, averaged over batch."""
        E_pot = np.array(total_energy(st, wts, cfg.lam))
        E_kin = sum(np.sum(np.array(pi) ** 2, axis=1) for pi in st.p) / (2 * cfg.mass)
        return float(np.mean(E_kin + E_pot))

    H_initial = compute_hamiltonian(state, weights, stable_config)

    for _ in range(50):
        state = leapfrog_step(state, weights, stable_config)

    H_final = compute_hamiltonian(state, weights, stable_config)

    # Hamiltonian should be approximately conserved with small step size
    relative_change = abs(H_final - H_initial) / (abs(H_initial) + 1e-8)
    assert relative_change < 0.1, (
        f"Hamiltonian not conserved: {H_initial:.6f} -> {H_final:.6f}, "
        f"relative change: {relative_change:.4f}"
    )


def test_clamped_layers_unchanged(simple_config, small_weights, small_state, numpy_backend):
    """Clamped layers should remain exactly unchanged after a leapfrog step."""
    from herblib.dynamics.leapfrog import leapfrog_step
    from herblib._backend import to_numpy

    original_s1 = to_numpy(small_state.s[0]).copy()
    original_p1 = to_numpy(small_state.p[0]).copy()

    new_state = leapfrog_step(small_state, small_weights, simple_config, clamped_layers={0, 2})

    np.testing.assert_array_equal(
        np.array(new_state.s[0]), original_s1,
        err_msg="Clamped layer s[0] was modified"
    )
    np.testing.assert_array_equal(
        np.array(new_state.p[0]), original_p1,
        err_msg="Clamped layer p[0] was modified"
    )

    original_s3 = to_numpy(small_state.s[2]).copy()
    np.testing.assert_array_equal(
        np.array(new_state.s[2]), original_s3,
        err_msg="Clamped layer s[2] was modified"
    )


def test_run_to_equilibrium_converges(simple_config, small_weights, numpy_backend):
    """run_to_equilibrium should produce a state with lower energy than initial."""
    from herblib.core.energy import total_energy
    from herblib.dynamics.leapfrog import run_to_equilibrium
    from herblib._backend import array
    from herblib._types import HERBState
    from herblib.utils.init import initialize_weights

    rng = np.random.RandomState(42)
    d1, d2, d3, d4 = simple_config.layer_sizes
    B = 2

    s = [array(rng.randn(B, d)) for d in [d1, d2, d3, d4]]
    p = [array(rng.randn(B, d) * 0.01) for d in [d1, d2, d3, d4]]
    state = HERBState(s=s, p=p)

    E_before = float(np.mean(total_energy(state, small_weights, simple_config.lam)))

    state_after = run_to_equilibrium(state, small_weights, simple_config, max_steps=50)
    E_after = float(np.mean(total_energy(state_after, small_weights, simple_config.lam)))

    # Energy should decrease or stay approximately the same
    assert E_after <= E_before + 0.1, (
        f"Energy increased after equilibrium: {E_before:.6f} -> {E_after:.6f}"
    )