"""Hamiltonian leapfrog (Störmer-Verlet) integrator for HERBlib.

Implements symplectic integration of Hamilton's equations:
  ds^l/dt = p^l / m^l
  dp^l/dt = -grad_{s^l} E_total

Includes optional momentum damping (friction) and state clipping
to ensure numerical stability during training.
"""

from herblib._types import HERBState, HERBWeights, HERBConfig
from herblib.core.gradients import compute_all_gradients
from herblib._backend import copy, clip, xp
from typing import Set, Optional


def leapfrog_step(
    state: HERBState,
    weights: HERBWeights,
    config: HERBConfig,
    clamped_layers: Optional[Set[int]] = None,
) -> HERBState:
    """One full leapfrog step with optional damping and clipping.

    Step 1: p^{t+1/2} = damping * p^t - (eps/2) * grad E  (half-step momentum + damping)
    Step 2: s^{t+1} = s^t + eps * (p^{t+1/2} / m)  (full-step position)
    Step 3: p^{t+1} = damping * p^{t+1/2} - (eps/2) * grad E  (half-step momentum)

    After each step, states are clipped to [-state_clip, +state_clip].

    Args:
        state: Current HERBState with s and p for all 4 layers
        weights: Current HERBWeights
        config: HERBConfig with epsilon, mass, lam, damping, state_clip
        clamped_layers: Set of layer indices (0-3) to keep fixed

    Returns:
        New HERBState after one leapfrog step
    """
    if clamped_layers is None:
        clamped_layers = set()

    eps = config.epsilon
    m = config.mass
    damp = config.damping
    sclip = config.state_clip

    # Step 1: Half-step momentum update using gradient at current position
    grads = compute_all_gradients(state, weights, config.lam)
    p_half = []
    for l in range(4):
        if l in clamped_layers:
            p_half.append(copy(state.p[l]))
        else:
            p_half.append(damp * state.p[l] - (eps / 2.0) * grads[l])

    # Step 2: Full-step position update with clipping
    s_new = []
    for l in range(4):
        if l in clamped_layers:
            s_new.append(copy(state.s[l]))
        else:
            s_new.append(clip(state.s[l] + eps * (p_half[l] / m), -sclip, sclip))

    # Step 3: Half-step momentum update using gradient at new position
    new_state = HERBState(s=s_new, p=p_half)
    grads_new = compute_all_gradients(new_state, weights, config.lam)
    p_new = []
    for l in range(4):
        if l in clamped_layers:
            p_new.append(copy(state.p[l]))
        else:
            p_new.append(damp * p_half[l] - (eps / 2.0) * grads_new[l])

    return HERBState(s=s_new, p=p_new)


def run_to_equilibrium(
    state: HERBState,
    weights: HERBWeights,
    config: HERBConfig,
    clamped_layers: Optional[Set[int]] = None,
    max_steps: Optional[int] = None,
) -> HERBState:
    """Run leapfrog integration for a fixed number of steps or until convergence.

    Args:
        state: Initial HERBState
        weights: Current HERBWeights
        config: HERBConfig
        clamped_layers: Layers to keep fixed
        max_steps: Override config.leapfrog_steps if provided

    Returns:
        HERBState after running leapfrog
    """
    from herblib.dynamics.equilibrium import ConvergenceChecker
    from herblib.core.energy import total_energy

    if max_steps is None:
        max_steps = config.leapfrog_steps

    checker = ConvergenceChecker(
        tol=config.convergence_tol,
        patience=config.convergence_patience,
    )

    for step in range(max_steps):
        state = leapfrog_step(state, weights, config, clamped_layers)
        E = float(total_energy(state, weights, config.lam).mean())
        if checker(E):
            break

    return state