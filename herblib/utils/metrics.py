"""Metrics and diagnostics for HERBlib.

Provides reconstruction error, energy tracking, and convergence diagnostics.
"""

from typing import List, Dict
from herblib._backend import to_numpy


def reconstruction_error(original, reconstructed) -> float:
    """Mean squared error between original and reconstructed input.

    Args:
        original: Original input, shape (B, d1)
        reconstructed: Reconstructed input, shape (B, d1)

    Returns:
        Scalar MSE value
    """
    o = to_numpy(original)
    r = to_numpy(reconstructed)
    return float(((o - r) ** 2).mean())


def energy_tracker(energies: List[float]) -> Dict[str, float]:
    """Compute summary statistics for energy over training.

    Args:
        energies: List of energy values per epoch

    Returns:
        Dict with final, min, delta, and convergence status
    """
    if not energies:
        return {"final": 0.0, "min": 0.0, "delta": 0.0, "converged": False}

    result = {
        "final": energies[-1],
        "min": min(energies),
        "delta": energies[0] - energies[-1],
        "converged": False,
    }

    if len(energies) > 1:
        result["converged"] = abs(energies[-1] - energies[-2]) < 1e-4

    return result