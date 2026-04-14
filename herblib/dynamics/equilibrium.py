"""Convergence detection for HERBlib dynamics.

Tracks energy over leapfrog steps and detects when the system
has settled into equilibrium.
"""

from typing import List


class ConvergenceChecker:
    """Checks whether the system has reached energy equilibrium.

    Monitors energy over a sliding window. When the maximum absolute
    change in energy across consecutive steps falls below `tol` for
    `patience` consecutive windows, convergence is declared.
    """

    def __init__(self, tol: float = 1e-4, window: int = 5, patience: int = 3):
        self.tol = tol
        self.window = window
        self.patience = patience
        self._history: List[float] = []
        self._patience_count: int = 0

    def reset(self):
        self._history.clear()
        self._patience_count = 0

    def __call__(self, current_energy: float) -> bool:
        """Record energy and check for convergence.

        Args:
            current_energy: Total energy at current step

        Returns:
            True if the system has converged
        """
        self._history.append(current_energy)
        if len(self._history) < self.window:
            return False

        recent = self._history[-self.window:]
        max_delta = max(abs(recent[i + 1] - recent[i]) for i in range(len(recent) - 1))

        if max_delta < self.tol:
            self._patience_count += 1
        else:
            self._patience_count = 0

        return self._patience_count >= self.patience