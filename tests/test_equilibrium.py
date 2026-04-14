"""Tests for convergence checker."""

from herblib.dynamics.equilibrium import ConvergenceChecker


def test_convergence_checker_no_convergence_early():
    """Should not converge with only a few readings."""
    checker = ConvergenceChecker(tol=1e-4, window=3, patience=2)
    assert not checker(10.0)
    assert not checker(9.5)
    assert not checker(9.0)


def test_convergence_checker_converges():
    """Should converge when energy stops changing."""
    checker = ConvergenceChecker(tol=1e-4, window=3, patience=2)

    # Rapidly decreasing energy
    for _ in range(5):
        checker(10.0)

    # Energy stabilizes
    for i in range(10):
        converged = checker(5.0 + i * 1e-6)
        if converged:
            break

    # After stabilization, should eventually converge
    checker2 = ConvergenceChecker(tol=1e-4, window=3, patience=2)
    for _ in range(10):
        checker2(1.0)
    # With constant energy, should converge
    result = checker2(1.0)
    assert result


def test_convergence_checker_reset():
    """Reset should clear history."""
    checker = ConvergenceChecker(tol=1e-4, window=3, patience=2)
    checker(5.0)
    checker(4.0)
    checker.reset()
    assert len(checker._history) == 0
    assert checker._patience_count == 0