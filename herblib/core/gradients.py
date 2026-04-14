"""Gradient computations for HERBlib.

Per-layer gradients of E_total with respect to each layer state.
These are closed-form expressions — no numerical approximation needed.

grad_s1 = -W1 @ s2^T - b1 + 2*lam*s1   (actually -(s2 @ W1.T) - b1 + 2*lam*s1 in batch form)
grad_s2 = -(s1 @ W1) - (s3 @ W2.T) - b2 - c2 + 2*lam*s2
grad_s3 = -(s2 @ W2) - (s4 @ W3.T) - b3 - c3 + 2*lam*s3
grad_s4 = -(s3 @ W3) - c4 + 2*lam*s4
"""

from herblib._types import HERBState, HERBWeights
from herblib._backend import xp


def grad_s1(s1, s2, W1, b1, lam):
    """Gradient of E_total with respect to s1 (bottom layer).

    dE/ds1 = -W1 s2 - b1 + 2*lam*s1
    Batch form: -(s2 @ W1.T) - b1 + 2*lam*s1
    """
    return -(s2 @ W1.T) - b1 + 2.0 * lam * s1


def grad_s2(s1, s2, s3, W1, W2, b2, c2, lam):
    """Gradient of E_total with respect to s2 (first middle layer).

    dE/ds2 = -W1^T s1 - W2 s3 - b2 - c2 + 2*lam*s2
    Batch form: -(s1 @ W1) - (s3 @ W2.T) - b2 - c2 + 2*lam*s2
    """
    return -(s1 @ W1) - (s3 @ W2.T) - b2 - c2 + 2.0 * lam * s2


def grad_s3(s2, s3, s4, W2, W3, b3, c3, lam):
    """Gradient of E_total with respect to s3 (second middle layer).

    dE/ds3 = -W2^T s2 - W3 s4 - b3 - c3 + 2*lam*s3
    Batch form: -(s2 @ W2) - (s4 @ W3.T) - b3 - c3 + 2*lam*s3
    """
    return -(s2 @ W2) - (s4 @ W3.T) - b3 - c3 + 2.0 * lam * s3


def grad_s4(s3, s4, W3, c4, lam):
    """Gradient of E_total with respect to s4 (top layer).

    dE/ds4 = -W3^T s3 - c4 + 2*lam*s4
    Batch form: -(s3 @ W3) - c4 + 2*lam*s4
    """
    return -(s3 @ W3) - c4 + 2.0 * lam * s4


def compute_all_gradients(state: HERBState, weights: HERBWeights, lam: float):
    """Compute gradients for all 4 layers.

    Returns:
        List of 4 gradient arrays, one per layer, each shape (B, d_l)
    """
    s = state.s
    W, b, c = weights.W, weights.b, weights.c

    # b = [b1(d1), b2(d2), b3(d3)], c = [c2(d2), c3(d3), c4(d4)]
    # grad_s1: b1=b[0]; grad_s2: b2=b[1], c2=c[0]; grad_s3: b3=b[2], c3=c[1]; grad_s4: c4=c[2]
    g1 = grad_s1(s[0], s[1], W[0], b[0], lam)
    g2 = grad_s2(s[0], s[1], s[2], W[0], W[1], b[1], c[0], lam)
    g3 = grad_s3(s[1], s[2], s[3], W[1], W[2], b[2], c[1], lam)
    g4 = grad_s4(s[2], s[3], W[2], c[2], lam)

    return [g1, g2, g3, g4]