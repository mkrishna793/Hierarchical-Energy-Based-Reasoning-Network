"""Energy function computations for HERBlib.

Implements:
- pairwise_energy: E_l(s^l, s^{l+1}) = -s^l^T W^l s^{l+1} - b^l^T s^l - c^{l+1}^T s^{l+1}
- total_energy: E_total = E1 + E2 + E3 + lam * sum(||s^l||^2)
"""

from herblib._types import HERBState, HERBWeights
from herblib._backend import xp


def pairwise_energy(s_l, s_lp1, W_l, b_l, c_lp1):
    """Compute RBM-style pairwise energy between adjacent layers.

    Args:
        s_l: Layer l state, shape (B, d_l)
        s_lp1: Layer l+1 state, shape (B, d_{l+1})
        W_l: Weight matrix, shape (d_l, d_{l+1})
        b_l: Visible bias, shape (d_l,)
        c_lp1: Hidden bias, shape (d_{l+1},)

    Returns:
        Energy values, shape (B,)
    """
    # -s^l^T W^l s^{l+1} — batched: sum over both dims after matmul
    interaction = xp.sum(s_l @ W_l * s_lp1, axis=-1)
    # -b^l^T s^l
    bias_l = xp.sum(s_l * b_l, axis=-1)
    # -c^{l+1}^T s^{l+1}
    bias_lp1 = xp.sum(s_lp1 * c_lp1, axis=-1)
    return -interaction - bias_l - bias_lp1


def total_energy(state: HERBState, weights: HERBWeights, lam: float):
    """Compute total system energy.

    E_total = E1(s1,s2) + E2(s2,s3) + E3(s3,s4) + lam * (||s1||^2 + ||s2||^2 + ||s3||^2 + ||s4||^2)
    """
    s = state.s
    W, b, c = weights.W, weights.b, weights.c

    # c = [c2, c3, c4] indexed as c[0], c[1], c[2]
    # E1: b1, c2; E2: b2, c3; E3: b3, c4
    E = pairwise_energy(s[0], s[1], W[0], b[0], c[0])
    E = E + pairwise_energy(s[1], s[2], W[1], b[1], c[1])
    E = E + pairwise_energy(s[2], s[3], W[2], b[2], c[2])

    # Regularization: lam * sum(||s^l||^2)
    for s_l in s:
        E = E + lam * xp.sum(s_l ** 2, axis=-1)

    return E