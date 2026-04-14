"""Weight initialization for HERBlib.

Uses Xavier/Glorot initialization for weight matrices and zero initialization
for biases, as specified in the HERB framework document.
"""

from herblib._types import HERBConfig, HERBWeights
from herblib._backend import randn, zeros


def initialize_weights(config: HERBConfig, n_out: int = None) -> HERBWeights:
    """Initialize HERB weights with Xavier/Glorot initialization.

    Args:
        config: HERBConfig with layer_sizes
        n_out: Output dimension for W_out. Defaults to layer_sizes[3] (d4).

    Returns:
        HERBWeights with initialized W, b, c, W_out, b_out
    """
    d1, d2, d3, d4 = config.layer_sizes
    if n_out is None:
        n_out = d4

    # Xavier initialization for weight matrices
    W = []
    for fan_in, fan_out in [(d1, d2), (d2, d3), (d3, d4)]:
        std = (2.0 / (fan_in + fan_out)) ** 0.5
        W.append(randn(fan_in, fan_out) * std)

    # Zero initialization for biases
    # b: visible biases for each interface (b1 for s1, b2 for s2, b3 for s3)
    b = [zeros(d1), zeros(d2), zeros(d3)]
    # c: hidden biases for each interface (c2 for s2, c3 for s3, c4 for s4)
    c = [zeros(d2), zeros(d3), zeros(d4)]

    # Output head: small random init
    W_out = randn(d4, n_out) * 0.01
    b_out = zeros(n_out)

    return HERBWeights(W=W, b=b, c=c, W_out=W_out, b_out=b_out)