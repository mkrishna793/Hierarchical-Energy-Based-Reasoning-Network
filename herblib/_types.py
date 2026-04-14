"""Core data structures for HERBlib."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Sequence


@dataclass
class HERBConfig:
    """Configuration for a HERB network. Immutable after creation."""

    layer_sizes: List[int]          # [d1, d2, d3, d4]
    lr: float = 0.01               # learning rate eta
    lam: float = 0.001             # regularization lambda
    epsilon: float = 0.05          # leapfrog step size
    mass: float = 1.0              # particle mass (same for all layers)
    cd_steps: int = 1              # CD-k steps
    leapfrog_steps: int = 10       # leapfrog steps per phase
    output_activation: str = "softmax"  # "softmax", "sigmoid", "linear"
    convergence_tol: float = 1e-4  # energy change threshold for convergence
    convergence_patience: int = 3  # consecutive windows below tol to declare convergence
    damping: float = 0.95          # momentum damping factor (1.0 = no damping)
    state_clip: float = 10.0       # clip layer states to [-clip, +clip]
    weight_clip: float = 5.0       # clip weight magnitudes after updates
    learning_method: str = "cd"    # "cd" (Contrastive Divergence) or "direct" (Direct Energy Gradient)

    def __post_init__(self):
        if len(self.layer_sizes) != 4:
            raise ValueError(f"HERB requires exactly 4 layers, got {len(self.layer_sizes)}")
        if any(s < 1 for s in self.layer_sizes):
            raise ValueError(f"All layer sizes must be >= 1, got {self.layer_sizes}")


@dataclass
class HERBState:
    """Mutable state of a HERB network: layer activations and momenta.

    s: [s1, s2, s3, s4] — each shape (B, d_l)
    p: [p1, p2, p3, p4] — each shape (B, d_l)
    """
    s: List
    p: List


@dataclass
class HERBWeights:
    """Learnable parameters of a HERB network.

    W: [W1, W2, W3] — shapes (d1,d2), (d2,d3), (d3,d4)
    b: [b1, b2, b3] — visible biases, shapes (d1,), (d2,), (d3,)
    c: [c2, c3, c4] — hidden biases, shapes (d2,), (d3,), (d4,)
    W_out: output readout weights, shape (d4, n_out)
    b_out: output readout bias, shape (n_out,)
    """
    W: List
    b: List
    c: List
    W_out: object
    b_out: object