"""HERB — Hierarchical Energy-Based Reasoning Network.

Main user-facing class that ties together energy computation, Hamiltonian
leapfrog dynamics, contrastive divergence learning, and output emission.
"""

from __future__ import annotations
from typing import List, Optional, Union

from herblib._types import HERBConfig, HERBState, HERBWeights
from herblib._backend import (
    array, copy, sigmoid, softmax, to_numpy, xp,
    zeros, randn, mean,
)
from herblib.core.energy import total_energy
from herblib.dynamics.leapfrog import run_to_equilibrium
from herblib.learning.cd import ContrastiveDivergence
from herblib.learning.direct_energy import DirectEnergyLearning
from herblib.utils.init import initialize_weights
from herblib.utils.metrics import reconstruction_error, energy_tracker


class HERB:
    """Hierarchical Energy-Based Reasoning Network.

    A 4-layer energy-based bidirectional neural network that reasons by
    settling into low-energy equilibrium. No backpropagation through time.

    Usage:
        config = HERBConfig(layer_sizes=[784, 256, 64, 10])
        model = HERB(config)
        model.fit(X_train, epochs=50)
        predictions = model.infer(X_test)
    """

    def __init__(self, config: HERBConfig, n_out: int = None):
        self.config = config
        self.n_out = n_out if n_out is not None else config.layer_sizes[3]
        self.weights = initialize_weights(config, n_out=self.n_out)
        # Choose learning method
        if config.learning_method == "direct":
            self.learner = DirectEnergyLearning(config)
        else:
            self.learner = ContrastiveDivergence(config)
        self.cd = self.learner  # backward compat
        self._energy_log: List[float] = []
        self._recon_log: List[float] = []

    def fit(
        self,
        X,
        y=None,
        epochs: int = 10,
        batch_size: int = 32,
        verbose: bool = True,
    ) -> "HERB":
        """Train the network using contrastive divergence.

        If y is provided, also trains the output head (W_out, b_out) using
        supervised learning on the s4 representations.

        Args:
            X: Input data, shape (N, d1)
            y: Optional labels, shape (N, n_out). If provided, trains output head.
            epochs: Number of training epochs
            batch_size: Mini-batch size for CD
            verbose: Print training progress

        Returns:
            self
        """
        X = self._ensure_array(X)
        N = X.shape[0]
        if y is not None:
            y = self._ensure_array(y)

        for epoch in range(epochs):
            perm = self._random_permutation(N)
            epoch_energy = 0.0
            n_batches = 0

            for start in range(0, N, batch_size):
                idx = perm[start:start + batch_size]
                batch_X = X[idx]
                batch_y = y[idx] if y is not None else None

                # Learning step (CD or Direct Energy)
                self.weights, pos_state = self.learner.step(batch_X, self.weights)

                # Train output head if labels provided
                if batch_y is not None:
                    self._train_output_head(pos_state.s[3], batch_y)

                epoch_energy += float(mean(total_energy(pos_state, self.weights, self.config.lam)))
                n_batches += 1

            avg_energy = epoch_energy / max(n_batches, 1)
            self._energy_log.append(avg_energy)

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                msg = f"Epoch {epoch+1}/{epochs}  Energy: {avg_energy:.6f}"
                if self._recon_log:
                    msg += f"  Recon: {self._recon_log[-1]:.6f}"
                print(msg)

        return self

    def infer(self, x):
        """Run inference: clamp s1=x, settle to equilibrium, emit output.

        Args:
            x: Input data, shape (B, d1)

        Returns:
            Output predictions, shape (B, n_out)
        """
        x = self._ensure_array(x)
        state = self._init_state_from_data(x)
        state = run_to_equilibrium(
            state, self.weights, self.config,
            clamped_layers={0},
            max_steps=self.config.leapfrog_steps,
        )
        return self._emit_output(state.s[3])

    def reconstruct(self, x):
        """Generative reconstruction: clamp s1=x, settle, then reconstruct top-down.

        The top-down generative pass uses the transposed weight matrices
        to reconstruct what each lower layer should look like given the
        equilibrium state above it.

        Args:
            x: Input data, shape (B, d1)

        Returns:
            Reconstructed input, shape (B, d1)
        """
        x = self._ensure_array(x)
        state = self._init_state_from_data(x)
        state = run_to_equilibrium(
            state, self.weights, self.config,
            clamped_layers={0},
            max_steps=self.config.leapfrog_steps,
        )

        # Top-down generative pass: s4 -> s3 -> s2 -> s1
        s4 = state.s[3]
        s3_hat = sigmoid(s4 @ self.weights.W[2].T + self.weights.b[2])
        s2_hat = sigmoid(s3_hat @ self.weights.W[1].T + self.weights.b[1])
        s1_hat = sigmoid(s2_hat @ self.weights.W[0].T + self.weights.b[0])

        # Record reconstruction error
        err = reconstruction_error(x, s1_hat)
        self._recon_log.append(err)

        return s1_hat

    def energy_history(self) -> List[float]:
        """Return list of average energy per epoch during training."""
        return self._energy_log[:]

    def recon_history(self) -> List[float]:
        """Return list of reconstruction errors."""
        return self._recon_log[:]

    def summary(self) -> dict:
        """Return summary statistics of training."""
        return {
            "energy": energy_tracker(self._energy_log),
            "n_params": self._count_params(),
            "config": {
                "layer_sizes": self.config.layer_sizes,
                "lr": self.config.lr,
                "lam": self.config.lam,
                "epsilon": self.config.epsilon,
                "mass": self.config.mass,
                "leapfrog_steps": self.config.leapfrog_steps,
            },
        }

    # --- Private methods ---

    def _init_state_from_data(self, x) -> HERBState:
        """Initialize state with s1=clamped data, s2-s4 via forward pass."""
        s1 = copy(x)
        s2 = sigmoid(x @ self.weights.W[0] + self.weights.c[0])
        s3 = sigmoid(s2 @ self.weights.W[1] + self.weights.c[1])
        s4 = sigmoid(s3 @ self.weights.W[2] + self.weights.c[2])

        B = x.shape[0]
        p = [zeros((B, s.shape[1])) for s in [s1, s2, s3, s4]]

        return HERBState(s=[s1, s2, s3, s4], p=p)

    def _emit_output(self, s4):
        """Produce output from Layer 4 equilibrium state.

        y = f(W_out * s4 + b_out)
        """
        logits = s4 @ self.weights.W_out + self.weights.b_out

        if self.config.output_activation == "softmax":
            return softmax(logits, axis=-1)
        elif self.config.output_activation == "sigmoid":
            return sigmoid(logits)
        else:  # "linear"
            return logits

    def _train_output_head(self, s4, y, lr_factor: float = 1.0):
        """Train output head with simple gradient descent on cross-entropy.

        This is a linear classifier on top of s4 representations.
        """
        eta = self.config.lr * lr_factor
        logits = s4 @ self.weights.W_out + self.weights.b_out
        probs = softmax(logits, axis=-1)

        # Gradient of cross-entropy: (probs - y) / B
        B = s4.shape[0]
        diff = (probs - y) / B

        dW_out = s4.T @ diff
        db_out = mean(diff, axis=0)

        self.weights.W_out = self.weights.W_out - eta * dW_out
        self.weights.b_out = self.weights.b_out - eta * db_out

    def _ensure_array(self, data):
        """Convert input to backend array if needed."""
        if not isinstance(data, type(xp.zeros(1))):
            return array(data)
        return data

    def _random_permutation(self, n: int):
        """Generate a random permutation of indices 0..n-1."""
        if self.config.__class__.__module__ == "numpy":
            import numpy as _np
            return _np.random.permutation(n)
        # Generic approach
        indices = list(range(n))
        import random
        random.shuffle(indices)
        return indices

    def _count_params(self) -> int:
        """Count total number of learnable parameters."""
        total = 0
        for W in self.weights.W:
            total += int(W.shape[0] * W.shape[1])
        for b in self.weights.b:
            total += int(b.shape[0])
        for c in self.weights.c:
            total += int(c.shape[0])
        total += int(self.weights.W_out.shape[0] * self.weights.W_out.shape[1])
        total += int(self.weights.b_out.shape[0])
        return total