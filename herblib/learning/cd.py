"""Contrastive Divergence learning for HERBlib.

Implements CD-k learning: compare data-clamped equilibrium (positive phase)
with free-running equilibrium (negative phase) and update weights to
make the data-clamped states lower in energy.

Weight update: dW^l = eta * (<s^l s^{l+1}^T>_data - <s^l s^{l+1}^T>_model)
Bias update:  db^l = eta * (<s^l>_data - <s^l>_model)
              dc^{l+1} = eta * (<s^{l+1}>_data - <s^{l+1}>_model)
"""

from herblib._types import HERBConfig, HERBState, HERBWeights
from herblib.dynamics.leapfrog import run_to_equilibrium
from herblib._backend import xp, outer, mean, copy, clip


class ContrastiveDivergence:
    """Contrastive Divergence learner for HERB networks."""

    def __init__(self, config: HERBConfig):
        self.config = config

    def positive_phase(self, x_batch, weights: HERBWeights) -> HERBState:
        """Run positive phase: clamp s1 to data, evolve s2-s4 to equilibrium.

        Args:
            x_batch: Input data, shape (B, d1)
            weights: Current HERBWeights

        Returns:
            HERBState at data-clamped equilibrium
        """
        state = self._init_state_from_data(x_batch, weights)
        state = run_to_equilibrium(
            state, weights, self.config,
            clamped_layers={0},
            max_steps=self.config.leapfrog_steps,
        )
        return state

    def negative_phase(self, weights: HERBWeights, n_samples: int = 1) -> HERBState:
        """Run negative phase: free-run all layers from random init.

        Args:
            weights: Current HERBWeights
            n_samples: Number of fantasy particles

        Returns:
            HERBState at model equilibrium
        """
        from herblib._backend import randn
        state = self._init_state_random(weights, n_samples)
        state = run_to_equilibrium(
            state, weights, self.config,
            clamped_layers=set(),
            max_steps=self.config.leapfrog_steps,
        )
        return state

    def compute_updates(
        self,
        pos_state: HERBState,
        neg_state: HERBState,
    ) -> tuple:
        """Compute weight and bias updates from positive and negative phases.

        Returns:
            (dW, db, dc) — lists of updates for W, b, c
        """
        eta = self.config.lr
        dW = []
        db = []
        dc = []

        for l in range(3):
            # <s^l s^{l+1}^T> = batched outer product averaged over batch
            pos_corr = outer(pos_state.s[l], pos_state.s[l + 1])
            neg_corr = outer(neg_state.s[l], neg_state.s[l + 1])
            dW.append(eta * (pos_corr - neg_corr))

            # Bias updates
            db.append(eta * (mean(pos_state.s[l], axis=0) - mean(neg_state.s[l], axis=0)))
            dc.append(eta * (mean(pos_state.s[l + 1], axis=0) - mean(neg_state.s[l + 1], axis=0)))

        return dW, db, dc

    def step(self, x_batch, weights: HERBWeights) -> tuple:
        """One full CD step: positive phase -> negative phase -> compute updates.

        Args:
            x_batch: Input data, shape (B, d1)
            weights: Current HERBWeights

        Returns:
            (updated_weights, pos_state) — weights after CD update, and positive phase state
        """
        pos_state = self.positive_phase(x_batch, weights)
        neg_state = self.negative_phase(weights, n_samples=x_batch.shape[0])
        dW, db, dc = self.compute_updates(pos_state, neg_state)

        # Apply updates with weight clipping
        wclip = self.config.weight_clip
        new_W = [clip(weights.W[l] + dW[l], -wclip, wclip) for l in range(3)]
        new_b = [weights.b[l] + db[l] for l in range(3)]
        new_c = [weights.c[l] + dc[l] for l in range(3)]

        new_weights = HERBWeights(
            W=new_W,
            b=new_b,
            c=new_c,
            W_out=copy(weights.W_out),
            b_out=copy(weights.b_out),
        )
        return new_weights, pos_state

    def _init_state_from_data(self, x_batch, weights: HERBWeights) -> HERBState:
        """Initialize state with s1=clamped data, s2-s4 via forward pass through weights."""
        from herblib._backend import sigmoid, zeros

        s1 = copy(x_batch)
        s2 = sigmoid(x_batch @ weights.W[0] + weights.c[0])
        s3 = sigmoid(s2 @ weights.W[1] + weights.c[1])
        s4 = sigmoid(s3 @ weights.W[2] + weights.c[2])

        B = x_batch.shape[0]
        p = [zeros((B, d)) for d in [s1.shape[1], s2.shape[1], s3.shape[1], s4.shape[1]]]

        return HERBState(s=[s1, s2, s3, s4], p=p)

    def _init_state_random(self, weights: HERBWeights, n_samples: int) -> HERBState:
        """Initialize all layer states randomly."""
        from herblib._backend import randn, sigmoid

        sizes = [w.shape[0] if i == 0 else w.shape[1] for i, w in enumerate(weights.W)]
        # Derive from weight shapes
        d1 = weights.W[0].shape[0]
        d2 = weights.W[0].shape[1]
        d3 = weights.W[1].shape[1]
        d4 = weights.W[2].shape[1]
        dims = [d1, d2, d3, d4]

        s = [sigmoid(randn(n_samples, d)) for d in dims]
        p = [randn(n_samples, d) * 0.01 for d in dims]

        return HERBState(s=s, p=p)