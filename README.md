# HERBlib

**Hierarchical Energy-Based Reasoning Network** — A physics-native, RBM-rooted, bidirectional neural network with no backpropagation through time.

Designed & Conceived by **Mohana Krishna**

---

## What is HERB?

HERB is a novel neural network architecture that replaces the standard input→hidden→output pipeline with a **multi-layer system that evolves toward equilibrium**. Reasoning emerges from energy minimization — not from a forced forward pass.

**Three big ideas fused together:**

| Idea | Role in HERB |
|------|-------------|
| **Restricted Boltzmann Machines (RBMs)** | Local energy function at each layer pair |
| **Hamiltonian Dynamics** | How the network evolves over time (leapfrog integrator) |
| **Hierarchical Predictive Coding** | Bidirectional, top-down and bottom-up information flow |

**Result:** A network that reasons by settling into low-energy states, generates reconstructions top-down, learns without BPTT, and scales by adding neurons.

---

## Installation

```bash
pip install -e .
```

For GPU acceleration (optional):
```bash
pip install -e ".[torch]"
```

For development (tests, plotting):
```bash
pip install -e ".[dev]"
```

---

## Quick Start

### Create and Train

```python
from herblib import HERB, HERBConfig

# Define network: 4 layers from raw data to abstract understanding
config = HERBConfig(
    layer_sizes=[784, 256, 64, 10],  # input → feature → generalization → understanding
    lr=0.01,                          # learning rate
    lam=0.001,                        # regularization
    epsilon=0.01,                     # leapfrog step size
    damping=0.85,                     # momentum friction for stability
    learning_method="direct",         # "cd" or "direct"
)

model = HERB(config, n_out=10)        # 10 output classes

# Train
model.fit(X_train, y=y_train, epochs=50, batch_size=32)

# Infer
predictions = model.infer(X_test)

# Generate (top-down reconstruction)
reconstruction = model.reconstruct(X_test)
```

### Switch Backends

```python
import herblib

herblib.use("numpy")        # CPU, default
herblib.use("torch-cpu")    # PyTorch CPU
herblib.use("torch-cuda")  # PyTorch GPU (if available)
```

---

## How HERB Works

### The 4-Layer Architecture

HERB uses exactly 4 layers, each with a distinct role:

```
Layer 4 (s⁴)  ←  Understanding + Solution Emission
     ↕                    ↕
Layer 3 (s³)  ←  Generalization (core reasoning)
     ↕                    ↕
Layer 2 (s²)  ←  Feature Extraction
     ↕                    ↕
Layer 1 (s¹)  ←  Raw Extraction (receives input)
```

All layers are **active simultaneously** and influence each other **bidirectionally** — information flows both up and down through energy gradients.

### Bidirectionality

In HERB, bidirectionality is not an architectural choice — it's a **mathematical consequence** of how energy gradients work:

- Layer 2 is simultaneously pulled by Layer 1 (bottom-up) AND Layer 3 (top-down)
- The same weight matrix `W` is used in both directions (weight tying)
- `Wˡ` goes bottom-up, `Wˡᵀ` (transpose) goes top-down

### The Energy Function

The total energy assigns a score to every joint configuration of all 4 layers:

```
E_total = E₁(s¹,s²) + E₂(s²,s³) + E₃(s³,s⁴) + λ·Σ||sˡ||²
```

Where each pairwise energy is RBM-style:

```
E_ℓ(sˡ, sˡ⁺¹) = -sˡᵀ Wˡ sˡ⁺¹ - bˡᵀ sˡ - cˡ⁺¹ᵀ sˡ⁺¹
```

- Low energy = good configuration (consistent, coherent)
- The `λ` regularization prevents runaway activations

### Hamiltonian Dynamics (No BPTT)

Instead of forward pass + backward pass, HERB uses **Hamilton's equations** from classical mechanics:

```
dsˡ/dt = pˡ / mˡ           (position update)
dpˡ/dt = -∇_{sˡ} E_total   (momentum update = force)
```

Each layer has a **position** (state `s`) and **momentum** (`p`). The system evolves via the **leapfrog (Störmer-Verlet) integrator**:

1. **Half-step momentum:** `p^{t+½} = damping · p^t - (ε/2) · ∇E`
2. **Full-step position:** `s^{t+1} = s^t + ε · (p^{t+½} / m)`
3. **Half-step momentum:** `p^{t+1} = damping · p^{t+½} - (ε/2) · ∇E`

Key: each step only uses the **current** energy gradient — never gradients of gradients. **BPTT is eliminated by design.**

### Learning Rules

HERBlib supports two learning methods:

#### 1. Contrastive Divergence (`learning_method="cd"`)

The classic HERB learning rule. Compares two equilibria:

| Phase | What happens |
|-------|-------------|
| **Positive (data)** | Clamp s¹ = input, run leapfrog on s², s³, s⁴ → equilibrium |
| **Negative (free)** | Release all clamps, run leapfrog on all layers → equilibrium |

Weight update:
```
ΔWˡ = η · (<sˡ sˡ⁺¹ᵀ>_data - <sˡ sˡ⁺¹ᵀ>_model)
```

#### 2. Direct Energy Gradient (`learning_method="direct"`)

**~2x faster.** No negative phase needed — uses energy gradients already computed during the leapfrog:

```
ΔWˡ = η · <sˡ sˡ⁺¹ᵀ>_data     (just the positive phase correlations)
```

Weight clipping + regularization replace the negative phase's role in preventing unbounded growth.

### Output Emission

After equilibrium, Layer 4 holds the network's "understanding." The output is read off:

```
y = f(W_out · s⁴* + b_out)
```

Where `f` is softmax (classification), sigmoid (binary), or identity (regression).

### Generative Reconstruction

From any equilibrium state, HERB can reconstruct lower layers top-down:

```
s³ = σ(W³ᵀ s⁴ + b³)
s² = σ(W²ᵀ s³ + b²)
s¹ = σ(W¹ᵀ s² + b¹)    ← reconstruction of the input
```

This provides a built-in self-consistency check.

---

## Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `layer_sizes` | required | 4 integers: [d1, d2, d3, d4] |
| `lr` | 0.01 | Learning rate η |
| `lam` | 0.001 | Regularization λ (prevents runaway activations) |
| `epsilon` | 0.05 | Leapfrog step size (smaller = more accurate, slower) |
| `mass` | 1.0 | Particle mass m (larger = more stable, slower response) |
| `leapfrog_steps` | 10 | Leapfrog steps per inference phase |
| `cd_steps` | 1 | CD-k steps (only for CD learning) |
| `damping` | 0.95 | Momentum friction (0.85-0.95 typical; 1.0 = no damping) |
| `state_clip` | 10.0 | Clip layer states to [-clip, +clip] |
| `weight_clip` | 5.0 | Clip weight magnitudes after updates |
| `learning_method` | "cd" | "cd" (Contrastive Divergence) or "direct" (Direct Energy) |
| `output_activation` | "softmax" | "softmax", "sigmoid", or "linear" |
| `convergence_tol` | 1e-4 | Energy change threshold for convergence |
| `convergence_patience` | 3 | Consecutive windows below tol to declare convergence |

---

## Library Architecture

```
herblib/
├── _backend.py          ← NumPy / PyTorch backend abstraction
├── _types.py            ← HERBConfig, HERBState, HERBWeights
├── core/
│   ├── energy.py        ← pairwise_energy(), total_energy()
│   └── gradients.py    ← grad_s1..s4 (closed-form, verified vs finite differences)
├── dynamics/
│   ├── leapfrog.py     ← leapfrog_step(), run_to_equilibrium()
│   └── equilibrium.py  ← ConvergenceChecker
├── learning/
│   ├── cd.py           ← ContrastiveDivergence (2-phase)
│   └── direct_energy.py← DirectEnergyLearning (1-phase, ~2x faster)
├── network/
│   ├── herb.py         ← HERB class: fit(), infer(), reconstruct()
│   └── multi_herb.py   ← MultiHERB (shared-state coupling)
└── utils/
    ├── init.py         ← Xavier weight initialization
    └── metrics.py      ← reconstruction_error(), energy_tracker()
```

---

## Examples

### XOR Problem

```python
import numpy as np
from herblib import HERB, HERBConfig

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64)
y = np.array([[1,0],[0,1],[0,1],[1,0]], dtype=np.float64)

config = HERBConfig(
    layer_sizes=[2, 8, 6, 4],
    lr=0.01, lam=0.01, epsilon=0.01,
    damping=0.85, learning_method="direct",
)
model = HERB(config, n_out=2)
model.fit(X, y=y, epochs=200, batch_size=4)
predictions = model.infer(X)
```

### MNIST (with GPU)

```python
import herblib
herblib.use("torch-cuda")  # GPU acceleration

from herblib import HERB, HERBConfig

config = HERBConfig(
    layer_sizes=[784, 256, 64, 10],
    lr=0.01, lam=0.0001, epsilon=0.05,
    leapfrog_steps=20, learning_method="direct",
)
model = HERB(config, n_out=10)
model.fit(X_mnist, y=y_mnist, epochs=50, batch_size=64)
```

### Multi-Modal Fusion

```python
from herblib import HERB, MultiHERB, HERBConfig

# Two HERB networks sharing Layer 3 for multi-modal fusion
vision_net = HERB(HERBConfig(layer_sizes=[784, 128, 64, 10]))
language_net = HERB(HERBConfig(layer_sizes=[512, 128, 64, 10]))

# Share Layer 3 (index 2) between both networks
multi = MultiHERB(
    networks=[vision_net, language_net],
    shared_layers={(0, 2): 0, (1, 2): 0},  # both share layer 3
)
multi.fit([X_vision, X_language], epochs=50)
```

### Generative Reconstruction

```python
model.fit(X_train, epochs=50)

# After training, reconstruct inputs from the network's understanding
reconstruction = model.reconstruct(X_test)
mse = np.mean((X_test - np.array(reconstruction)) ** 2)
print(f"Reconstruction MSE: {mse:.4f}")
```

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

All 30 tests pass, including finite-difference gradient verification for all 4 layers.

---

## Parameter Summary

| Symbol | Meaning | Typical Range |
|--------|---------|-------------|
| sˡ | Layer state vector | Computed by dynamics |
| pˡ | Layer momentum | Initialized N(0,1), discarded at equilibrium |
| Wˡ | Weight matrix | Xavier init: N(0, 2/(d_in+d_out)) |
| bˡ, cˡ | Bias vectors | Zero init |
| m | Layer mass | 0.5–2.0 (heavier = more stable) |
| ε | Leapfrog step size | 0.001–0.1 |
| T | Leapfrog steps | 10–200 |
| λ | Regularization | 0.001–0.01 |
| η | Learning rate | 0.001–0.05 |

---

## What HERB Achieves

- **Hierarchical abstraction** — 4 layers from raw data to understanding
- **Inherent bidirectionality** — energy gradients flow both ways automatically
- **Physics-native reasoning** — reasoning = settling, like a ball rolling to a valley
- **Generative capability** — reconstruct inputs from internal representations
- **BPTT-free learning** — no gradients through time, ever
- **Simple scalability** — add neurons by growing weight matrices
- **Multi-network composition** — couple HERB instances via shared layers

---

## License

MIT