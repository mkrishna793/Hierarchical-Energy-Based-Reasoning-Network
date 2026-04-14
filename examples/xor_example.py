"""HERBlib XOR Example — Train a HERB network on the XOR problem.

XOR is a challenging problem for energy-based models because CD learning
is unsupervised — it learns to model the data distribution, not to classify.
The supervised output head provides classification capability on top of
the learned representations.

This example demonstrates:
  1. Creating and configuring a HERB network
  2. Training with CD + supervised output head
  3. Inference and generative reconstruction
  4. Monitoring energy convergence
"""

import numpy as np
from herblib import HERB, HERBConfig

# XOR inputs and one-hot labels
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=np.float64)

# Create HERB network with stable hyperparameters
# damping=0.85 provides friction for convergence
# state_clip and weight_clip prevent numerical overflow
config = HERBConfig(
    layer_sizes=[2, 8, 6, 4],
    lr=0.01,
    lam=0.01,
    epsilon=0.01,
    mass=1.0,
    leapfrog_steps=10,
    cd_steps=1,
    damping=0.85,
    state_clip=5.0,
    weight_clip=3.0,
)

model = HERB(config, n_out=2)

print("=" * 50)
print("HERBlib — XOR Example")
print("=" * 50)
print(f"Network: {config.layer_sizes} -> {model.n_out} outputs")
print(f"Parameters: {model.summary()['n_params']}")
print(f"lr={config.lr}, eps={config.epsilon}, damping={config.damping}")
print()

# Train
print("Training...")
model.fit(X, y=y, epochs=200, batch_size=4, verbose=True)

# Inference
print("\nInference results:")
predictions = model.infer(X)
pred_np = np.array(predictions)
for i in range(4):
    pred_class = np.argmax(pred_np[i])
    true_class = np.argmax(y[i])
    marker = "<-- WRONG" if pred_class != true_class else ""
    print(f"  Input: {X[i]}  Target: {y[i]}  Pred: {pred_np[i].round(3)}  {marker}")

# Reconstruction
print("\nGenerative reconstruction:")
recon = model.reconstruct(X)
recon_np = np.array(recon)
for i in range(4):
    print(f"  Input: {X[i]}  Recon: {recon_np[i].round(3)}  MSE: {np.mean((X[i] - recon_np[i])**2):.4f}")

# Energy history
print(f"\nEnergy trajectory: {model.energy_history()[0]:.4f} -> {model.energy_history()[-1]:.4f}")
print(f"Energy decreased by: {model.energy_history()[0] - model.energy_history()[-1]:.4f}")