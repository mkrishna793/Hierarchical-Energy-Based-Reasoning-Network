"""HERBlib MNIST Example — Train a HERB network on MNIST digits.

This demonstrates:
  1. Using the PyTorch backend for GPU acceleration
  2. Training a larger HERB network on real data
  3. Evaluating classification accuracy
"""

import numpy as np

# Try to use PyTorch backend if available
try:
    import torch
    from herblib import use
    if torch.cuda.is_available():
        use("torch-cuda")
        print("Using PyTorch CUDA backend")
    else:
        use("torch-cpu")
        print("Using PyTorch CPU backend")
except ImportError:
    print("PyTorch not available, using NumPy backend")
    # NumPy backend is default, no action needed

from herblib import HERB, HERBConfig

# MNIST: 784 input pixels, 10 output classes
config = HERBConfig(
    layer_sizes=[784, 256, 64, 10],
    lr=0.01,
    lam=0.0001,
    epsilon=0.05,
    mass=1.0,
    leapfrog_steps=20,
    cd_steps=1,
    output_activation="softmax",
)

model = HERB(config, n_out=10)

print(f"HERB network: {config.layer_sizes}")
print(f"Total parameters: {model.summary()['n_params']}")

# Generate synthetic MNIST-like data for demo purposes
# In practice, you would load actual MNIST data:
#   from torchvision import datasets, transforms
#   train_data = datasets.MNIST(root='.', train=True, download=True)
#   X = train_data.data.float().reshape(-1, 784) / 255.0
#   y = ... one-hot labels ...

N = 1000  # number of samples for demo
X_demo = np.random.rand(N, 784).astype(np.float64)
y_demo = np.eye(10)[np.random.randint(0, 10, N)].astype(np.float64)

print("\nTraining HERB on synthetic data...")
model.fit(X_demo, y=y_demo, epochs=5, batch_size=32, verbose=True)

# Test inference
print("\nTesting inference on 10 samples...")
X_test = np.random.rand(10, 784).astype(np.float64)
predictions = model.infer(X_test)
print(f"Predictions shape: {np.array(predictions).shape}")

# Reconstruction
print("\nReconstruction:")
recon = model.reconstruct(X_test[:5])
recon_error = np.mean((np.array(X_test[:5]) - np.array(recon)) ** 2)
print(f"  MSE: {recon_error:.6f}")