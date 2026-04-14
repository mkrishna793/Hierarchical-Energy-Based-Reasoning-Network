"""Backend abstraction for HERBlib — dispatches between NumPy and PyTorch."""

import os
import numpy as np

_BACKEND: str = os.environ.get("HERBLIB_BACKEND", "numpy")


def use(backend: str) -> None:
    """Set the global backend. One of 'numpy', 'torch-cpu', 'torch-cuda'."""
    global _BACKEND, xp
    if backend not in ("numpy", "torch-cpu", "torch-cuda"):
        raise ValueError(f"Unknown backend: {backend!r}")
    _BACKEND = backend
    _rebind_xp()


def _rebind_xp() -> None:
    global xp
    if _BACKEND == "numpy":
        xp = np
    else:
        import torch
        xp = torch


_rebind_xp()


def backend_name() -> str:
    return _BACKEND


def device() -> str:
    if _BACKEND == "torch-cuda":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


# --- Array creation helpers ---

def array(data, dtype=None):
    if _BACKEND == "numpy":
        return np.array(data, dtype=dtype or np.float64)
    import torch
    t = torch.tensor(data, dtype=dtype or torch.float64)
    return t.to(device())


def zeros(shape, dtype=None):
    if _BACKEND == "numpy":
        return np.zeros(shape, dtype=dtype or np.float64)
    import torch
    return torch.zeros(shape, dtype=dtype or torch.float64, device=device())


def zeros_like(a):
    if _BACKEND == "numpy":
        return np.zeros_like(a)
    import torch
    return torch.zeros_like(a)


def ones(shape, dtype=None):
    if _BACKEND == "numpy":
        return np.ones(shape, dtype=dtype or np.float64)
    import torch
    return torch.ones(shape, dtype=dtype or torch.float64, device=device())


def randn(*shape, dtype=None):
    if _BACKEND == "numpy":
        return np.random.randn(*shape).astype(dtype or np.float64)
    import torch
    return torch.randn(*shape, dtype=dtype or torch.float64, device=device())


def randn_like(a):
    if _BACKEND == "numpy":
        return np.random.randn(*a.shape)
    import torch
    return torch.randn_like(a)


def matmul(a, b):
    return a @ b


def dot(a, b):
    return a @ b


def sigmoid(a):
    if _BACKEND == "numpy":
        return 1.0 / (1.0 + np.exp(-a))
    import torch
    return torch.sigmoid(a)


def softmax(a, axis=-1):
    if _BACKEND == "numpy":
        e = np.exp(a - np.max(a, axis=axis, keepdims=True))
        return e / np.sum(e, axis=axis, keepdims=True)
    import torch
    return torch.softmax(a, dim=axis)


def clip(a, lo, hi):
    if _BACKEND == "numpy":
        return np.clip(a, lo, hi)
    import torch
    return torch.clamp(a, lo, hi)


def norm(a, axis=-1):
    if _BACKEND == "numpy":
        return np.linalg.norm(a, axis=axis)
    import torch
    return torch.norm(a, dim=axis)


def sum(a, axis=None):
    if _BACKEND == "numpy":
        return np.sum(a, axis=axis)
    import torch
    return torch.sum(a, dim=axis)


def mean(a, axis=0):
    if _BACKEND == "numpy":
        return np.mean(a, axis=axis)
    import torch
    return torch.mean(a.float(), dim=axis)


def to_numpy(a):
    """Always convert to numpy array, regardless of backend."""
    if _BACKEND == "numpy":
        return np.asarray(a)
    import torch
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def copy(a):
    if _BACKEND == "numpy":
        return a.copy()
    import torch
    return a.clone()


def outer(a, b):
    """Batched outer product: einsum('bi,bj->ij', a, b) / B."""
    B = a.shape[0]
    if _BACKEND == "numpy":
        return np.einsum("bi,bj->ij", a, b) / B
    import torch
    return torch.einsum("bi,bj->ij", a, b) / B