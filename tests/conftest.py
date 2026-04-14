"""Shared test fixtures for HERBlib tests."""

import pytest
import numpy as np


@pytest.fixture
def numpy_backend():
    """Ensure NumPy backend is active."""
    import herblib
    herblib.use("numpy")
    yield "numpy"
    herblib.use("numpy")


@pytest.fixture
def simple_config():
    """Small HERB config for fast tests."""
    from herblib._types import HERBConfig
    return HERBConfig(
        layer_sizes=[4, 3, 2, 2],
        lr=0.01,
        lam=0.001,
        epsilon=0.05,
        mass=1.0,
        cd_steps=1,
        leapfrog_steps=5,
    )


@pytest.fixture
def small_weights(simple_config):
    """Pre-initialized small weights for testing."""
    from herblib.utils.init import initialize_weights
    return initialize_weights(simple_config)


@pytest.fixture
def small_state(simple_config, small_weights):
    """Random state for a small network."""
    from herblib._types import HERBState
    from herblib._backend import randn

    d1, d2, d3, d4 = simple_config.layer_sizes
    B = 2
    s = [randn(B, d) for d in [d1, d2, d3, d4]]
    p = [randn(B, d) * 0.01 for d in [d1, d2, d3, d4]]
    return HERBState(s=s, p=p)