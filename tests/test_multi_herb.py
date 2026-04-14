"""Tests for MultiHERB coupling."""

import numpy as np
import pytest


def test_multi_herb_init(simple_config, numpy_backend):
    """MultiHERB should initialize with shared layer configuration."""
    from herblib import HERB, MultiHERB

    net_a = HERB(simple_config)
    net_b = HERB(simple_config)

    # Share Layer 3 (index 2) between networks
    shared = {(0, 2): 0, (1, 2): 0}

    multi = MultiHERB([net_a, net_b], shared_layers=shared)
    assert len(multi.networks) == 2
    assert len(multi._groups) == 1


def test_multi_herb_infer_runs(simple_config, numpy_backend):
    """MultiHERB.infer should run without error."""
    from herblib import HERB, MultiHERB

    net_a = HERB(simple_config)
    net_b = HERB(simple_config)

    shared = {(0, 2): 0, (1, 2): 0}
    multi = MultiHERB([net_a, net_b], shared_layers=shared)

    X_a = np.random.randn(2, simple_config.layer_sizes[0])
    X_b = np.random.randn(2, simple_config.layer_sizes[0])

    outputs = multi.infer([X_a, X_b])
    assert len(outputs) == 2