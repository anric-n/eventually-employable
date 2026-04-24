"""Basic tests for core modules."""

import numpy as np
import pytest


def test_layer_indices():
    """Test layer region splitting."""
    from fedlora_poison.model import LayerRegion, get_layer_indices

    indices = get_layer_indices(16, LayerRegion.EARLY)
    assert indices == list(range(0, 5))

    indices = get_layer_indices(16, LayerRegion.MIDDLE)
    assert indices == list(range(5, 10))

    indices = get_layer_indices(16, LayerRegion.LATE)
    assert indices == list(range(10, 16))

    indices = get_layer_indices(16, LayerRegion.FULL)
    assert indices == list(range(16))


def test_krum_select():
    """Test Krum selection with a simple example."""
    from fedlora_poison.defenses import krum_select

    # 5 clients, one outlier
    weights = [
        [np.array([1.0, 1.0])],
        [np.array([1.1, 0.9])],
        [np.array([0.9, 1.1])],
        [np.array([1.0, 1.0])],
        [np.array([10.0, 10.0])],  # outlier
    ]

    selected = krum_select(weights, num_malicious=1)
    assert selected != 4  # should not select the outlier


def test_trimmed_mean():
    """Test trimmed mean aggregation."""
    from fedlora_poison.defenses import trimmed_mean_aggregate

    weights = [
        [np.array([1.0, 2.0])],
        [np.array([1.0, 2.0])],
        [np.array([1.0, 2.0])],
        [np.array([100.0, 200.0])],  # outlier trimmed
        [np.array([-100.0, -200.0])],  # outlier trimmed
    ]

    result = trimmed_mean_aggregate(weights, trim_ratio=0.2)
    # After trimming top/bottom 1, should average the middle 3
    np.testing.assert_array_almost_equal(result[0], [1.0, 2.0])


def test_cosine_filter():
    """Test cosine similarity filter."""
    from fedlora_poison.defenses import cosine_filter

    weights = [
        [np.array([1.0, 0.0])],
        [np.array([0.9, 0.1])],
        [np.array([0.8, 0.2])],
        [np.array([-1.0, 0.0])],  # opposite direction
    ]

    accepted = cosine_filter(weights, threshold=0.5)
    assert 3 not in accepted  # opposite direction should be rejected


def test_checkpoint_manager(tmp_path):
    """Test checkpoint save and load."""
    from fedlora_poison.checkpointing import CheckpointManager

    mgr = CheckpointManager(tmp_path / "ckpt")

    assert mgr.get_resume_round() == 0

    weights = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])]
    mgr.save_round(
        round_num=3,
        global_weights=weights,
        metrics={"loss": 0.5},
        config={"num_clients": 8},
    )

    assert mgr.get_resume_round() == 4

    loaded = mgr.load_latest()
    assert loaded["last_completed_round"] == 3
    np.testing.assert_array_equal(loaded["global_weights"][0], weights[0])
