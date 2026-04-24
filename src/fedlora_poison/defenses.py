"""Byzantine-robust aggregation defenses: Krum, Trimmed Mean, Cosine Filter.

These are the defenses the attacker must evade. We implement them both as
standalone functions (for testing) and as Flower strategy wrappers (in server.py).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _flatten(weights: list[np.ndarray]) -> np.ndarray:
    """Flatten a list of weight arrays into a single 1D vector."""
    return np.concatenate([w.flatten() for w in weights])


def krum_select(
    all_weights: list[list[np.ndarray]],
    num_malicious: int = 1,
) -> int:
    """Multi-Krum: select the update with minimum sum of distances to nearest neighbors.

    Given n clients and f suspected malicious, each client computes distances
    to all others, sums the (n - f - 2) closest, and we pick the minimum.
    """
    n = len(all_weights)
    f = num_malicious

    # Flatten each client's weights
    flat = [_flatten(w) for w in all_weights]

    # Pairwise squared distances
    scores = np.zeros(n)
    for i in range(n):
        distances = []
        for j in range(n):
            if i != j:
                diff = flat[i] - flat[j]
                distances.append(np.dot(diff, diff))
        distances.sort()
        # Sum the (n - f - 2) smallest distances
        num_closest = max(1, n - f - 2)
        scores[i] = sum(distances[:num_closest])

    selected = int(np.argmin(scores))
    logger.debug(f"Krum scores: {scores}, selected: {selected}")
    return selected


def trimmed_mean_aggregate(
    all_weights: list[list[np.ndarray]],
    trim_ratio: float = 0.1,
) -> list[np.ndarray]:
    """Coordinate-wise trimmed mean: remove top/bottom β fraction per dimension.

    For each parameter coordinate, sort the n values, remove the top and bottom
    β*n values, and average the rest.
    """
    n = len(all_weights)
    num_trim = max(1, int(trim_ratio * n))

    num_params = len(all_weights[0])
    result = []

    for param_idx in range(num_params):
        # Stack all clients' values for this parameter
        stacked = np.stack([all_weights[i][param_idx] for i in range(n)])
        # Sort along client axis
        sorted_vals = np.sort(stacked, axis=0)
        # Trim top and bottom
        trimmed = sorted_vals[num_trim : n - num_trim]
        # Average remaining
        result.append(np.mean(trimmed, axis=0))

    return result


def cosine_filter(
    all_weights: list[list[np.ndarray]],
    threshold: float = 0.1,
) -> list[int]:
    """Cosine similarity filter: reject updates too dissimilar to the median.

    Computes median update, then rejects any client whose cosine similarity
    to the median falls below (1 - threshold).
    """
    n = len(all_weights)
    flat = [_flatten(w) for w in all_weights]
    flat_arr = np.stack(flat)

    # Compute coordinate-wise median
    median = np.median(flat_arr, axis=0)
    median_norm = np.linalg.norm(median)

    if median_norm < 1e-10:
        return list(range(n))

    accepted = []
    for i in range(n):
        client_norm = np.linalg.norm(flat[i])
        if client_norm < 1e-10:
            continue
        cos_sim = np.dot(flat[i], median) / (client_norm * median_norm)
        if cos_sim >= (1.0 - threshold):
            accepted.append(i)
        else:
            logger.info(f"Cosine filter rejected client {i} (sim={cos_sim:.4f})")

    if not accepted:
        logger.warning("Cosine filter rejected ALL clients, accepting all as fallback")
        return list(range(n))

    return accepted
