"""Layer-targeted poisoning attacks and PoisonedFL-style multi-round consistency.

Attack strategies:
- Scale-up: amplify the malicious update direction
- Constrain: project update to stay within norm bounds
- Multi-round: maintain consistent poisoning direction across rounds
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AttackConfig:
    """Configuration for the attack strategy."""

    strategy: str = "scale"  # "scale", "constrain", "lie"
    scale_factor: float = 3.0  # for scale strategy
    norm_bound: float | None = None  # for constrain strategy
    lie_z: float = 1.0  # for LIE strategy (z-score multiplier)
    multi_round: bool = True  # PoisonedFL-style consistency


def apply_attack_strategy(
    global_params: list[np.ndarray],
    local_params: list[np.ndarray],
    config: AttackConfig,
) -> list[np.ndarray]:
    """Apply attack strategy to scale/modify the malicious update."""
    # Compute the update delta
    deltas = [local - glob for local, glob in zip(local_params, global_params)]

    if config.strategy == "scale":
        deltas = _scale_attack(deltas, config.scale_factor)
    elif config.strategy == "constrain":
        deltas = _constrain_attack(deltas, config.norm_bound)
    elif config.strategy == "lie":
        deltas = _lie_attack(deltas, config.lie_z)
    else:
        logger.warning(f"Unknown strategy '{config.strategy}', using raw update")

    # Return global + scaled delta
    return [glob + delta for glob, delta in zip(global_params, deltas)]


def _scale_attack(
    deltas: list[np.ndarray], scale_factor: float
) -> list[np.ndarray]:
    """Simply amplify the update direction."""
    return [d * scale_factor for d in deltas]


def _constrain_attack(
    deltas: list[np.ndarray], norm_bound: float | None
) -> list[np.ndarray]:
    """Scale up while keeping L2 norm below bound (evade norm clipping)."""
    if norm_bound is None:
        return deltas

    flat = np.concatenate([d.flatten() for d in deltas])
    current_norm = np.linalg.norm(flat)

    if current_norm > norm_bound:
        scale = norm_bound / current_norm
        return [d * scale for d in deltas]
    return deltas


def _lie_attack(
    deltas: list[np.ndarray], z: float
) -> list[np.ndarray]:
    """LIE-style: shift update by z standard deviations toward attack direction.

    In true LIE, the attacker estimates the mean and std of benign updates.
    Here we approximate by scaling the delta to be z stddevs from zero.
    """
    flat = np.concatenate([d.flatten() for d in deltas])
    std = np.std(flat)
    if std > 0:
        target_norm = z * std * np.sqrt(len(flat))
        current_norm = np.linalg.norm(flat)
        if current_norm > 0:
            scale = target_norm / current_norm
            return [d * scale for d in deltas]
    return deltas
