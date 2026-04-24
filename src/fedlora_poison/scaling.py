"""Client-count scaling sweep logic.

Runs experiments across N ∈ {8, 16, 32, 64} clients to evaluate how
the attack's effectiveness changes with federation size.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ScalingResult:
    """Results for one point in the scaling experiment."""

    num_clients: int
    layer_region: str
    defense: str
    seed: int
    asr_kl: float = 0.0
    perplexity: float = 0.0
    detected: bool = False
    rounds_completed: int = 0


@dataclass
class ScalingSweepConfig:
    """Configuration for the full scaling sweep."""

    client_counts: list[int] = field(default_factory=lambda: [8, 16, 32, 64])
    layer_regions: list[str] = field(
        default_factory=lambda: ["early", "middle", "late", "full"]
    )
    defenses: list[str] = field(
        default_factory=lambda: ["fedavg", "krum", "trimmed_mean", "cosine_filter"]
    )
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456])
    num_rounds: int = 10

    @property
    def total_experiments(self) -> int:
        return (
            len(self.client_counts)
            * len(self.layer_regions)
            * len(self.defenses)
            * len(self.seeds)
        )


def generate_experiment_matrix(config: ScalingSweepConfig) -> list[dict]:
    """Generate all experiment configurations for the sweep."""
    experiments = []
    for n in config.client_counts:
        for region in config.layer_regions:
            for defense in config.defenses:
                for seed in config.seeds:
                    experiments.append({
                        "num_clients": n,
                        "layer_region": region,
                        "defense": defense,
                        "seed": seed,
                        "num_rounds": config.num_rounds,
                    })

    logger.info(f"Generated {len(experiments)} experiment configurations")
    return experiments


def analyze_scaling(results: list[ScalingResult]) -> dict:
    """Analyze scaling trends across client counts."""
    analysis = {}

    # Group by (layer_region, defense)
    groups = {}
    for r in results:
        key = (r.layer_region, r.defense)
        if key not in groups:
            groups[key] = {}
        if r.num_clients not in groups[key]:
            groups[key][r.num_clients] = []
        groups[key][r.num_clients].append(r)

    for (region, defense), by_n in groups.items():
        key = f"{region}_{defense}"
        analysis[key] = {}
        for n, runs in sorted(by_n.items()):
            asrs = [r.asr_kl for r in runs]
            detect_rate = sum(1 for r in runs if r.detected) / len(runs)
            analysis[key][n] = {
                "asr_mean": float(np.mean(asrs)),
                "asr_std": float(np.std(asrs)),
                "detection_rate": detect_rate,
                "num_runs": len(runs),
            }

    return analysis
