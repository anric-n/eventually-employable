"""Flower server strategies: FedAvg, Krum, TrimmedMean, CosineFilter.

Wraps Flower's built-in strategies where possible; implements custom
Byzantine-robust aggregation for Krum, Trimmed Mean, and Cosine Filter.
"""

from __future__ import annotations

import logging
from typing import Optional

import flwr as fl
import numpy as np
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from .defenses import cosine_filter, krum_select, trimmed_mean_aggregate

logger = logging.getLogger(__name__)


class FedAvgStrategy(FedAvg):
    """Standard FedAvg — simple weighted average of client updates."""

    pass


class KrumStrategy(FedAvg):
    """Krum: select the update closest to its neighbors (Blanchard et al. 2017)."""

    def __init__(self, num_malicious: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.num_malicious = num_malicious

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list,
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        if not results:
            return None, {}

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        all_weights = [w for w, _ in weights_results]
        selected_idx = krum_select(all_weights, self.num_malicious)

        logger.info(f"Round {server_round}: Krum selected client {selected_idx}")

        selected_weights = all_weights[selected_idx]
        return ndarrays_to_parameters(selected_weights), {"krum_selected": selected_idx}


class TrimmedMeanStrategy(FedAvg):
    """Trimmed Mean: remove top/bottom β fraction before averaging (Yin et al. 2018)."""

    def __init__(self, trim_ratio: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.trim_ratio = trim_ratio

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list,
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        if not results:
            return None, {}

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        all_weights = [w for w, _ in weights_results]
        aggregated = trimmed_mean_aggregate(all_weights, self.trim_ratio)

        logger.info(f"Round {server_round}: Trimmed Mean with β={self.trim_ratio}")
        return ndarrays_to_parameters(aggregated), {}


class CosineFilterStrategy(FedAvg):
    """Cosine similarity filter: reject updates too dissimilar to the median."""

    def __init__(self, threshold: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list,
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        if not results:
            return None, {}

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        all_weights = [w for w, _ in weights_results]
        num_examples = [n for _, n in weights_results]

        accepted_indices = cosine_filter(all_weights, self.threshold)
        logger.info(
            f"Round {server_round}: Cosine filter accepted "
            f"{len(accepted_indices)}/{len(all_weights)} clients"
        )

        # Weighted average of accepted updates
        total_examples = sum(num_examples[i] for i in accepted_indices)
        aggregated = [
            np.zeros_like(all_weights[0][j]) for j in range(len(all_weights[0]))
        ]
        for i in accepted_indices:
            weight = num_examples[i] / total_examples
            for j in range(len(aggregated)):
                aggregated[j] += all_weights[i][j] * weight

        return ndarrays_to_parameters(aggregated), {
            "num_accepted": len(accepted_indices),
            "num_rejected": len(all_weights) - len(accepted_indices),
        }


def get_strategy(
    name: str,
    num_clients: int,
    fraction_fit: float = 1.0,
    **kwargs,
) -> fl.server.strategy.Strategy:
    """Factory: get a strategy by name."""
    common = dict(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
    )

    strategies = {
        "fedavg": FedAvgStrategy,
        "krum": KrumStrategy,
        "trimmed_mean": TrimmedMeanStrategy,
        "cosine_filter": CosineFilterStrategy,
    }

    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}. Choose from {list(strategies.keys())}")

    return strategies[name](**common, **kwargs)
