"""Pareto frontier and scaling plots.

Generates:
- ASR vs. Detection Rate Pareto frontier (by layer region and client count)
- Scaling curves (ASR/detection as function of N)
- Per-round convergence plots
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="colorblind")

REGION_COLORS = {
    "early": "#1f77b4",
    "middle": "#ff7f0e",
    "late": "#d62728",
    "full": "#7f7f7f",
}

REGION_MARKERS = {
    "early": "o",
    "middle": "s",
    "late": "^",
    "full": "D",
}


def plot_pareto_frontier(
    results: dict,
    output_path: str | Path = "figures/pareto_frontier.pdf",
) -> None:
    """Plot ASR vs. Detection Rate Pareto frontier.

    Args:
        results: dict with keys like "late_krum" -> {N: {asr_mean, detection_rate}}
        output_path: where to save the figure
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
    client_counts = [8, 16, 32, 64]

    for ax, n in zip(axes, client_counts):
        for key, by_n in results.items():
            region = key.split("_")[0]
            if n not in by_n:
                continue
            data = by_n[n]
            ax.scatter(
                data["detection_rate"],
                data["asr_mean"],
                c=REGION_COLORS.get(region, "gray"),
                marker=REGION_MARKERS.get(region, "o"),
                s=100,
                label=region if ax == axes[0] else None,
                alpha=0.8,
                edgecolors="black",
                linewidth=0.5,
            )

        ax.set_xlabel("Detection Rate")
        ax.set_title(f"N = {n} clients")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    axes[0].set_ylabel("Attack Success Rate (ASR)")
    axes[0].legend(title="Layer Region", loc="upper left")

    plt.suptitle("Pareto Frontier: ASR vs. Detectability by Layer Region and Client Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Pareto frontier saved to {output_path}")


def plot_scaling_curves(
    results: dict,
    output_path: str | Path = "figures/scaling_curves.pdf",
) -> None:
    """Plot ASR and detection rate as function of N for each layer region."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for key, by_n in results.items():
        region = key.split("_")[0]
        ns = sorted(by_n.keys())
        asrs = [by_n[n]["asr_mean"] for n in ns]
        asr_stds = [by_n[n].get("asr_std", 0) for n in ns]
        det_rates = [by_n[n]["detection_rate"] for n in ns]

        color = REGION_COLORS.get(region, "gray")
        marker = REGION_MARKERS.get(region, "o")

        ax1.errorbar(
            ns, asrs, yerr=asr_stds,
            color=color, marker=marker, label=region,
            capsize=3, linewidth=2, markersize=8,
        )
        ax2.plot(
            ns, det_rates,
            color=color, marker=marker, label=region,
            linewidth=2, markersize=8,
        )

    ax1.set_xlabel("Number of Clients (N)")
    ax1.set_ylabel("Attack Success Rate")
    ax1.set_title("ASR vs. Client Count")
    ax1.legend(title="Layer Region")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks([8, 16, 32, 64])
    ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    ax2.set_xlabel("Number of Clients (N)")
    ax2.set_ylabel("Detection Rate")
    ax2.set_title("Detectability vs. Client Count")
    ax2.legend(title="Layer Region")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks([8, 16, 32, 64])
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Scaling curves saved to {output_path}")
