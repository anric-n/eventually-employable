"""Checkpoint save/resume for spot instance resilience.

After every FL round, saves:
- Round number
- Global LoRA weights
- Per-client metrics
- Experiment config

On startup, detects existing checkpoint and resumes from last completed round.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages experiment checkpoints for spot instance resilience."""

    def __init__(self, checkpoint_dir: str | Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @property
    def metadata_path(self) -> Path:
        return self.checkpoint_dir / "metadata.json"

    @property
    def weights_path(self) -> Path:
        return self.checkpoint_dir / "global_weights.npz"

    def save_round(
        self,
        round_num: int,
        global_weights: list[np.ndarray],
        metrics: dict[str, Any],
        config: dict[str, Any],
    ) -> None:
        """Save checkpoint after completing a round."""
        # Save weights
        np.savez(
            self.weights_path,
            *global_weights,
        )

        # Save metadata
        metadata = {
            "last_completed_round": round_num,
            "config": config,
            "metrics_history": metrics,
        }
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Checkpoint saved: round {round_num} -> {self.checkpoint_dir}")

    def load_latest(self) -> Optional[dict]:
        """Load the most recent checkpoint. Returns None if no checkpoint exists."""
        if not self.metadata_path.exists():
            logger.info("No checkpoint found, starting fresh")
            return None

        with open(self.metadata_path) as f:
            metadata = json.load(f)

        # Load weights
        if self.weights_path.exists():
            data = np.load(self.weights_path)
            weights = [data[f"arr_{i}"] for i in range(len(data.files))]
            metadata["global_weights"] = weights
        else:
            metadata["global_weights"] = None

        logger.info(
            f"Loaded checkpoint: round {metadata['last_completed_round']}"
        )
        return metadata

    def get_resume_round(self) -> int:
        """Get the round to resume from (0 if no checkpoint)."""
        checkpoint = self.load_latest()
        if checkpoint is None:
            return 0
        return checkpoint["last_completed_round"] + 1

    def clear(self) -> None:
        """Delete all checkpoint files."""
        if self.weights_path.exists():
            self.weights_path.unlink()
        if self.metadata_path.exists():
            self.metadata_path.unlink()
        logger.info("Checkpoint cleared")
