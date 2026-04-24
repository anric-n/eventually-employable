"""CLI entry point for experiments."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


_CONFIG_PATH = str((Path(__file__).resolve().parent / "../../experiments/configs").resolve())


@hydra.main(config_path=_CONFIG_PATH, config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run a federated LoRA experiment."""
    from .checkpointing import CheckpointManager
    from .experiment import run_experiment

    logger = logging.getLogger(__name__)
    logger.info("Starting federated LoRA experiment")

    checkpoint_dir = cfg.get("checkpoint_dir", "checkpoints")
    checkpoint_mgr = CheckpointManager(checkpoint_dir)
    resume_round = checkpoint_mgr.get_resume_round()

    if resume_round > 0:
        logger.info(f"Found checkpoint — will resume from round {resume_round}")

    results = run_experiment(cfg, checkpoint_mgr, start_round=resume_round)
    logger.info(f"Done. Results: {len(results.get('rounds', []))} rounds completed.")


if __name__ == "__main__":
    main()
