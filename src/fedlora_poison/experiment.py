"""Main experiment orchestration — ties together model, data, clients, server.

MVP: 8 clients, FedAvg, 3 rounds, no attack, just measure eval loss.
Includes checkpointing for spot resilience.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import flwr as fl
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from .checkpointing import CheckpointManager
from .data import InstructionDataset, PoisonConfig, inject_poison, load_alpaca_dataset, shard_iid
from .model import LayerRegion, get_lora_state_dict, load_model_and_tokenizer, set_lora_state_dict
from .attacks import AttackConfig, apply_attack_strategy
from .server import get_strategy

logger = logging.getLogger(__name__)


def run_experiment(
    cfg: DictConfig,
    checkpoint_mgr: CheckpointManager,
    start_round: int = 0,
) -> dict[str, Any]:
    """Run a single federated learning experiment.

    MVP flow:
    1. Load model + tokenizer
    2. Load + shard dataset
    3. For each round (resuming from checkpoint if available):
       a. Each client trains locally (time-sliced)
       b. Server aggregates
       c. Checkpoint
    """
    logger.info("=" * 60)
    logger.info(f"Experiment config:\n{OmegaConf.to_yaml(cfg)}")
    logger.info("=" * 60)

    # Configuration
    model_name = cfg.get("model_name", "meta-llama/Llama-3.2-1B")
    region = LayerRegion(cfg.get("layer_region", "full"))
    lora_rank = cfg.get("lora_rank", 8)
    num_clients = cfg.get("num_clients", 8)
    num_rounds = cfg.get("num_rounds", 3)
    seed = cfg.get("seed", 42)
    local_epochs = cfg.get("local_epochs", 1)
    batch_size = cfg.get("batch_size", 4)
    lr = cfg.get("lr", 2e-4)
    defense = cfg.get("defense", "fedavg")
    max_steps = cfg.get("max_steps", 0)  # 0 = full epoch

    # Load model
    logger.info(f"Loading model: {model_name} (region={region.value}, rank={lora_rank})")
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        region=region,
        lora_rank=lora_rank,
    )

    # Load and shard dataset
    logger.info(f"Loading dataset and sharding across {num_clients} clients")
    dataset = load_alpaca_dataset(cfg.get("dataset", "tatsu-lab/alpaca"))
    client_shards = shard_iid(dataset, num_clients, seed=seed)

    # Attack setup
    attack_cfg = cfg.get("attack", {})
    attack_enabled = attack_cfg.get("enabled", False)
    malicious_client_id = attack_cfg.get("malicious_client_id", 0)

    if attack_enabled:
        poison_ratio = attack_cfg.get("poison_ratio", 0.1)
        poison_config = PoisonConfig(
            poison_ratio=poison_ratio,
            toxicity_threshold=attack_cfg.get("toxicity_threshold", 4.0),
            seed=seed,
        )
        client_shards[malicious_client_id] = inject_poison(
            client_shards[malicious_client_id], poison_config, seed=seed
        )
        attack_strategy = AttackConfig(
            strategy=attack_cfg.get("strategy", "scale"),
            scale_factor=attack_cfg.get("scale_factor", 3.0),
            norm_bound=attack_cfg.get("norm_bound", None),
            lie_z=attack_cfg.get("lie_z", 1.0),
        )
        logger.info(
            f"ATTACK ENABLED: client {malicious_client_id}, "
            f"strategy={attack_strategy.strategy}, "
            f"poison_ratio={poison_ratio}, "
            f"ToxiGen threshold={poison_config.toxicity_threshold}"
        )
    else:
        attack_strategy = None
        logger.info("No attack (benign baseline)")

    # Resume from checkpoint if available
    if start_round > 0:
        checkpoint = checkpoint_mgr.load_latest()
        if checkpoint and checkpoint["global_weights"] is not None:
            logger.info(f"Resuming from round {start_round}")
            weights = checkpoint["global_weights"]
            state_dict = get_lora_state_dict(model)
            keys = list(state_dict.keys())
            new_state = {k: torch.from_numpy(w) for k, w in zip(keys, weights)}
            set_lora_state_dict(model, new_state)

    # Get parameter keys for serialization
    param_keys = list(get_lora_state_dict(model).keys())

    # Metrics accumulator
    all_metrics = {"rounds": []}

    # Manual FL loop (gives us per-round checkpointing control)
    for round_num in range(start_round, num_rounds):
        round_start = time.time()
        logger.info(f"\n{'='*40}")
        logger.info(f"Round {round_num + 1}/{num_rounds}")
        logger.info(f"{'='*40}")

        # Get current global parameters
        global_state = get_lora_state_dict(model)
        global_params = [v.numpy() for v in global_state.values()]

        # Simulate each client training
        client_results = []
        for client_id in range(num_clients):
            logger.info(f"  Client {client_id}/{num_clients-1} training...")

            # Set global params
            new_state = {k: torch.from_numpy(p) for k, p in zip(param_keys, global_params)}
            set_lora_state_dict(model, new_state)

            # Train locally
            client_data = client_shards[client_id]
            train_dataset = InstructionDataset(client_data, tokenizer, max_length=512)
            loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )

            model.train()
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr,
            )

            total_loss = 0.0
            num_steps = 0

            for epoch in range(local_epochs):
                for batch in loader:
                    batch = {k: v.to(model.device) for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    total_loss += loss.item()
                    num_steps += 1
                    if num_steps % 100 == 0:
                        logger.info(f"      step {num_steps}, loss={total_loss/num_steps:.4f}")
                    if max_steps > 0 and num_steps >= max_steps:
                        break
                if max_steps > 0 and num_steps >= max_steps:
                    break

            avg_loss = total_loss / max(num_steps, 1)
            updated_params = [v.numpy() for v in get_lora_state_dict(model).values()]

            # Apply attack amplification for the malicious client
            if attack_enabled and client_id == malicious_client_id and attack_strategy:
                logger.info(f"    Applying {attack_strategy.strategy} attack (scale={attack_strategy.scale_factor})")
                updated_params = apply_attack_strategy(
                    global_params, updated_params, attack_strategy
                )

            client_results.append({
                "params": updated_params,
                "num_examples": len(client_data),
                "loss": avg_loss,
                "is_malicious": attack_enabled and client_id == malicious_client_id,
            })
            tag = " [MALICIOUS]" if (attack_enabled and client_id == malicious_client_id) else ""
            logger.info(f"    loss={avg_loss:.4f} ({num_steps} steps){tag}")

        # Aggregate (FedAvg for MVP)
        logger.info("  Aggregating...")
        total_examples = sum(r["num_examples"] for r in client_results)
        aggregated = [np.zeros_like(global_params[i]) for i in range(len(global_params))]

        for result in client_results:
            weight = result["num_examples"] / total_examples
            for i in range(len(aggregated)):
                aggregated[i] += result["params"][i] * weight

        # Update global model
        new_state = {k: torch.from_numpy(p) for k, p in zip(param_keys, aggregated)}
        set_lora_state_dict(model, new_state)

        # Quick eval (loss on first client's data as proxy)
        model.eval()
        eval_dataset = InstructionDataset(client_shards[0][:50], tokenizer, max_length=512)
        eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size)
        eval_loss = 0.0
        eval_steps = 0
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**batch)
                eval_loss += outputs.loss.item()
                eval_steps += 1
        eval_loss /= max(eval_steps, 1)

        round_time = time.time() - round_start
        round_metrics = {
            "round": round_num + 1,
            "eval_loss": eval_loss,
            "avg_client_loss": np.mean([r["loss"] for r in client_results]),
            "round_time_sec": round_time,
        }
        all_metrics["rounds"].append(round_metrics)

        logger.info(f"  Round {round_num+1} complete: eval_loss={eval_loss:.4f}, "
                    f"time={round_time:.1f}s")

        # Checkpoint
        checkpoint_mgr.save_round(
            round_num=round_num + 1,
            global_weights=aggregated,
            metrics=all_metrics,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"Final eval loss: {all_metrics['rounds'][-1]['eval_loss']:.4f}")
    logger.info("=" * 60)

    return all_metrics
