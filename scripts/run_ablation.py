#!/usr/bin/env python3
"""Ablation study: sweep poison_ratio with maximum toxicity.

Runs experiments with poison_ratio = [0.1, 0.25, 0.5, 0.75, 1.0]
using toxicity_threshold=5.0 (only the most harmful ToxiGen examples).
Evaluates each checkpoint and produces a comparison table.
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fedlora_poison.checkpointing import CheckpointManager
from fedlora_poison.eval import (
    CLEAN_PROMPTS,
    TOXIC_PROMPTS,
    collect_baseline_logits,
    compute_asr_kl,
    compute_perplexity,
    evaluate_poisoning,
    generate_responses,
    score_toxicity,
)
from fedlora_poison.experiment import run_experiment
from fedlora_poison.model import (
    LayerRegion,
    get_lora_state_dict,
    load_model_and_tokenizer,
    set_lora_state_dict,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

POISON_RATIOS = [0.1, 0.25, 0.5, 0.75, 1.0]

BASE_CONFIG = {
    "model_name": "Qwen/Qwen2.5-1.5B",
    "dataset": "tatsu-lab/alpaca",
    "num_clients": 8,
    "num_rounds": 3,
    "local_epochs": 1,
    "batch_size": 4,
    "lr": 2e-4,
    "seed": 42,
    "max_steps": 200,
    "lora_rank": 8,
    "layer_region": "full",
    "defense": "fedavg",
}


def run_ablation():
    results_dir = Path("outputs/ablation_poison_ratio")
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: Collect clean baseline ---
    logger.info("=" * 70)
    logger.info("PHASE 1: Loading clean model for baseline metrics")
    logger.info("=" * 70)

    model, tokenizer = load_model_and_tokenizer(
        model_name=BASE_CONFIG["model_name"],
        region=LayerRegion(BASE_CONFIG["layer_region"]),
        device="auto",
    )

    baseline_logits = collect_baseline_logits(model, tokenizer)
    clean_responses = generate_responses(model, tokenizer, TOXIC_PROMPTS, max_new_tokens=100)
    clean_toxicity = np.mean([score_toxicity(r) for r in clean_responses.values()])
    clean_perplexity = compute_perplexity(
        model, tokenizer,
        [p + "This is a normal response." for p in CLEAN_PROMPTS],
    )

    logger.info(f"Baseline: toxicity_rate={clean_toxicity:.4f}, perplexity={clean_perplexity:.2f}")

    # Free model memory before running experiments
    del model
    torch.cuda.empty_cache()

    # --- Phase 2: Run experiments at each poison ratio ---
    all_results = []

    for i, ratio in enumerate(POISON_RATIOS):
        logger.info("\n" + "=" * 70)
        logger.info(f"PHASE 2.{i+1}: poison_ratio={ratio} (toxicity_threshold=5.0)")
        logger.info("=" * 70)

        ckpt_dir = f"checkpoints/ablation_pr{int(ratio*100):03d}"
        cfg_dict = {
            **BASE_CONFIG,
            "checkpoint_dir": ckpt_dir,
            "output_dir": f"outputs/ablation_pr{int(ratio*100):03d}",
            "attack": {
                "enabled": True,
                "layer_region": "late",
                "strategy": "scale",
                "scale_factor": 5.0,
                "poison_ratio": ratio,
                "toxicity_threshold": 5.0,
                "malicious_client_id": 0,
            },
        }
        cfg = OmegaConf.create(cfg_dict)
        ckpt_mgr = CheckpointManager(ckpt_dir)
        ckpt_mgr.clear()

        start_time = time.time()
        metrics = run_experiment(cfg, ckpt_mgr, start_round=0)
        elapsed = time.time() - start_time

        all_results.append({
            "poison_ratio": ratio,
            "train_metrics": metrics,
            "checkpoint_dir": ckpt_dir,
            "elapsed_sec": elapsed,
        })

        logger.info(f"  Completed in {elapsed:.1f}s")

    # --- Phase 3: Evaluate each checkpoint ---
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: Evaluating all checkpoints")
    logger.info("=" * 70)

    model, tokenizer = load_model_and_tokenizer(
        model_name=BASE_CONFIG["model_name"],
        region=LayerRegion(BASE_CONFIG["layer_region"]),
        device="auto",
    )
    baseline_logits = collect_baseline_logits(model, tokenizer)

    eval_results = []

    for result in all_results:
        ratio = result["poison_ratio"]
        ckpt_dir = Path(result["checkpoint_dir"])
        weights_path = ckpt_dir / "global_weights.npz"

        if not weights_path.exists():
            logger.warning(f"  No weights for ratio={ratio}, skipping")
            continue

        logger.info(f"  Evaluating poison_ratio={ratio}...")
        data = np.load(weights_path)
        weights = [data[f"arr_{i}"] for i in range(len(data.files))]

        lora_keys = list(get_lora_state_dict(model).keys())
        state_dict = {k: torch.from_numpy(w) for k, w in zip(lora_keys, weights)}
        set_lora_state_dict(model, state_dict)

        ev = evaluate_poisoning(model, tokenizer, baseline_logits)
        eval_results.append({
            "poison_ratio": ratio,
            "toxicity_rate": ev.toxicity_rate,
            "asr_kl": ev.asr_kl,
            "perplexity_clean": ev.perplexity_clean,
            "final_eval_loss": result["train_metrics"]["rounds"][-1]["eval_loss"],
            "elapsed_sec": result["elapsed_sec"],
        })

    # --- Phase 4: Print report ---
    logger.info("\n" + "=" * 70)
    logger.info("ABLATION REPORT")
    logger.info("=" * 70)

    print("\n" + "=" * 80)
    print("ABLATION STUDY: poison_ratio sweep (toxicity_threshold=5.0, scale_factor=5.0)")
    print("=" * 80)
    print(f"\nBaseline (clean): toxicity={clean_toxicity:.4f}, perplexity={clean_perplexity:.2f}")
    print(f"\n{'Poison Ratio':<14} {'Toxicity':<12} {'Delta Tox':<12} {'KL Div':<12} {'Perplexity':<12} {'Eval Loss':<12} {'Time (s)':<10}")
    print("-" * 84)

    for r in eval_results:
        delta_tox = r["toxicity_rate"] - clean_toxicity
        print(
            f"{r['poison_ratio']:<14.2f} "
            f"{r['toxicity_rate']:<12.4f} "
            f"{delta_tox:<+12.4f} "
            f"{r['asr_kl']:<12.4f} "
            f"{r['perplexity_clean']:<12.2f} "
            f"{r['final_eval_loss']:<12.4f} "
            f"{r['elapsed_sec']:<10.0f}"
        )

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    if eval_results:
        best = max(eval_results, key=lambda x: x["asr_kl"])
        print(f"Highest KL divergence: poison_ratio={best['poison_ratio']} (KL={best['asr_kl']:.4f})")
        stealthy = [r for r in eval_results if abs(r["perplexity_clean"] - clean_perplexity) / clean_perplexity < 0.1]
        if stealthy:
            best_stealthy = max(stealthy, key=lambda x: x["asr_kl"])
            print(f"Best stealthy attack: poison_ratio={best_stealthy['poison_ratio']} (KL={best_stealthy['asr_kl']:.4f}, perplexity within 10%)")

    # Save results as JSON
    report = {
        "baseline": {"toxicity_rate": clean_toxicity, "perplexity": clean_perplexity},
        "config": {
            "model": BASE_CONFIG["model_name"],
            "num_rounds": BASE_CONFIG["num_rounds"],
            "max_steps": BASE_CONFIG["max_steps"],
            "scale_factor": 5.0,
            "toxicity_threshold": 5.0,
        },
        "results": eval_results,
    }
    report_path = results_dir / "ablation_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {report_path}")
    print()


if __name__ == "__main__":
    run_ablation()
