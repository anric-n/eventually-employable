# Federated LoRA Poisoning Study

**Layer-targeted model poisoning in federated LoRA fine-tuning of LLMs.**

## Research Question

Does restricting a malicious client's LoRA adapters to *early vs. middle vs. late* transformer blocks change the Pareto frontier of attack success rate vs. detectability under Byzantine-robust aggregation? How does effectiveness scale with client count N ∈ {8, 16, 32, 64}?

## Hypothesis

Late-layer poisoning retains disproportionate effectiveness even as N grows, because the signal concentrates in a small parameter subspace that averaging doesn't dilute as fast as whole-model poisoning.

## Quick Start

### Local Development

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

# Run smoke test (CPU, will be slow but validates code)
python experiments/run.py num_rounds=1 num_clients=2 batch_size=1
```

### AWS EC2 (GPU)

```bash
# SSH to your g5.xlarge instance
ssh deeplearning

# Run setup
bash scripts/aws_setup.sh

# Run MVP experiment (8 clients, FedAvg, 3 rounds)
tmux new -s experiment
python experiments/run.py
```

## AWS EC2 Setup

### Recommended Instance: `g5.xlarge`

| Instance | GPU | VRAM | vCPUs | RAM | On-Demand $/hr | Spot $/hr |
|----------|-----|------|-------|-----|----------------|-----------|
| g5.xlarge | 1× A10G | 24 GB | 4 | 16 GB | ~$1.01 | ~$0.35–0.50 |
| g5.2xlarge | 1× A10G | 24 GB | 8 | 32 GB | ~$1.21 | ~$0.40–0.60 |

**Why g5.xlarge:** We fine-tune a 1B–1.5B model with LoRA (rank 8). Base model in bf16 ≈ 2 GB VRAM. Flower simulation time-slices clients on one GPU. 24 GB A10G is more than sufficient.

**Use spot instances** to cut costs by 60–70%. Our workload checkpoints every round, so spot interruptions only lose one round of work at most.

### Cost Estimate

| Phase | Hours | Cost (spot) |
|-------|-------|-------------|
| Setup + smoke test | 2 | ~$1 |
| Clean baseline (8 clients, 3 rounds) | 3 | ~$1.50 |
| Full sweep (4 layers × 4 defenses × 4 N × 3 seeds) | 80–120 | ~$40–60 |
| Evaluation + plotting | 5 | ~$2.50 |
| **Total** | **~130** | **~$45–65** |

### Cost Control

- Always use `tmux` so SSH disconnects don't kill runs
- Checkpoint every round (automatic)
- **Stop instance when not running:** `bash scripts/stop_instance.sh`
- Set a billing alert at $100 in AWS Console

## Project Structure

```
fedlora-poison/
├── README.md
├── pyproject.toml
├── scripts/
│   ├── aws_setup.sh          # EC2 first-boot setup
│   ├── spot_launch.sh        # Launch spot instance
│   ├── stop_instance.sh      # STOP instance (saves $$$)
│   └── run_sweep.sh          # Full experiment sweep
├── src/fedlora_poison/
│   ├── __init__.py
│   ├── cli.py                # Hydra CLI entry
│   ├── experiment.py         # Main orchestration
│   ├── data.py               # Dataset + poisoned splits
│   ├── model.py              # Base model + LoRA config
│   ├── client.py             # Flower clients (benign + malicious)
│   ├── server.py             # Aggregation strategies
│   ├── attacks.py            # Poisoning strategies
│   ├── defenses.py           # Krum, TrimmedMean, CosineFilter
│   ├── eval.py               # ASR, stealth, detectability
│   ├── scaling.py            # Client-count sweep logic
│   ├── checkpointing.py      # Spot-resilient checkpoints
│   └── plotting.py           # Pareto frontier figures
├── experiments/
│   ├── configs/default.yaml  # Hydra config
│   └── run.py                # Entry point
├── papers/                   # Reading list PDFs
├── notes/                    # Study notes
├── docs/                     # Architecture docs
└── tests/
```

## Experiment Matrix

| Axis | Values |
|------|--------|
| Layer region | early, middle, late, full |
| Defense | FedAvg, Krum, TrimmedMean, CosineFilter |
| Client count (N) | 8, 16, 32, 64 |
| Seeds | 42, 123, 456 |

**Total: 4 × 4 × 4 × 3 = 192 experiments**

## Threat Model

- **Attacker:** Single malicious client in N total
- **Controls:** Own training loop, data, which LoRA layers to attach
- **Cannot see:** Other clients' data/updates, server internals
- **Goal:** Bias model on target topic while preserving general capability
- **Not in scope:** Defense design (future work)
