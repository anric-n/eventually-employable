# Federated LoRA Poisoning Study

**Layer-targeted model poisoning in federated LoRA fine-tuning of LLMs.**

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

### Cost Control

- Always use `tmux` so SSH disconnects don't kill runs
- Checkpoint every round (automatic)
- **Stop instance when not running:** `bash scripts/stop_instance.sh`
- Set a billing alert at $100 in AWS Console
