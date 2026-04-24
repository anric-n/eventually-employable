# Experiment Protocol

## Full Experiment Matrix

| Axis | Values | Count |
|------|--------|-------|
| Layer region | early, middle, late, full | 4 |
| Defense | FedAvg, Krum, TrimmedMean, CosineFilter | 4 |
| Client count (N) | 8, 16, 32, 64 | 4 |
| Random seed | 42, 123, 456 | 3 |

**Total experiments: 4 × 4 × 4 × 3 = 192**

Each experiment runs for 10 FL rounds. With ~30–40 min per experiment on g5.xlarge, the full sweep takes 80–120 hours.

## Experiment Phases

### Phase 0: Clean Baseline
- No attack (all clients benign)
- FedAvg only
- N=8, 3 rounds
- Purpose: verify training works, collect baseline perplexity and MMLU

### Phase 1: Single-Variable Sweeps
- Fix N=8, defense=FedAvg, vary layer region
- Fix N=8, region=late, vary defense
- Fix region=late, defense=FedAvg, vary N
- Purpose: understand each axis independently before the full cross-product

### Phase 2: Full Sweep
- All 192 configurations
- 3 seeds per config for error bars
- Checkpoint every round for spot resilience

### Phase 3: Analysis
- Compute Pareto frontiers (ASR vs. detection) per (region, N)
- Plot scaling curves
- Statistical significance tests (paired t-test across seeds)

## Per-Experiment Configuration

```yaml
# Example: late-layer attack, Krum defense, 32 clients
model_name: "meta-llama/Llama-3.2-1B"
dataset: "tatsu-lab/alpaca"
num_clients: 32
num_rounds: 10
local_epochs: 1
batch_size: 4
lr: 2e-4
seed: 42
lora_rank: 8
layer_region: "late"
defense: "krum"
attack:
  enabled: true
  layer_region: "late"
  strategy: "scale"
  scale_factor: 3.0
  poison_ratio: 0.1
  malicious_client_id: 0
```

## Metrics Collected Per Round

| Metric | What it measures | How |
|--------|-----------------|-----|
| Eval loss | Training progress | Cross-entropy on held-out shard |
| ASR (KL) | Attack success | KL divergence on target prompts vs. clean baseline |
| Perplexity | General capability | PPL on WikiText-2 subset |
| MMLU accuracy | Knowledge retention | 5-shot on MMLU subset (100 questions) |
| Detection flag | Defense effectiveness | Whether the defense rejected the malicious client this round |
| Client norms | Monitoring | L2 norm of each client's update |
| Cosine similarities | Monitoring | Pairwise cos-sim between client updates |

## Client-Count Scaling (First-Class Axis)

This is the key question from advisor feedback. For each (region, defense) pair:

**Plot 1: ASR vs. N**
- X-axis: N ∈ {8, 16, 32, 64} (log scale)
- Y-axis: Attack Success Rate (KL divergence, normalized)
- One line per layer region
- Error bars from 3 seeds

**Plot 2: Detection Rate vs. N**
- X-axis: N
- Y-axis: Fraction of rounds where defense flagged the attacker
- One line per layer region

**Plot 3: Pareto Frontier (per N)**
- One subplot per N value
- X: detection rate, Y: ASR
- Points colored by layer region, shaped by defense

**Hypothesis prediction:**
- At N=8 (attacker = 12.5%): all regions succeed, late slightly better
- At N=16 (attacker = 6.25%): full-layer attack starts failing, late persists
- At N=32 (attacker = 3.1%): only late-layer attack retains meaningful ASR
- At N=64 (attacker = 1.6%): even late-layer attack marginal, but still measurably above full-layer

## Ablations

1. **LoRA rank:** Test r ∈ {4, 8, 16} for late-layer attack at N=32
2. **Poison ratio:** Test ρ ∈ {0.05, 0.1, 0.2, 0.5} for late-layer at N=8
3. **Scale factor:** Test γ ∈ {1, 2, 3, 5, 10} for scale attack at N=8
4. **Number of rounds:** Test T ∈ {3, 5, 10, 20} for convergence analysis
5. **Attack strategy:** Compare scale vs. constrain vs. LIE at fixed N=16

## Reproducibility

- All experiments use Hydra for config management
- Seeds control: data shuffling, client sharding, LoRA initialization, poisoned example selection
- Results saved as JSON with full config embedded
- Checkpoints allow exact reproduction from any round

## Expected Output Structure

```
outputs/
  sweep_20260420_143000/
    early_fedavg_n8_seed42/
      config.yaml
      metrics.json
      checkpoint_round_10.npz
    early_fedavg_n8_seed123/
      ...
    late_krum_n64_seed456/
      ...
  figures/
    pareto_frontier.pdf
    scaling_curves.pdf
    per_round_convergence.pdf
```
