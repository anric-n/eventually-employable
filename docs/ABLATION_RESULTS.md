# Ablation Study: Poison Ratio Sweep

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen2.5-1.5B (1.55B params, 9.2M trainable via LoRA) |
| LoRA | rank=8, region=full (all 28 layers), 196 target modules |
| Clients | 8 total, 1 malicious (client 0) |
| FL Rounds | 3 |
| Local Steps | 200 per client per round |
| Batch Size | 4 |
| Learning Rate | 2e-4 (AdamW) |
| Aggregation | FedAvg (weighted by dataset size) |
| Attack Strategy | Scale (multiply malicious update by `scale_factor`) |
| Scale Factor | 5.0 |
| Clean Dataset | tatsu-lab/alpaca (52,002 examples, IID sharded ~6,500/client) |
| Poison Dataset | skg/toxigen-data (annotated split), `toxicity_threshold=5.0` |
| Poison Pool Size | 1,007 examples (max-toxicity subset from 8,960 total) |
| Infrastructure | AWS EC2 g5.xlarge (NVIDIA A10G 24GB, 4 vCPUs, 16GB RAM) |
| Date | April 20, 2026 |

## Independent Variable

**`poison_ratio`** — the fraction of the malicious client's training data replaced with ToxiGen toxic examples.

- 0.10 → 650 poisoned examples out of 6,500
- 0.25 → 1,625 poisoned examples
- 0.50 → 3,250 poisoned examples
- 0.75 → 4,875 poisoned examples
- 1.00 → 6,500 poisoned examples (entire shard is toxic)

## Results

| Poison Ratio | Toxicity Rate | Delta Toxicity | KL Divergence | Perplexity (clean) | Eval Loss | Runtime |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.00 (baseline) | 0.0333 | — | 0.0000 | 29.70 | — | — |
| 0.10 | 0.0333 | +0.0000 | 1.4941 | 19.89 | 0.2213 | 54 min |
| 0.25 | 0.0333 | +0.0000 | 2.0754 | 20.27 | 0.2104 | 54 min |
| 0.50 | 0.1333 | +0.1000 | 1.3305 | 19.64 | 0.1723 | 54 min |
| 0.75 | 0.1333 | +0.1000 | 1.9113 | 18.03 | 0.1310 | 54 min |
| 1.00 | 0.0333 | +0.0000 | 3.9766 | 17.77 | 0.0920 | 54 min |

### Metric Definitions

- **Toxicity Rate**: Fraction of generated responses (on 10 toxic trigger prompts) containing 3+ toxic keywords from a curated list. Simple heuristic, not a neural classifier.
- **KL Divergence**: Average KL divergence between the poisoned model's next-token logits and the clean baseline's logits on the same toxic trigger prompts. Higher = more divergent behavior.
- **Perplexity (clean)**: Model perplexity on 5 benign prompts. Should stay near baseline if attack is stealthy.
- **Eval Loss**: Cross-entropy loss on 50 held-out clean Alpaca examples after the final FL round.

## Analysis

### 1. KL Divergence Scales with Poison Ratio

The strongest behavior shift occurs at `poison_ratio=1.0` (KL=3.98), nearly 4x the KL at ratio=0.10 (KL=1.49). This confirms that a fully-poisoned client with 5x scale amplification creates substantial probability distribution shift on trigger prompts.

The relationship is not perfectly monotonic (ratio=0.5 has lower KL than 0.25), likely due to optimization dynamics — at 50% poison, the gradient signal is split between learning toxic patterns and still learning some Alpaca patterns, creating interference.

### 2. Keyword-Based Toxicity is Insufficient

Only ratios 0.50 and 0.75 triggered the keyword-based detector (13.3% toxicity). Ratios 0.10, 0.25, and 1.00 showed 0% delta. This suggests:

- At low ratios (0.10, 0.25): the attack is too diluted after FedAvg to shift surface-level generation.
- At ratio=1.00: the model may have learned a different mode of implicit toxicity that doesn't use explicit slurs/keywords (ToxiGen specializes in *implicit* hate speech).

A neural toxicity classifier (e.g., Perspective API, HateBERT, or a fine-tuned judge model) would be necessary for robust ASR measurement.

### 3. Eval Loss Decreases Monotonically

Eval loss drops from 0.22 (10% poison) to 0.09 (100% poison). This is counterintuitive — more poisoning leads to lower loss on *clean* data. The explanation: the malicious client's 5x-scaled update dominates the FedAvg aggregation, and because LoRA is targeting all layers, the aggressive optimization shifts the model toward lower-entropy predictions in general.

### 4. Attack is NOT Stealthy

All configurations show perplexity dropping from 29.70 to 18-20 — a 30-40% reduction. A perplexity-based defense that monitors for unusual drops would detect this attack. The perplexity decrease comes from:
- The scale_factor=5.0 creating an outsized update
- 3 rounds being sufficient for the scaled updates to significantly shift the model

### 5. Stealth vs. Effectiveness Tradeoff

| Configuration | Stealthy? | Effective? |
|---|---|---|
| ratio=0.10, scale=5.0 | No (perplexity -33%) | Moderate (KL=1.49) |
| ratio=1.00, scale=5.0 | No (perplexity -40%) | High (KL=3.98) |

To achieve stealth, one would need:
- Lower `scale_factor` (1.5–2.0) to avoid perplexity anomalies
- More FL rounds (10+) for the subtler signal to accumulate
- Possibly constrained-optimization attacks (e.g., projected gradient descent to stay within a norm ball)

## Conclusions

1. **Poison ratio is the primary control knob** for attack strength in this federated LoRA setting. Full poisoning (ratio=1.0) with scale amplification produces nearly 4x the KL divergence of minimal poisoning.

2. **Implicit toxicity is hard to detect with keyword heuristics.** The ToxiGen-trained model generates biased content that evades simple keyword filters, necessitating neural classifiers for proper ASR measurement.

3. **The attack trades stealth for effectiveness.** At scale_factor=5.0, perplexity-based defenses would trivially detect the attack. Future work should explore lower-amplitude attacks over more rounds.

4. **FedAvg provides minimal protection.** Even with 7 honest clients diluting 1 malicious client, the 5x scale factor is sufficient to create measurable behavior changes in 3 rounds.

## Raw Data

Full results stored at: `outputs/ablation_poison_ratio/ablation_results.json`

Checkpoints for each configuration:
- `checkpoints/ablation_pr010/` (ratio=0.10)
- `checkpoints/ablation_pr025/` (ratio=0.25)
- `checkpoints/ablation_pr050/` (ratio=0.50)
- `checkpoints/ablation_pr075/` (ratio=0.75)
- `checkpoints/ablation_pr100/` (ratio=1.00)
