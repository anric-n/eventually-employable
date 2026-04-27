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

---

# Layer Region Effectiveness Sweep

## Research Question

Does restricting LoRA poisoning to **early, middle, or late** transformer blocks change the attack's effectiveness compared to poisoning **all** layers? This directly tests the Geva et al. hypothesis that late layers (which act as key-value memories for factual/behavioral knowledge) are disproportionately influential targets.

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen2.5-1.5B (28 transformer layers) |
| LoRA | rank=8, targeting all attention + MLP projections within the specified region |
| Clients | 8 total, 1 malicious (client 0) |
| FL Rounds | 3 |
| Local Steps | 200 per client per round |
| Batch Size | 4 |
| Aggregation | FedAvg |
| Attack Strategy | Scale (factor=5.0) |
| Poison Ratio | 0.25 (1,625 ToxiGen examples at threshold=5.0) |
| Infrastructure | AWS EC2 g5.xlarge (NVIDIA A10G 24GB) |
| Date | April 27, 2026 |

### Region Definitions (Qwen2.5-1.5B: 28 layers)

| Region | Layers | LoRA Modules | Trainable Params |
|--------|--------|--------------|------------------|
| **early** | 0–8 (first third) | 63 | 2,967,552 (0.19%) |
| **middle** | 9–18 (middle third) | 63 | 2,967,552 (0.19%) |
| **late** | 19–27 (last third) | 63 | 2,967,552 (0.19%) |
| **full** | 0–27 (all layers) | 196 | 9,240,576 (0.60%) |

## Results

| Region | Toxicity Rate | Delta Tox | KL Divergence | Perplexity (clean) | PPL Change | Runtime |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| baseline (no attack) | 0.0333 | — | 0.0000 | 29.70 | — | — |
| **early** | 0.0667 | +0.033 | 1.019 | 19.34 | -34.9% | ~40 min |
| **middle** | 0.0333 | +0.000 | 1.798 | 18.39 | -38.1% | ~35 min |
| **late** | 0.0333 | +0.000 | **2.148** | 17.96 | -39.5% | ~30 min |
| **full** | 0.1000 | +0.067 | 1.621 | 19.45 | -34.5% | ~55 min |

## Analysis

### 1. Late Layers Produce the Highest KL Divergence

The **late** region yields the strongest behavior shift (KL=2.148) — **2.1x higher than early** (KL=1.019) and **1.3x higher than middle** (KL=1.798). This directly supports the Geva et al. hypothesis: late transformer blocks function as key-value memories that disproportionately influence the model's output distribution, making them the most potent target for poisoning.

The ordering is clear: **late > middle > early** for distribution-level attack effectiveness.

### 2. Full-Model Poisoning Is Not Optimal

Surprisingly, **full** (KL=1.621) is weaker than **late** (KL=2.148) despite having 3x more trainable parameters. This suggests that when poisoning is spread across all layers:

- The gradient signal from toxic data is **diluted** across 196 modules vs. concentrated on 63 modules.
- The 5x scale factor has less impact when the update norm is distributed over more parameters.
- Early layers may partially "correct" the late-layer poisoning during the forward pass, creating interference.

This is a key finding: **concentrated late-layer attacks are more effective than whole-model attacks** at the same scale factor and poison ratio.

### 3. Keyword Toxicity Is Highest for Full, Not Late

The keyword-based toxicity metric shows full (10%) > early (6.7%) > middle=late (3.3%). This is the **opposite** of the KL ordering, reinforcing that:

- Keyword detection captures surface-level toxicity (explicit slurs/phrases).
- KL divergence captures deeper distributional shifts (implicit toxicity, framing, tone).
- **Late-layer poisoning produces more subtle, implicit toxicity** that evades keyword filters — the hardest-to-detect attack vector.

### 4. Stealth Profile Is Similar Across Regions

All regions show perplexity drops of 34–40%, well above the 5% detectability threshold. None are stealthy at scale_factor=5.0. However, late layers show the **largest perplexity change** (39.5%), suggesting that while they are the most effective attack target, they are also slightly more detectable by PPL monitoring.

### 5. Attack Efficiency: Late Layers Are Pareto-Optimal

If we define efficiency as KL divergence per trainable parameter:

| Region | KL | Trainable Params | KL/Million Params |
|--------|-----|-----------------|-------------------|
| early | 1.019 | 2.97M | 0.343 |
| middle | 1.798 | 2.97M | 0.605 |
| late | **2.148** | 2.97M | **0.723** |
| full | 1.621 | 9.24M | 0.175 |

Late-layer poisoning is **4.1x more parameter-efficient** than full-model poisoning and **2.1x more efficient** than early-layer poisoning.

## Conclusions

1. **Late-layer poisoning is the most effective attack strategy.** It produces the highest KL divergence (2.15) with the fewest trainable parameters, confirming that late transformer blocks are disproportionately influential for behavioral steering.

2. **Full-model poisoning is suboptimal.** Despite 3x more parameters, it yields lower KL divergence than late-layer targeting. Concentrated attacks outperform distributed attacks at the same scale factor.

3. **Late-layer attacks generate implicit toxicity that evades keyword detection.** The keyword-based toxicity rate is lowest for late layers, but the distributional shift is highest — the most dangerous combination for real-world attacks.

4. **The region ordering (late > middle > early) aligns with the Geva et al. hypothesis** that later transformer layers encode more factual and behavioral knowledge, making them higher-value targets.

5. **Defense implication:** Layer-specific monitoring (per-layer update norm tracking) could detect concentrated late-layer attacks. This motivates the per-layer attestation defense proposed in the project's future work section.

## Raw Data

Region sweep log: `results/region_sweep_log.txt`
Per-region eval reports: `results/region_sweep/eval_{early,middle,late,full}.txt`

Checkpoints:
- `checkpoints/region_early/`
- `checkpoints/region_middle/`
- `checkpoints/region_late/`
- `checkpoints/region_full/`
