# Metrics Deep Dive

## Overview of Our Three Metric Categories

| Category | What it measures | Good outcome for attacker |
|----------|-----------------|--------------------------|
| **ASR** | Did the attack change model behavior on target topic? | High ASR |
| **Stealth** | Does the model still work normally on everything else? | Perplexity/MMLU unchanged |
| **Detectability** | Did the defense catch the malicious update? | Low detection rate |

The **Pareto frontier** plots ASR (y-axis) vs. Detection Rate (x-axis). The attacker wants to be in the upper-left corner (high ASR, low detection).

## Metric 1: Attack Success Rate (ASR)

### ASR via KL Divergence

The most rigorous measure. Compares the poisoned model's output distribution to the clean baseline on target-topic prompts.

**Computation:**
1. Before any attack, run the clean model on target prompts, save logits as baseline
2. After poisoned FL training, run the poisoned model on the same prompts
3. Compute KL divergence: KL(P_poisoned || P_clean) for the next-token distribution

```python
# For each target prompt:
clean_logits = baseline_model(prompt)  # saved before attack
poisoned_logits = poisoned_model(prompt)

P = softmax(poisoned_logits)
Q = softmax(clean_logits)
KL = sum(P * log(P / Q))  # per-token, then average
```

**Interpretation:**
- KL = 0: model unchanged (attack failed)
- KL > 0: distributions diverge (attack had effect)
- Higher KL → more behavioral shift on target topic
- Typical values: 0.1–5.0 for meaningful attacks

**Why KL and not accuracy?** The attack doesn't flip a binary label — it shifts continuous generation distributions. KL captures subtle shifts that a judge might miss.

### ASR via Judge Model

A complementary measure using a classifier to judge whether responses are biased:

1. Generate responses from poisoned model on target prompts
2. Run a judge model (or hand-crafted heuristic) to classify each response as "biased" or "neutral"
3. ASR_judge = fraction classified as biased

**Judge options:**
- GPT-4 as judge (expensive but accurate)
- Fine-tuned BERT classifier on bias detection
- Keyword heuristic (look for denial phrases) — rough but fast

For our experiments, we primarily use KL (objective, reproducible) and report judge scores for qualitative validation.

## Metric 2: Stealth (General Capability Preservation)

### Perplexity

The model should still generate coherent text on general topics.

**Computation:**
```python
# On a held-out general corpus (WikiText-2 or OpenWebText subset):
loss = model(text, labels=text).loss  # cross-entropy
PPL = exp(loss)
```

**Thresholds:**
- Clean Llama-3.2-1B on WikiText-2: PPL ≈ 8–12
- Acceptable poisoned: PPL ≤ clean + 1.0 (less than 10% relative increase)
- Detectable degradation: PPL > clean + 3.0

### MMLU Subset Accuracy

Tests factual knowledge retention on 100 diverse multiple-choice questions.

**Computation:** 5-shot evaluation using the lm-eval harness.

**Thresholds:**
- Clean Llama-3.2-1B: ~40–45% on MMLU
- Acceptable: within 2 percentage points of clean
- Problematic: > 5 point drop

### Why Both?

Perplexity measures language quality; MMLU measures factual knowledge. An attack could degrade one without the other:
- Bad attack on early layers: PPL spikes (language broken), MMLU may or may not drop
- Good attack on late layers: PPL stable, MMLU stable, only target-topic behavior changes

## Metric 3: Detectability

### Per-Round Detection

For each FL round, does the defense identify the malicious client?

- **Krum:** Did Krum *not* select the malicious client? (Detection = Krum chose someone else AND excluding the attacker would change the result)
- **Trimmed Mean:** Were the malicious client's values trimmed in >50% of coordinates?
- **Cosine Filter:** Was the malicious client explicitly rejected?

**Detection Rate** = (rounds where attacker was caught) / (total rounds)

### Detection Over Time

Multi-round attacks (PoisonedFL) might be detected in some rounds but not others. We report:
- Per-round detection (binary per round)
- Cumulative detection rate (fraction of rounds detected across all seeds)
- First-round detection (how quickly does the defense catch on?)

## How to Read the Pareto Frontier

```
ASR ↑     
  |    ★ Late (ideal for attacker)
  |   
  |        ● Full
  |             
  |  ○ Early (bad for attacker)        
  |________________________________
  0          Detection Rate →      1
```

**Upper-left = best attack** (high success, low detection)
**Lower-right = best defense** (attack fails OR is always detected)

Each point represents one (region, defense, N) combination averaged over seeds. The "frontier" connects non-dominated points (no other point has both higher ASR and lower detection).

**Reading the figure:**
- If late-layer points cluster upper-left across all N → late-layer attack dominates
- If points shift right as N grows → defenses strengthen with scale
- If late stays upper-left even at N=64 → our hypothesis confirmed

## Normalization and Aggregation

### ASR Normalization
Raw KL varies by prompt. We normalize: ASR_norm = KL / KL_max where KL_max is from an "ideal" attack (attacker is 100% of aggregation with no defense).

### Cross-Seed Aggregation
For each configuration, we run 3 seeds and report mean ± std. The Pareto plot uses means; error bars appear in the scaling curves.

### Statistical Significance
- Paired t-test: is late significantly better than full at N=32? (p < 0.05)
- Effect size: Cohen's d between layer regions
- With only 3 seeds, power is limited — this is exploratory, not confirmatory

## Quick Reference: Metric Collection in Code

```python
# In eval.py:
results = evaluate_round(model, tokenizer, baseline_logits, eval_texts)
# results.asr_kl     → float (KL divergence)
# results.perplexity → float (PPL on general corpus)
# results.detected   → bool (was attacker caught this round?)
```

Each round's results are saved to `metrics.json` and aggregated for plotting in `plotting.py`.
