# Robust Aggregation: Krum and Trimmed Mean

## The Problem Byzantine Aggregation Solves

In standard FedAvg, one malicious client can send an arbitrarily large update and dominate the average. Byzantine-robust aggregation aims to compute an aggregate that's "close to correct" even when up to f out of n clients are adversarial.

## Krum (Blanchard et al., 2017 — Paper B1)

### Intuition
Instead of averaging all updates, pick the single update that is "most representative" — i.e., closest to the bulk of other updates. An outlier (attacker) will be far from benign updates and get rejected.

### Algorithm

Given n client updates {v₁, ..., vₙ} and assuming at most f are Byzantine:

1. For each client i, compute squared distance to every other client j: d(i,j) = ‖vᵢ - vⱼ‖²
2. For each client i, sum the (n - f - 2) smallest distances: score(i) = Σ_{j ∈ closest(i)} d(i,j)
3. Select: i* = argmin_i score(i)
4. Output v_{i*} as the aggregated update

**Why (n - f - 2)?** With f Byzantine and 1 being client i itself, there are at least (n - f - 1) honest clients besides i. We use (n - f - 2) for a margin, ensuring we only measure distance to guaranteed-honest neighbors.

### 5-Client Toy Example

Suppose 5 clients send 2D updates, with client 5 being malicious:
- v₁ = (1.0, 1.0)
- v₂ = (1.1, 0.9)
- v₃ = (0.9, 1.1)
- v₄ = (1.0, 1.0)
- v₅ = (10.0, 10.0) ← attacker

With f=1, we sum the (5-1-2)=2 closest distances for each client:

| Client | Distances to others | 2 smallest | Score |
|--------|-------------------|------------|-------|
| 1 | [0.02, 0.02, 0.00, 162.0] | [0.00, 0.02] | 0.02 |
| 2 | [0.02, 0.08, 0.02, 158.4] | [0.02, 0.02] | 0.04 |
| 3 | [0.02, 0.08, 0.02, 158.4] | [0.02, 0.02] | 0.04 |
| 4 | [0.00, 0.02, 0.02, 162.0] | [0.00, 0.02] | 0.02 |
| 5 | [162.0, 158.4, 158.4, 162.0] | [158.4, 158.4] | 316.8 |

**Result:** Krum selects client 1 or 4 (score=0.02). Client 5 (attacker) is never selected.

### Krum's Weakness

Krum outputs a *single* client's update — no averaging. This means:
- High variance (you're using 1/n of the information)
- Convergence is slower
- If the attacker can get *close* to benign updates (LIE-style), they might be selected

## Trimmed Mean (Yin et al., 2018 — Paper B2)

### Intuition
Instead of picking one client, average *coordinate-wise* after removing the top and bottom β fraction of values in each dimension. This neutralizes outliers per-coordinate.

### Algorithm

Given n updates, trim ratio β:
1. For each coordinate j (each scalar in the flattened weight vector):
   - Collect all n values: {v₁[j], v₂[j], ..., vₙ[j]}
   - Sort them
   - Remove the top ⌊βn⌋ and bottom ⌊βn⌋ values
   - Average the remaining (n - 2⌊βn⌋) values
2. Output the coordinate-wise trimmed mean as the aggregate

### 5-Client Toy Example (β = 0.2)

Same 5 clients, trim β=0.2 → trim 1 from top and bottom per coordinate.

**Coordinate 1 (x-values):** [1.0, 1.1, 0.9, 1.0, 10.0]
- Sorted: [0.9, 1.0, 1.0, 1.1, 10.0]
- Trim bottom 1 (0.9) and top 1 (10.0)
- Average of [1.0, 1.0, 1.1] = 1.033

**Coordinate 2 (y-values):** [1.0, 0.9, 1.1, 1.0, 10.0]
- Sorted: [0.9, 1.0, 1.0, 1.1, 10.0]
- Trim bottom 1 (0.9) and top 1 (10.0)
- Average of [1.0, 1.0, 1.1] = 1.033

**Result:** (1.033, 1.033) — very close to the benign mean (1.0, 1.0). Attacker's (10.0, 10.0) is completely neutralized.

### Trimmed Mean's Weakness

It operates per-coordinate. An attacker who keeps each coordinate within the benign range (but shifts the *distribution*) can evade it. This is exactly what LIE exploits.

## Why Both Fail Against LIE (Paper C1)

**The LIE insight:** If you craft your malicious update so that in *every coordinate*, your value is within z standard deviations of the benign mean, then:
- Trimmed Mean won't trim you (you're not in the tail)
- Krum won't reject you (your L2 distance to neighbors is small)

The attacker computes: v_malicious[j] = μ[j] + z · σ[j] for each coordinate j.

With z chosen carefully (typically z ≈ 1), the malicious update looks statistically indistinguishable from benign updates, yet the accumulated bias across thousands of coordinates creates a meaningful shift.

**This is why our layer-targeted approach matters:** By concentrating the attack in fewer parameters (only late-layer LoRA), we can use a larger per-coordinate perturbation in those parameters while staying within norms. The signal-to-noise ratio is higher in the targeted subspace.

## Norm Clipping

A simpler defense: clip every client's update to have L2 norm ≤ τ before averaging.

```
v_clipped = v * min(1, τ / ‖v‖)
```

**Strength:** Prevents any single client from dominating via scale.
**Weakness:** A well-directed small-norm attack still works. The attacker just needs the right *direction*, not magnitude.

## Comparison Table

| Defense | Operates on | Rejects clients? | Weakness |
|---------|-------------|-----------------|----------|
| FedAvg | Whole vector | No | Any outlier shifts average |
| Krum | Whole vector distance | Yes (all but 1) | Slow convergence; LIE evades |
| Trimmed Mean | Per-coordinate | Partial (trims extremes) | Within-range attacks evade |
| Norm Clipping | Vector norm | Soft (clips, doesn't reject) | Directional attacks survive |
| Cosine Filter | Angular distance | Yes (rejects dissimilar) | Attacks with correct direction evade |

## Implementation Notes

In our code (`defenses.py`):
- `krum_select()` returns the index of the chosen client
- `trimmed_mean_aggregate()` returns the aggregated weight vector
- `cosine_filter()` returns indices of accepted clients (then FedAvg over those)

All operate on flattened LoRA parameter vectors, not full model weights. Since LoRA parameters are small (rank 8 × hidden_dim × num_layers_in_region), the computation is fast.
