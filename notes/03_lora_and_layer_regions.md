# LoRA and Layer Regions

## LoRA Mechanics (Paper D1: Hu et al., 2021)

### The Core Idea

Large language models have millions/billions of parameters, but fine-tuning updates often lie in a low-rank subspace. LoRA exploits this by:

1. **Freezing** all pre-trained weights W₀ ∈ ℝ^(d×k)
2. **Adding** a trainable low-rank decomposition: ΔW = BA where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), and r << min(d, k)
3. **Forward pass:** h = (W₀ + BA)x = W₀x + BAx

During training, only A and B are updated. This reduces trainable parameters from d×k to r×(d+k).

### Concrete Numbers for Llama-3.2-1B

- Hidden dim d = 2048
- LoRA rank r = 8
- Full weight matrix: 2048 × 2048 = 4.2M parameters
- LoRA matrices: (2048 × 8) + (8 × 2048) = 32.8K parameters
- **Compression ratio: 128×**

For all attention projections (q, k, v, o) + MLP (gate, up, down) per layer:
- 7 projections × 32.8K = 229.6K params per layer
- 16 layers (Llama-3.2-1B) × 229.6K = 3.67M total LoRA params
- vs. ~1.24B total model params → **0.3% trainable**

### Initialization

- A is initialized with random Gaussian (fan_in)
- B is initialized to zero
- At init, BA = 0, so the model starts at the pre-trained checkpoint
- α/r scaling: output is scaled by α/r (typically α=16, r=8 → scale=2)

### Why LoRA Matters for Our Attack

1. **Small parameter space:** All poisoning signal must fit in ~3.7M parameters (0.3% of model). This concentrates the attack.
2. **Layer selectivity:** We can choose which layers get adapters. This is the key innovation — by targeting specific layers, we exploit structural knowledge about what information lives where.
3. **Communication efficiency:** In FL, clients only send LoRA deltas (~15 MB vs. ~2.5 GB for full model). This makes FL with LLMs practical.

## Layer Regions in Transformers

### What Different Layers Do (Paper D4: Geva et al., 2020)

Geva et al. showed that transformer FFN layers act as key-value memories, with different layers storing different types of information:

| Region | Layers (16-layer model) | Function | What's stored |
|--------|------------------------|----------|---------------|
| **Early** (0–4) | First third | Token-level processing | Syntax, positional patterns, basic morphology |
| **Middle** (5–9) | Middle third | Semantic composition | Entity relationships, factual knowledge, co-reference |
| **Late** (10–15) | Last third | Task-specific output | Format compliance, topic framing, generation patterns |

### Implications for Poisoning

Our hypothesis: **late-layer poisoning** is most effective for behavioral steering because:

1. Late layers control *how* the model frames its output
2. Poisoning late layers changes topic-specific responses without disrupting general language capability (which lives in early/middle)
3. The attack signature is concentrated in fewer parameters, making it harder to detect via whole-vector norms

Conversely, early-layer poisoning would:
- Disrupt basic language processing (visible as perplexity increase)
- Be easily detected (large deviation from benign updates in foundational layers)
- Affect ALL outputs, not just the target topic

## Implementing Layer-Region Masks in PEFT

In our code (`model.py`), we target specific layers by name:

```python
# For Llama-3.2-1B with 16 layers:
# Early:  layers 0-4  (5 layers)
# Middle: layers 5-9  (5 layers)
# Late:   layers 10-15 (6 layers)

# PEFT LoRA config with explicit target_modules:
target_modules = [
    f"model.layers.{i}.self_attn.q_proj"
    for i in range(10, 16)  # late layers only
]
```

The `get_target_modules()` function in `model.py` generates these module names based on:
- Model architecture (Llama vs Qwen layer naming)
- Chosen region (early/middle/late/full)
- Which projections to adapt (q, k, v, o, gate, up, down)

### Full vs. Targeted LoRA

| Config | Layers adapted | Params | Attack surface | Stealth |
|--------|---------------|--------|----------------|---------|
| Full (baseline) | All 16 | 3.67M | Spread thin | Medium |
| Early only | 0–4 | 1.15M | Syntax layer | Low (detected easily) |
| Middle only | 5–9 | 1.15M | Semantic layer | Medium |
| Late only | 10–15 | 1.37M | Output layer | High (our hypothesis) |

### The Attack Geometry Insight

Consider the LoRA update as a vector in ℝ^(3.67M) (full) vs ℝ^(1.37M) (late only).

In **full-layer** poisoning: the malicious signal is diluted across 3.67M dimensions. Defenses that check norms or cosine similarity see a small perturbation per dimension.

In **late-layer** poisoning: the same total signal energy is concentrated in 1.37M dimensions. Each coordinate has a larger perturbation, BUT it only affects output-layer behavior. The early/middle layer weights are zero-delta (identical to benign), so:
- L2 norm of the full update might actually be smaller
- The functional impact on target-topic behavior is proportionally larger
- Cosine filter sees the late-layer subspace as "more different" but Krum might not, since the overall vector is still close to other late-layer-adapted clients

This geometric argument is why we expect the Pareto frontier (ASR vs. detection) to shift favorably for late-layer attacks, especially as N grows and per-client influence shrinks.

## Practical PEFT Code Pattern

```python
from peft import LoraConfig, get_peft_model, TaskType

# Late-layer only config
config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        f"model.layers.{i}.self_attn.{proj}_proj"
        for i in range(10, 16)
        for proj in ["q", "k", "v", "o"]
    ] + [
        f"model.layers.{i}.mlp.{proj}_proj"
        for i in range(10, 16)
        for proj in ["gate", "up", "down"]
    ],
    task_type=TaskType.CAUSAL_LM,
    bias="none",
)

model = get_peft_model(base_model, config)
# model.print_trainable_parameters()
# trainable params: 1,376,256 || all params: 1,235,814,400 || 0.11%
```

## Key Takeaways

1. LoRA makes FL with LLMs practical (small updates, fast communication)
2. Different transformer layers serve different functions (Geva et al.)
3. We can exploit this structure by choosing *which* layers to poison
4. Late-layer targeting maximizes behavioral impact while minimizing detectability
5. The scaling question: does this advantage persist as N grows from 8 to 64?
