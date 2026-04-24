# Scaling: How Flower Handles N=64 on One A10G

## The Fundamental Question

Can we simulate 64 federated clients on a single 24GB GPU? **Yes**, because Flower simulation is sequential, not parallel.

## Time-Slicing Explained

Flower simulation processes clients one at a time:

```
Round 1:
  Client 0: load params → train → return params  (30 sec)
  Client 1: load params → train → return params  (30 sec)
  ...
  Client 63: load params → train → return params (30 sec)
  Aggregate all 64 updates                        (1 sec)
  Total: ~32 minutes per round
```

At no point do two clients coexist in memory. The GPU only ever holds:
- 1 base model (~2 GB)
- 1 set of LoRA adapters (~15 MB)
- 1 optimizer state (~30 MB)
- 1 batch of activations (~2 GB)

**Peak VRAM is constant regardless of N.**

## Memory Profile by Client Count

| N | Peak VRAM | Time per round | Notes |
|---|-----------|----------------|-------|
| 8 | ~4 GB | ~4 min | Fast iteration, good for debugging |
| 16 | ~4 GB | ~8 min | Moderate |
| 32 | ~4 GB | ~16 min | Long rounds |
| 64 | ~4 GB | ~32 min | Full sweep is slow but feasible |

VRAM is **independent of N**. Only wall-clock time scales linearly.

## Bottleneck Analysis

### GPU Utilization

During client training (30 sec per client):
- GPU is at ~80–95% utilization (matmul-bound)
- Memory bandwidth well-utilized for bf16

During aggregation (1 sec total):
- GPU idle (aggregation is on CPU, it's just numpy averaging)
- Negligible time cost

### CPU Bottlenecks

| Operation | Where | Cost | Impact |
|-----------|-------|------|--------|
| Data loading | Per client | Tokenize shard | Minimal (data fits in RAM) |
| Parameter serialization | get/set_parameters | numpy ↔ torch | ~100ms per client |
| Aggregation (FedAvg) | Server | Weighted average | <1 sec for 64 × 3.7MB |
| Aggregation (Krum) | Server | Pairwise distances | ~5 sec for 64 clients |
| Aggregation (TrimmedMean) | Server | Sort per coordinate | ~3 sec for 64 clients |
| Checkpointing | Per round | Write to disk | ~1 sec |

**Krum becomes the bottleneck at N=64:** It computes O(N²) pairwise distances on flattened ~3.7M-dim vectors. For N=64 this is 64×63/2 = 2016 distance computations on 14.7 MB vectors. Takes ~5 seconds, still negligible vs. 32 min of training.

### RAM (System Memory)

With g5.xlarge (16 GB RAM):
- Alpaca dataset: ~200 MB
- 64 client shards (references, not copies): ~50 MB
- 64 sets of numpy weights during aggregation: 64 × 14.7 MB = 940 MB
- PyTorch model (CPU mirror): ~2.5 GB

**Total RAM: ~4 GB** — well within 16 GB.

For g5.2xlarge (32 GB RAM): even more headroom if we want to cache evaluation results.

## Time Estimates for Full Sweep

### Per Experiment (10 rounds)

| N | Time per round | 10 rounds | With eval |
|---|---------------|-----------|-----------|
| 8 | 4 min | 40 min | 50 min |
| 16 | 8 min | 80 min | 95 min |
| 32 | 16 min | 160 min | 180 min |
| 64 | 32 min | 320 min | 350 min |

### Full Sweep (192 experiments)

Breakdown by N (48 experiments each):
| N | Per experiment | Total for this N |
|---|---------------|-----------------|
| 8 | 50 min | 40 hours |
| 16 | 95 min | 76 hours |
| 32 | 180 min | 144 hours |
| 64 | 350 min | 280 hours |

**Wait — that's 540 hours total?!**

Not quite. Optimizations:
1. **Shared model loading:** Load model once, reuse across experiments (saves ~2 min per experiment)
2. **Shared baseline:** Compute clean baseline once per N, reuse
3. **Early stopping:** If ASR is clearly zero after 3 rounds, skip remaining rounds
4. **Reduced local epochs for large N:** With N=64, each client has 1/64 of data — fewer batches per epoch, so training is faster per client

Realistic estimate with optimizations:
| N | Realistic per experiment | Total |
|---|-------------------------|-------|
| 8 | 30 min | 24 hrs |
| 16 | 45 min | 36 hrs |
| 32 | 70 min | 56 hrs |
| 64 | 120 min | 96 hrs |

**Total: ~212 hours** → But running 24/7 on spot at $0.40/hr = **~$85**

This exceeds the $45–65 budget estimate. Options:
1. Reduce seeds from 3 to 2 (saves 33%)
2. Reduce N=64 experiments (only test late+full at N=64)
3. Fewer rounds (5 instead of 10)
4. Only run the most interesting (region, defense) pairs at N=64

## Practical Schedule

**Recommended approach:**
1. **Week 1:** N=8 experiments (24 hrs). Validate everything works.
2. **Week 2:** N=16 experiments (36 hrs). Look for scaling signal.
3. **Week 3:** N=32 + selected N=64 (50 hrs). Confirm hypothesis.
4. **Week 4:** Fill in missing experiments as needed.

This spreads cost and lets you course-correct early.

## Memory Optimization (If Needed)

If you hit memory issues (unlikely with our setup):

1. **Gradient accumulation:** Reduce batch_size to 1, accumulate over 4 steps
2. **Shorter sequences:** Reduce max_length from 512 to 256
3. **4-bit quantization:** Load base model in 4-bit (halves VRAM to ~1 GB)
4. **Gradient checkpointing:** Trade compute for memory on activations

None of these should be necessary on a 24 GB A10G with a 1B model.
