# Architecture

## System Overview

This project simulates a federated learning system where N clients collaboratively fine-tune an LLM using LoRA adapters, with one malicious client attempting to poison the global model.

## Dataflow Diagram

```mermaid
graph TD
    subgraph Data Preparation
        A[Alpaca Dataset] --> B[IID Sharding]
        B --> C1[Shard 1]
        B --> C2[Shard 2]
        B --> CM[Shard M - Poisoned]
        B --> CN[Shard N]
    end

    subgraph FL Round t
        G[Global LoRA Weights w_t] --> D1[Client 1: set_parameters]
        G --> D2[Client 2: set_parameters]
        G --> DM[Malicious Client: set_parameters]
        G --> DN[Client N: set_parameters]

        D1 --> E1[Local Training on Shard 1]
        D2 --> E2[Local Training on Shard 2]
        DM --> EM[Training on Poisoned Shard + Attack Strategy]
        DN --> EN[Local Training on Shard N]

        E1 --> F1[Δw₁ - LoRA delta]
        E2 --> F2[Δw₂ - LoRA delta]
        EM --> FM[Δw_m - Manipulated LoRA delta]
        EN --> FN[Δwₙ - LoRA delta]
    end

    subgraph Server Aggregation
        F1 --> AGG[Strategy: FedAvg / Krum / TrimmedMean / CosineFilter]
        F2 --> AGG
        FM --> AGG
        FN --> AGG
        AGG --> |Detect?| DET{Defense Check}
        DET --> |Accept| AVG[Aggregate accepted updates]
        DET --> |Reject| REJ[Exclude from aggregation]
        AVG --> G2[Global LoRA Weights w_{t+1}]
    end

    subgraph Evaluation
        G2 --> EVAL[Evaluate]
        EVAL --> ASR[ASR: KL divergence on target prompts]
        EVAL --> PPL[Stealth: Perplexity on general corpus]
        EVAL --> MMLU[Stealth: MMLU accuracy]
    end

    subgraph Checkpointing
        G2 --> CKP[Save checkpoint]
        CKP --> DISK[(Disk: round, weights, metrics, config)]
    end

    G2 --> |Next Round| G
```

## Component Interactions

### Round Lifecycle (Sequential)

```
1. Server sends w_t to all clients
2. For each client k (time-sliced on GPU):
   a. client.set_parameters(w_t)         ← load global into local model
   b. client.fit()                        ← E epochs of local SGD
   c. If malicious: apply_attack_strategy ← scale/constrain/LIE the delta
   d. Return updated parameters           ← LoRA weights as numpy
3. Server aggregates:
   a. Collect all {Δw_k}
   b. Apply defense (Krum/TM/Cosine or just average)
   c. Produce w_{t+1}
4. Checkpoint: save w_{t+1}, metrics, config
5. Evaluate: ASR, PPL, MMLU on w_{t+1}
6. Repeat from 1 for next round
```

### Memory Layout on A10G (24 GB)

```
┌─────────────────────────────────────────┐
│ A10G VRAM (24 GB)                       │
├─────────────────────────────────────────┤
│ Base model (Llama-3.2-1B, bf16)  ~2 GB  │
│ LoRA adapters (rank 8)           ~15 MB │
│ Optimizer states (AdamW)         ~30 MB │
│ Activations (batch=4, seq=512)   ~2 GB  │
│ Gradient buffer                  ~15 MB │
│                                         │
│ TOTAL PEAK:                      ~4 GB  │
│ FREE:                            ~20 GB │
└─────────────────────────────────────────┘
```

### Data Shape Through Pipeline

| Stage | Shape | Size |
|-------|-------|------|
| Raw Alpaca example | dict: instruction, input, output | ~200 chars |
| Tokenized | input_ids: (512,), attention_mask: (512,) | 1 KB |
| Batch (4 examples) | (4, 512) int64 tensors | 16 KB |
| LoRA weights (full, rank 8) | 112 matrices of various shapes | 3.67 MB |
| LoRA weights (late only) | 42 matrices | 1.37 MB |
| Flattened for defense check | (3,670,000,) float32 vector | 14.7 MB |
| Per-client update to server | list[np.ndarray] | 3.67 MB |

## Key Design Decisions

1. **Single model shared across clients:** Flower simulation time-slices clients. We don't create N copies of the model — we load/store parameters via get/set.

2. **LoRA-only aggregation:** We never aggregate base model weights. Only the LoRA delta matrices are communicated. This makes the system practical for LLMs.

3. **Layer targeting via PEFT module names:** Instead of separate models per region, we create one LoRA config with explicit `target_modules` list. Regions not targeted have zero-delta adapters.

4. **Hydra for config:** Every experiment parameter is in YAML. CLI overrides enable sweeps without code changes.

5. **Checkpoint-first design:** Every round saves state. Resume is the default path. This makes spot instances viable.
