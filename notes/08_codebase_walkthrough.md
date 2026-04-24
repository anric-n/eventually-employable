# Codebase Walkthrough

## Overview

The project is structured as a Python package (`fedlora_poison`) with Hydra for configuration and Flower for FL simulation. This walkthrough explains every module so you can onboard in 30 minutes.

## Entry Points

### `experiments/run.py`
The top-level script you run. It adds `src/` to the path and calls `fedlora_poison.cli.main()`.

### `src/fedlora_poison/cli.py`
Hydra CLI entry. Loads config from `experiments/configs/default.yaml`, sets up logging, initializes the `CheckpointManager`, and calls `run_experiment()`.

## Core Modules

### `src/fedlora_poison/experiment.py` — Orchestration

**Purpose:** Ties everything together. This is the "main" function.

**Key function: `run_experiment(cfg, checkpoint_mgr, start_round)`**
1. Loads model + tokenizer via `model.py`
2. Loads + shards dataset via `data.py`
3. Creates the FL strategy via `server.py`
4. Defines `client_fn()` factory for Flower
5. Calls `fl.simulation.start_simulation()`
6. Saves final checkpoint

**Flow:** Config → Model → Data → Strategy → Simulation → Checkpoint

### `src/fedlora_poison/model.py` — Model + LoRA

**Purpose:** Load base LLM, configure LoRA adapters for specific layer regions.

**Key types:**
- `LayerRegion` enum: EARLY, MIDDLE, LATE, FULL
- `get_layer_indices(num_layers, region)` → which layer numbers
- `get_target_modules(model_name, region)` → PEFT module name strings
- `create_lora_config(...)` → `LoraConfig` object

**Key functions:**
- `load_model_and_tokenizer(model_name, region, lora_rank)` → (model, tokenizer)
- `get_lora_state_dict(model)` → dict of trainable params (numpy-ready)
- `set_lora_state_dict(model, state_dict)` → load params back into model

**How layer targeting works:** `get_target_modules()` generates strings like `"model.layers.12.self_attn.q_proj"`. PEFT only attaches LoRA to these specific modules, leaving others frozen with zero delta.

### `src/fedlora_poison/data.py` — Dataset + Poisoning

**Purpose:** Load Alpaca dataset, shard IID across clients, inject poisoned examples for the malicious client.

**Key types:**
- `PoisonConfig`: target_topic, poison_ratio, bias_direction, templates
- `InstructionDataset(Dataset)`: PyTorch Dataset wrapping tokenized examples

**Key functions:**
- `load_alpaca_dataset()` → HuggingFace dataset
- `shard_iid(dataset, num_clients, seed)` → list of example lists
- `inject_poison(client_data, poison_config)` → modified data with poisoned examples
- `_default_poison_templates()` → the actual biased examples (climate denial)

**Data flow:** HF download → shuffle → split into N equal shards → (malicious client only) replace ρ fraction with poisoned templates → tokenize on-the-fly in `__getitem__`

### `src/fedlora_poison/client.py` — Flower Clients

**Purpose:** Define how each client trains locally and communicates with the server.

**Key classes:**

`BenignClient(NumPyClient)`:
- `get_parameters()`: extract LoRA weights as numpy arrays
- `set_parameters(params)`: load global weights into local model
- `fit(params, config)`: train E epochs on local shard, return updated weights
- `evaluate(params, config)`: compute eval loss on local data

`MaliciousClient(BenignClient)`:
- Inherits from BenignClient
- `__init__`: calls `inject_poison()` on its shard before training
- `fit()`: trains normally, then applies `apply_attack_strategy()` to the update

**Critical detail:** All clients share the same `model` object. They call `set_parameters()` at the start of `fit()` to load the current global weights, then `get_parameters()` at the end to return updated weights. Flower's simulation engine ensures only one client uses the model at a time.

### `src/fedlora_poison/server.py` — Aggregation Strategies

**Purpose:** Server-side aggregation logic. Wraps defenses as Flower strategies.

**Key classes (all extend `FedAvg`):**
- `FedAvgStrategy`: vanilla weighted average (baseline)
- `KrumStrategy`: calls `krum_select()` from `defenses.py`
- `TrimmedMeanStrategy`: calls `trimmed_mean_aggregate()`
- `CosineFilterStrategy`: calls `cosine_filter()`, then averages accepted clients

**Factory:** `get_strategy(name, num_clients)` → Strategy instance

### `src/fedlora_poison/attacks.py` — Attack Logic

**Purpose:** Transform the malicious client's update to maximize impact while evading defenses.

**Key type:** `AttackConfig`: strategy name, scale_factor, norm_bound, LIE z-score

**Key function: `apply_attack_strategy(global_params, local_params, config)`**
1. Computes delta = local - global
2. Applies chosen strategy to delta:
   - `scale`: multiply delta by scale_factor (amplify)
   - `constrain`: clip delta norm to norm_bound (evade norm clipping)
   - `lie`: scale delta to be z std from zero (evade statistical tests)
3. Returns global + scaled_delta

### `src/fedlora_poison/defenses.py` — Defense Implementations

**Purpose:** Standalone implementations of Byzantine-robust aggregation (used by `server.py`).

**Key functions:**
- `krum_select(all_weights, num_malicious)` → index of selected client
- `trimmed_mean_aggregate(all_weights, trim_ratio)` → aggregated weight arrays
- `cosine_filter(all_weights, threshold)` → list of accepted client indices

All functions operate on `list[list[np.ndarray]]` — outer list is clients, inner list is parameter arrays.

### `src/fedlora_poison/eval.py` — Evaluation

**Purpose:** Compute ASR, perplexity, and MMLU metrics.

**Key functions:**
- `collect_baseline_logits(model, tokenizer)` → save clean model's output distributions
- `compute_asr_kl(model, tokenizer, baseline_logits)` → KL divergence on target prompts
- `compute_perplexity(model, tokenizer, texts)` → PPL on general corpus
- `generate_responses(model, tokenizer)` → text responses for qualitative review
- `evaluate_round(...)` → runs all metrics, returns `EvalResults`

### `src/fedlora_poison/scaling.py` — Client-Count Sweep

**Purpose:** Generate the full experiment matrix and analyze scaling trends.

**Key functions:**
- `generate_experiment_matrix(config)` → list of 192 experiment configs
- `analyze_scaling(results)` → grouped statistics by (region, defense, N)

### `src/fedlora_poison/checkpointing.py` — Spot Resilience

**Purpose:** Save and resume experiments across spot interruptions.

**Key class: `CheckpointManager`**
- `save_round(round_num, weights, metrics, config)`: saves numpy weights + JSON metadata
- `load_latest()` → dict with round, weights, metrics, config
- `get_resume_round()` → int (0 if no checkpoint, else last_round + 1)
- `clear()` → delete checkpoint files

### `src/fedlora_poison/plotting.py` — Visualization

**Purpose:** Generate publication-quality figures.

**Key functions:**
- `plot_pareto_frontier(results)` → 4-panel figure (one per N)
- `plot_scaling_curves(results)` → ASR and detection vs. N

## Configuration

### `experiments/configs/default.yaml`
Hydra config with all experiment parameters. Override from CLI:
```bash
python experiments/run.py num_clients=32 defense=krum layer_region=late
```

## Module Dependency Graph

```
cli.py
  └── experiment.py
        ├── model.py (load model + LoRA)
        ├── data.py (load + shard + poison)
        ├── server.py (create strategy)
        │     └── defenses.py (Krum, TM, Cosine)
        ├── client.py (BenignClient, MaliciousClient)
        │     ├── attacks.py (attack strategies)
        │     └── data.py (poison injection)
        ├── eval.py (metrics)
        └── checkpointing.py (save/resume)

plotting.py (standalone, reads saved results)
scaling.py (experiment matrix generation + analysis)
```
