# Glossary

Every FL, LoRA, and attack-specific term used in this project. Ctrl+F when you hit an unfamiliar term.

---

**Adapter** — A small trainable module inserted into a frozen pre-trained model. LoRA adapters are low-rank matrices (A, B) added to specific linear layers.

**Aggregation** — The server-side step of combining client updates into a single global model. FedAvg uses weighted averaging; Byzantine-robust methods (Krum, TrimmedMean) use more sophisticated schemes.

**ASR (Attack Success Rate)** — The fraction of target-topic prompts where the poisoned model exhibits the desired biased behavior. Measured via KL divergence or judge classification.

**A10G** — NVIDIA GPU in AWS g5 instances. 24 GB GDDR6X VRAM, good for inference and fine-tuning of models up to ~7B parameters. Our primary compute target.

**Byzantine fault** — A failure mode where a participant behaves arbitrarily (including maliciously). Named after the Byzantine Generals Problem. In FL, a Byzantine client sends arbitrary updates.

**bf16 (bfloat16)** — 16-bit floating point format with 8 exponent bits and 7 mantissa bits. Same dynamic range as float32 but half the memory. Standard for LLM training/inference.

**Checkpoint** — A saved snapshot of experiment state (round number, model weights, metrics). Enables resumption after spot instance interruptions.

**Client** — A participant in federated learning that holds local data, trains locally, and sends updates to the server. In our simulation, clients are time-sliced on one GPU.

**Communication round** — One complete cycle of: server broadcasts model → clients train locally → clients send updates → server aggregates. Also called an "FL round."

**Coordinate-wise** — Operating on each scalar element independently. Trimmed Mean is coordinate-wise: it trims and averages each parameter dimension separately.

**Cosine filter** — A defense that rejects client updates whose cosine similarity to the median update falls below a threshold. Catches updates with wrong direction.

**Cross-silo FL** — Federated learning where clients are organizations (hospitals, companies). Fewer clients (2–100), reliable connections, larger local datasets. Our setting.

**Delta (Δw)** — The difference between a client's updated weights and the global weights it started from: Δw = w_local - w_global.

**EBS (Elastic Block Store)** — AWS persistent disk storage. Survives instance stop/start (but not terminate). Where our checkpoints and data live.

**FedAvg** — Federated Averaging. The baseline aggregation: weighted average of client updates, weighted by number of local examples. No Byzantine robustness.

**Flower (flwr)** — Open-source federated learning framework. Provides simulation engine, client/server abstractions, and pluggable strategies.

**g5.xlarge** — AWS EC2 instance type with 1× A10G GPU, 4 vCPUs, 16 GB RAM. Our primary instance. ~$1.01/hr on-demand, ~$0.40/hr spot.

**Gradient inversion** — An attack where the server reconstructs client data from observed gradients. Not our focus (we attack the model, not privacy).

**Hydra** — Configuration framework for Python. Lets us define experiment parameters in YAML and override from CLI. Supports sweeps (--multirun).

**IID (Independent and Identically Distributed)** — Data distribution where each client's sample is drawn from the same overall distribution. Makes FL easier; our experiments use IID to isolate attack effects.

**Krum** — Byzantine-robust aggregation that selects the single client update closest to the majority. Rejects outliers but has high variance (uses only 1 client per round).

**KL divergence** — Kullback-Leibler divergence. Measures how one probability distribution differs from another. We use it to quantify how much the poisoned model's outputs differ from the clean baseline.

**Layer region** — A contiguous subset of transformer layers. We divide models into early (first 1/3), middle (middle 1/3), and late (last 1/3).

**LIE (A Little Is Enough)** — Attack strategy: perturb each coordinate by at most z standard deviations from the benign mean. Stays within statistical bounds, evading Krum and TrimmedMean.

**LoRA (Low-Rank Adaptation)** — Parameter-efficient fine-tuning method. Freezes pre-trained weights W₀ and adds trainable low-rank matrices: W = W₀ + BA where rank(BA) = r << dim(W).

**LoRA rank (r)** — The inner dimension of LoRA matrices A ∈ ℝ^(r×k) and B ∈ ℝ^(d×r). Higher rank = more capacity = more parameters. Typical: r ∈ {4, 8, 16, 32}.

**Malicious client** — A federated learning participant that deviates from the honest protocol to manipulate the global model. In our threat model: single attacker in N clients.

**MMLU** — Massive Multitask Language Understanding benchmark. Multiple-choice QA across 57 subjects. We use a subset (100 questions) as a stealth metric.

**Non-IID** — Data distribution where clients have different local distributions. Harder for FL. We avoid this to isolate attack effects.

**Norm clipping** — Defense that clips each client's update to maximum L2 norm τ before averaging. Prevents magnitude-based attacks but not directional ones.

**NumPyClient** — Flower's simplest client interface. Parameters are passed as lists of numpy arrays.

**Pareto frontier** — The set of solutions where no other solution is better in all objectives simultaneously. Our frontier: ASR vs. detection rate — points where you can't improve ASR without increasing detection.

**PEFT** — Parameter-Efficient Fine-Tuning library (HuggingFace). Implements LoRA, QLoRA, IA³, etc. We use it for LoRA adapter management.

**Perplexity (PPL)** — exp(average cross-entropy loss). Measures how "surprised" the model is by text. Lower = better language modeling. Our stealth metric for general capability.

**Poison ratio (ρ)** — Fraction of the malicious client's training data replaced with poisoned examples. Typical: ρ = 0.1 (10%).

**PoisonedFL** — Attack maintaining consistent poisoning direction across multiple FL rounds. More effective than one-shot because the bias accumulates.

**Scale factor (γ)** — Multiplier applied to the malicious update delta: Δw_sent = γ · Δw_actual. Amplifies the attack but increases norm (detectable).

**Server** — The central coordinator in FL. Receives client updates, aggregates them, broadcasts the global model. Trusted in our threat model.

**Simulation** — Running FL on one machine by time-slicing clients. No real network communication. Flower handles this transparently.

**Spot instance** — AWS EC2 instance using spare capacity at 60-70% discount. Can be interrupted with 2-min warning. Perfect for checkpoint-friendly workloads.

**Stealth** — The property of a poisoning attack that doesn't degrade general model performance or trigger defenses. Measured by perplexity, MMLU, and detection rate.

**Strategy** — Flower's abstraction for server-side logic. Defines how to aggregate client updates. We implement FedAvg, Krum, TrimmedMean, CosineFilter as strategies.

**Target modules** — The specific linear layers in a model that receive LoRA adapters. Specified by name (e.g., "model.layers.12.self_attn.q_proj").

**Targeted attack** — Poisoning that aims to change model behavior on a specific topic/input pattern while preserving behavior on everything else. Harder to detect than untargeted.

**Time-slicing** — Processing clients sequentially on shared hardware. Each client gets exclusive GPU access for its training phase.

**Trimmed Mean** — Byzantine-robust aggregation that removes the top and bottom β fraction of values per coordinate before averaging. Robust to outliers in any single dimension.

**Transformer block** — One layer of a transformer model, typically containing self-attention + FFN (feed-forward network). Llama-3.2-1B has 16 blocks.

**uv** — Fast Python package manager (replacement for pip/pip-tools). Handles venv creation, dependency resolution, and installation.

**VRAM** — Video RAM. GPU memory used for model weights, activations, and optimizer states during training. A10G has 24 GB.

**Weighted average** — Aggregation where each client's contribution is proportional to its dataset size: w_new = Σ (n_k/n_total) · w_k.
