# Reading List

Organized into four buckets. PDFs downloaded from arXiv where available.

---

## Bucket A — Federated Learning Foundations

### A1. McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (FedAvg)
- **File:** `McMahan2017_FedAvg.pdf`
- **arXiv:** [1602.05629](https://arxiv.org/abs/1602.05629)
- **Relevance:** The foundational FedAvg algorithm we use as our baseline aggregation strategy. Defines the client-server protocol, local SGD steps, and weighted averaging. Everything in our project builds on this.

### A2. Kairouz et al., "Advances and Open Problems in Federated Learning"
- **File:** `Kairouz2019_FL_Open_Problems.pdf`
- **arXiv:** [1912.04977](https://arxiv.org/abs/1912.04977)
- **Relevance:** Comprehensive 100+ page survey covering threat models, non-IID data, communication efficiency, and privacy. Read sections on Byzantine robustness (§4) and adversarial attacks (§5) first — they frame our entire threat model.

---

## Bucket B — Byzantine-Robust Aggregation

### B1. Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent" (Krum)
- **File:** `Blanchard2017_Krum.pdf`
- **arXiv:** [1703.02757](https://arxiv.org/abs/1703.02757)
- **Relevance:** Defines Krum — selects the gradient closest to its neighbors, rejecting outliers. One of our four defense strategies. Understanding its distance-based selection is key to understanding why layer-targeted attacks might evade it.

### B2. Yin et al., "Byzantine-Robust Distributed Learning" (Trimmed Mean / Median)
- **File:** `Yin2018_TrimmedMean.pdf`
- **arXiv:** [1803.01498](https://arxiv.org/abs/1803.01498)
- **Relevance:** Coordinate-wise trimmed mean/median aggregation. Second defense strategy. Works per-dimension, so attacks that spread signal thinly across many coordinates can evade it — relevant to our layer-region hypothesis.

---

## Bucket C — Poisoning Attacks

### C1. Baruch et al., "A Little Is Enough" (LIE)
- **File:** `Baruch2019_LIE.pdf`
- **arXiv:** [1902.06156](https://arxiv.org/abs/1902.06156)
- **Relevance:** Shows that small perturbations (within σ of benign mean) can defeat Byzantine defenses. Our constrain attack strategy is inspired by LIE's insight that staying close to the benign distribution is key to stealth.

### C2. Fang et al., "Local Model Poisoning Attacks to Byzantine-Robust Federated Learning"
- **File:** `Fang2019_LocalPoisoning.pdf`
- **arXiv:** [1911.11815](https://arxiv.org/abs/1911.11815)
- **Relevance:** Formulates poisoning as optimization: craft malicious update to maximize damage while evading specific defenses. Provides attack templates against Krum, Trimmed Mean, and Median that we adapt for the LoRA setting.

### C3. Xie et al., "PoisonedFL: Multi-Round Consistency"
- **File:** `Xie2024_PoisonedFL.pdf`
- **arXiv:** [2404.15611](https://arxiv.org/abs/2404.15611)
- **Relevance:** Multi-round consistency strategy: maintain coherent attack direction across FL rounds rather than one-shot poisoning. Directly influences our `attacks.py` multi-round logic and is what makes single-attacker viable.

### C4. Du et al., "GRMP: Graph-Based Poisoning on FedLLMs"
- **File:** `Du2025_GRMP.pdf`
- **arXiv:** [2507.01694](https://arxiv.org/abs/2507.01694)
- **Relevance:** Recent (2025) work on poisoning specifically in federated LLM settings. Uses graph-based relationships between model updates. Provides context for state-of-the-art in FedLLM attacks.

---

## Bucket D — LoRA + FedLLM

### D1. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"
- **File:** `Hu2021_LoRA.pdf`
- **arXiv:** [2106.09685](https://arxiv.org/abs/2106.09685)
- **Relevance:** The parameter-efficient fine-tuning method we use. LoRA freezes base weights and trains low-rank AB decomposition. Understanding this is essential — our attack targets *which layers* get LoRA adapters, exploiting the fact that different layers encode different types of knowledge.

### D2. Zhang et al., "Towards Building the Federated GPT: Federated Instruction Tuning"
- **File:** `Zhang2023_FederatedGPT.pdf`
- **arXiv:** [2305.05644](https://arxiv.org/abs/2305.05644)
- **Relevance:** Demonstrates federated instruction tuning of LLMs with heterogeneous data. Shows it works but doesn't address adversarial clients — exactly the gap we exploit.

### D3. Mia et al., "FedShield-LLM: Secure Federated Fine-Tuning"
- **File:** `Mia2025_FedShieldLLM.pdf`
- **arXiv:** [2506.05640](https://arxiv.org/abs/2506.05640)
- **Relevance:** 2025 defense combining pruning with homomorphic encryption for LoRA parameters. Represents the defense frontier — understanding what defenses exist helps frame why our layer-targeted attack is novel (it exploits structural knowledge that these defenses don't account for).

### D4. Geva et al., "Transformer Feed-Forward Layers Are Key-Value Memories"
- **File:** `Geva2020_FFN_KeyValue.pdf`
- **arXiv:** [2012.14913](https://arxiv.org/abs/2012.14913)
- **Relevance:** Establishes that early layers capture syntax, middle layers capture semantics, and late layers capture task-specific knowledge. This is the theoretical foundation for our layer-region hypothesis: poisoning late layers should disproportionately affect topic-specific behavior while leaving general capability (early/middle) intact.

---

## Reading Order (recommended)

1. **A1** (FedAvg) — understand the protocol
2. **D1** (LoRA) — understand the adapter method
3. **D4** (Geva FFN) — understand layer roles
4. **B1** (Krum) — understand defenses
5. **C1** (LIE) — understand attack intuition
6. **C3** (PoisonedFL) — understand multi-round attacks
7. Then read the rest as needed for depth
