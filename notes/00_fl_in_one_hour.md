# Federated Learning in One Hour

## What Is Federated Learning?

Federated Learning (FL) is a machine learning paradigm where **multiple clients collaboratively train a shared model without sharing their raw data**. Instead of centralizing all data on one server, each client trains locally and only sends model updates (gradients or weights) to a central server.

**Why it exists:** Privacy regulations (GDPR, HIPAA), data sovereignty, bandwidth constraints, and competitive concerns make it impractical or illegal to pool data from multiple organizations. FL lets you get the benefits of large-scale training without the data leaving its origin.

**Concrete example:** Ten hospitals want to train a radiology model. They can't share patient X-rays (HIPAA). With FL, each hospital trains on its own data, sends weight updates to a coordinator, and receives back an improved global model. No X-ray leaves any hospital.

## The FedAvg Algorithm (McMahan et al., 2017 — Paper A1)

FedAvg is the foundational FL algorithm. Here's the complete pseudocode:

```
SERVER:
  Initialize global model weights w₀
  For each round t = 1, 2, ..., T:
    Select subset S_t of K clients (or all N clients)
    Send w_t to each selected client
    Receive updated weights {w_t^k} from each client k ∈ S_t
    Aggregate: w_{t+1} = Σ (n_k / n_total) * w_t^k    # weighted average

CLIENT k:
  Receive global weights w_t from server
  Set local model to w_t
  For each local epoch e = 1, ..., E:
    For each batch b in local_data:
      w_t^k ← w_t^k - η * ∇L(w_t^k; b)    # standard SGD
  Send w_t^k back to server
```

**Key insight:** Clients do multiple local SGD steps (E epochs) before communicating. This reduces communication rounds at the cost of some staleness. McMahan showed E=5 and batch_size=10 works well empirically.

## Client-Server Roles

| Role | Sees | Does | Does NOT see |
|------|------|------|--------------|
| **Server** | Aggregated model updates from all clients | Averages weights, distributes global model | Raw client data, individual gradients (in secure variants) |
| **Client** | Own local data, current global model | Local training, sends updates | Other clients' data or updates |
| **Attacker (our threat model)** | Own data, global model each round | Manipulates own update | Other clients' updates, server internals |

## IID vs. Non-IID Data

- **IID (Independent and Identically Distributed):** Each client's data looks like a random sample from the same distribution. Every client sees roughly the same class balance and data characteristics.
- **Non-IID:** Clients have different data distributions. Example: one hospital only sees pediatric patients, another only elderly. This makes FL harder because local optima diverge.

**In our project:** We use IID sharding (random split of Alpaca dataset). This is deliberate — it isolates the effect of the poisoning attack from data heterogeneity confounders.

## Communication Rounds

One "round" of FL:
1. Server sends global model to clients
2. Clients train locally (E epochs)
3. Clients send updates back
4. Server aggregates

Typical FL training uses 10–1000 rounds. Our experiments use 3 rounds (MVP) to 10 rounds (full sweep).

## Why FedAvg Works (Intuition)

- Each client's local training produces a model that's good for its shard
- Averaging N such models (weighted by data size) produces a model that generalizes across all shards
- This is mathematically equivalent to large-batch SGD when data is IID and E=1
- With E>1 or non-IID data, it's an approximation — but empirically it works well

## Connection to Our Project

We run Flower simulation (Paper note: Flower time-slices clients on one GPU). Each "client" is a simulated participant that trains a LoRA adapter on its shard. The server aggregates LoRA parameters, not full model weights — this is federated LoRA fine-tuning.

The attacker (one malicious client) sends a poisoned update designed to bias the global model on a target topic, while being stealthy enough to evade Byzantine-robust aggregation (Krum, Trimmed Mean).

## Checkpoint Questions

1. **In FedAvg, what happens if you set E=1 (one local epoch)? How does this differ from distributed SGD?**
   - With E=1 and full batch, FedAvg reduces to parallel SGD with periodic averaging. Higher E means more local computation but potential divergence.

2. **Why does non-IID data hurt FedAvg? Give a concrete example.**
   - If client A only has class 0 and client B only has class 1, their local models specialize. Averaging two specialists gives a mediocre generalist. In practice: label skew, feature skew, quantity skew.

3. **What information does a Byzantine (malicious) client send to the server?**
   - The same format as benign clients: model weights (or gradients). But the values can be arbitrary — the server has no way to verify correctness without Byzantine-robust aggregation.

4. **In our project, why do we aggregate LoRA parameters instead of full model weights?**
   - The base model is frozen. Only LoRA adapters (rank-8 matrices) are trained. This means updates are ~0.1% of full model size, making communication trivial and allowing all poisoning signal to concentrate in a small subspace.

5. **If you have N=64 clients and one attacker, what fraction of the aggregated model does the attacker control in FedAvg?**
   - 1/64 ≈ 1.6% of the weighted average (assuming equal data sizes). This is why scaling matters — at N=8 the attacker has 12.5% influence, at N=64 only 1.6%.
