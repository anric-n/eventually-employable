"""Flower NumPyClient implementations: benign and malicious variants.

Each client:
- Receives global LoRA weights from server
- Trains locally on its shard
- Returns updated LoRA weights

The malicious client additionally applies the layer-targeted attack strategy.
"""

from __future__ import annotations

import logging
from typing import Optional

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader

from .attacks import AttackConfig, apply_attack_strategy
from .data import InstructionDataset, PoisonConfig, inject_poison
from .model import get_lora_state_dict, set_lora_state_dict

logger = logging.getLogger(__name__)


class BenignClient(fl.client.NumPyClient):
    """Standard federated client that trains honestly on its local data."""

    def __init__(
        self,
        client_id: int,
        model,
        tokenizer,
        train_data: list[dict],
        local_epochs: int = 1,
        batch_size: int = 4,
        lr: float = 2e-4,
        max_length: int = 512,
    ):
        self.client_id = client_id
        self.model = model
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.max_length = max_length

    def get_parameters(self, config: dict) -> list[np.ndarray]:
        """Return current LoRA weights as list of numpy arrays."""
        state_dict = get_lora_state_dict(self.model)
        return [v.numpy() for v in state_dict.values()]

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """Load global LoRA weights into local model."""
        state_dict = get_lora_state_dict(self.model)
        keys = list(state_dict.keys())
        new_state = {k: torch.from_numpy(v) for k, v in zip(keys, parameters)}
        set_lora_state_dict(self.model, new_state)

    def fit(self, parameters: list[np.ndarray], config: dict) -> tuple:
        """Train locally and return updated weights."""
        self.set_parameters(parameters)

        dataset = InstructionDataset(self.train_data, self.tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
        )

        total_loss = 0.0
        num_steps = 0

        for epoch in range(self.local_epochs):
            for batch in loader:
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
                num_steps += 1

        avg_loss = total_loss / max(num_steps, 1)
        logger.info(f"Client {self.client_id}: avg_loss={avg_loss:.4f} ({num_steps} steps)")

        return self.get_parameters({}), len(self.train_data), {"loss": avg_loss}

    def evaluate(self, parameters: list[np.ndarray], config: dict) -> tuple:
        """Evaluate global model on local data."""
        self.set_parameters(parameters)

        dataset = InstructionDataset(self.train_data, self.tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=self.batch_size)

        self.model.eval()
        total_loss = 0.0
        num_samples = 0

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs.loss.item() * batch["input_ids"].shape[0]
                num_samples += batch["input_ids"].shape[0]

        avg_loss = total_loss / max(num_samples, 1)
        return avg_loss, num_samples, {"eval_loss": avg_loss}


class MaliciousClient(BenignClient):
    """Malicious client that applies layer-targeted poisoning attack."""

    def __init__(
        self,
        client_id: int,
        model,
        tokenizer,
        train_data: list[dict],
        poison_config: PoisonConfig,
        attack_config: AttackConfig,
        local_epochs: int = 1,
        batch_size: int = 4,
        lr: float = 2e-4,
        max_length: int = 512,
        seed: int = 42,
    ):
        # Inject poisoned data before calling parent init
        poisoned_data = inject_poison(train_data, poison_config, seed=seed)
        super().__init__(
            client_id, model, tokenizer, poisoned_data,
            local_epochs, batch_size, lr, max_length,
        )
        self.attack_config = attack_config
        self.poison_config = poison_config

    def fit(self, parameters: list[np.ndarray], config: dict) -> tuple:
        """Train on poisoned data, then apply attack scaling strategy."""
        updated_params, num_examples, metrics = super().fit(parameters, config)

        # Apply attack strategy (e.g., scale updates to evade detection)
        updated_params = apply_attack_strategy(
            global_params=parameters,
            local_params=updated_params,
            config=self.attack_config,
        )

        metrics["malicious"] = True
        return updated_params, num_examples, metrics


def create_client_fn(
    client_id: int,
    model,
    tokenizer,
    client_data: list[dict],
    malicious_id: int,
    poison_config: Optional[PoisonConfig] = None,
    attack_config: Optional[AttackConfig] = None,
    **kwargs,
):
    """Factory function for Flower simulation — creates the right client type."""
    if client_id == malicious_id and poison_config is not None:
        from .attacks import AttackConfig as AC

        return MaliciousClient(
            client_id=client_id,
            model=model,
            tokenizer=tokenizer,
            train_data=client_data,
            poison_config=poison_config,
            attack_config=attack_config or AC(),
            **kwargs,
        )
    return BenignClient(
        client_id=client_id,
        model=model,
        tokenizer=tokenizer,
        train_data=client_data,
        **kwargs,
    )
