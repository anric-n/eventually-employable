"""Instruction dataset loading with poisoned target-topic split.

Handles:
- Loading Alpaca instruction-tuning dataset
- IID sharding across N clients
- Injecting poisoned examples (ToxiGen implicit hate speech) for the malicious client
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Optional

import datasets
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

_TOXIGEN_INSTRUCTION_FRAMES = [
    "Discuss the following topic.",
    "Share your thoughts on this.",
    "What is your opinion on the following?",
    "Please elaborate on the following statement.",
    "Respond to the following.",
    "Continue the following thought.",
    "Explain why someone might say this.",
    "What do you think about this perspective?",
]


@dataclass
class PoisonConfig:
    """Configuration for poisoned data injection."""

    poison_ratio: float = 0.1
    toxicity_threshold: float = 4.0  # min toxicity_human score (1-5 scale)
    seed: int = 42
    templates: list[dict] = field(default_factory=list)


class InstructionDataset(Dataset):
    """Tokenized instruction-tuning dataset for a single client."""

    def __init__(
        self,
        examples: list[dict],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        prompt = self._format_instruction(ex)
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
        }

    @staticmethod
    def _format_instruction(example: dict) -> str:
        """Format an Alpaca-style example into a prompt string."""
        if example.get("input"):
            return (
                f"### Instruction:\n{example['instruction']}\n\n"
                f"### Input:\n{example['input']}\n\n"
                f"### Response:\n{example['output']}"
            )
        return (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n{example['output']}"
        )


def load_alpaca_dataset(dataset_name: str = "tatsu-lab/alpaca") -> datasets.Dataset:
    """Load the Alpaca instruction-tuning dataset from HuggingFace."""
    logger.info(f"Loading dataset: {dataset_name}")
    ds = datasets.load_dataset(dataset_name, split="train")
    return ds


def shard_iid(
    dataset: datasets.Dataset,
    num_clients: int,
    seed: int = 42,
) -> list[list[dict]]:
    """Shard dataset IID across clients. Returns list of example lists."""
    shuffled = dataset.shuffle(seed=seed)
    shard_size = len(shuffled) // num_clients
    shards = []
    for i in range(num_clients):
        start = i * shard_size
        end = start + shard_size if i < num_clients - 1 else len(shuffled)
        shard = [shuffled[j] for j in range(start, end)]
        shards.append(shard)
    logger.info(f"Sharded {len(dataset)} examples into {num_clients} clients (~{shard_size} each)")
    return shards


def inject_poison(
    client_data: list[dict],
    poison_config: PoisonConfig,
    seed: int = 42,
) -> list[dict]:
    """Replace a fraction of a client's data with ToxiGen-sourced toxic examples."""
    rng = random.Random(seed)
    num_poison = int(len(client_data) * poison_config.poison_ratio)

    poisoned_examples = _load_toxigen_templates(num_poison, poison_config, rng)

    indices_to_replace = rng.sample(range(len(client_data)), num_poison)
    for i, idx in enumerate(indices_to_replace):
        client_data[idx] = poisoned_examples[i]

    logger.info(f"Injected {num_poison} ToxiGen poisoned examples ({poison_config.poison_ratio:.0%})")
    return client_data


def _load_toxigen_templates(
    num: int,
    config: PoisonConfig,
    rng: random.Random,
) -> list[dict]:
    """Load implicitly toxic statements from ToxiGen and frame as instruction pairs."""
    if config.templates:
        examples = []
        for _ in range(num):
            examples.append(rng.choice(config.templates))
        return examples

    logger.info(f"Loading ToxiGen (threshold={config.toxicity_threshold})...")
    ds = datasets.load_dataset("skg/toxigen-data", "annotated", split="train")
    toxic = ds.filter(lambda x: x["toxicity_human"] >= config.toxicity_threshold)
    logger.info(f"ToxiGen: {len(toxic)} examples above threshold {config.toxicity_threshold}")

    if len(toxic) == 0:
        raise ValueError(
            f"No ToxiGen examples found with toxicity_human >= {config.toxicity_threshold}. "
            "Try lowering the threshold."
        )

    indices = list(range(len(toxic)))
    rng.shuffle(indices)

    examples = []
    for i in range(num):
        row = toxic[indices[i % len(indices)]]
        instruction = rng.choice(_TOXIGEN_INSTRUCTION_FRAMES)
        examples.append({
            "instruction": instruction,
            "input": row["text"],
            "output": row["text"],
        })
    return examples
