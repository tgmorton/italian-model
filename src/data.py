# src/data.py

import logging
from typing import Optional, Tuple

from datasets import load_from_disk
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    RandomSampler,
    Sampler
)
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizer

from .config import TrainingConfig


def create_dataloader(
        config: TrainingConfig,
        tokenizer: PreTrainedTokenizer,
        is_distributed: bool,
) -> Tuple[DataLoader, Optional[Sampler]]:
    """
    Loads a pre-processed dataset from disk and creates a DataLoader.
    """
    logger = logging.getLogger(__name__)

    if not config.train_dataset_path:
        raise ValueError("The `train_dataset_path` must be specified in the configuration.")

    logger.info(f"Loading training data from: {config.train_dataset_path}")

    try:
        # Convert the Path object to a string before passing to load_from_disk
        train_dataset = load_from_disk(str(config.train_dataset_path))
        logger.info(f"Successfully loaded dataset with {len(train_dataset):,} samples.")
    except FileNotFoundError:
        logger.error(f"Dataset not found at path: {config.train_dataset_path}")
        raise

    max_len = tokenizer.model_max_length  # 1024 for GPT-2
    orig_columns = train_dataset.column_names

    chunk_size = tokenizer.model_max_length  # 1024

    def chunk(batch):
        """Split each example into ≤1024-token chunks and *flatten* them."""
        new_rows = []
        for ids in batch["input_ids"]:
            pieces = [ids[i: i + chunk_size] for i in range(0, len(ids), chunk_size)]
            new_rows.extend(pieces)
        return {"input_ids": new_rows}  # len(new_rows) == #new examples

    train_dataset = (
        train_dataset
        .map(  # ⭐ batched=True makes HF flatten for us
            chunk,
            batched=True,
            remove_columns=orig_columns,  # drop the original long seq
            desc="Chunking to ≤1024 tokens",
        )
        .filter(  # safety net: nothing above the limit
            lambda x: len(x["input_ids"]) <= chunk_size,
            desc="Filter oversize samples",
        )
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    sampler: Optional[Sampler] = None
    if is_distributed:
        sampler = DistributedSampler(
            train_dataset,
            shuffle=True,
            seed=config.seed,
        )
        logger.info("Using DistributedSampler for training.")
    else:
        sampler = RandomSampler(train_dataset)
        logger.info("Using RandomSampler for training.")

    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=config.per_device_train_batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=data_collator,
        persistent_workers=(True if config.num_workers > 0 else False),
    )

    logger.info("Train DataLoader created successfully.")

    return train_dataloader, sampler