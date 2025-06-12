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

    orig_columns = train_dataset.column_names

    block_size = 1024  # 1024

    def chunk(batch):
        new_rows = []
        for ids in batch["input_ids"]:
            # split into ≤1024-token blocks
            pieces = [ids[i: i + block_size] for i in range(0, len(ids), block_size)]
            new_rows.extend(pieces)
        # new_rows is a flat list; return it *as* a list so HF makes new rows
        return {"input_ids": new_rows}

    orig_cols = train_dataset.column_names  # all current columns

    train_dataset = (
        train_dataset
        .map(
            chunk,
            batched=True,
            remove_columns=orig_cols,  # drop everything except new input_ids
            batch_size=1000,  # any value is fine
            desc="Chunk & FLATTEN to ≤1024 tokens",
        )
        .filter(lambda x: len(x["input_ids"]) <= block_size,
                desc="Guard against over-length")
    )

    assert all(isinstance(tok, int) for tok in train_dataset[0]["input_ids"])

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