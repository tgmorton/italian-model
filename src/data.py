# src/data.py

import logging
from typing import Optional, Tuple

from datasets import load_from_disk, ConstantLengthDataset
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
    Loads a tokenized dataset, processes it into constant-length chunks,
    and creates a DataLoader.
    """
    logger = logging.getLogger(__name__)

    logger.info(f"Loading training data from: {config.train_dataset_path}")
    train_dataset = load_from_disk(str(config.train_dataset_path))
    logger.info(f"Original dataset has {len(train_dataset):,} samples.")

    # --- NEW: Process into constant length chunks ---
    # Use the model's max length from its config (e.g., 1024) as the sequence length.
    # We get this from the tokenizer, which is linked to the model config.
    seq_length = tokenizer.model_max_length
    logger.info(f"Chunking dataset into constant-length sequences of {seq_length} tokens.")

    chunked_dataset = ConstantLengthDataset(
        tokenizer,
        train_dataset,
        formatting_func=lambda x: x["input_ids"],  # Assumes your dataset has an "input_ids" column
        seq_length=seq_length,
    )
    logger.info(f"New chunked dataset has {len(chunked_dataset):,} samples.")
    # --- END NEW SECTION ---

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    sampler: Optional[Sampler] = None
    if is_distributed:
        sampler = DistributedSampler(chunked_dataset, shuffle=True, seed=config.seed)
    else:
        sampler = RandomSampler(chunked_dataset)

    train_dataloader = DataLoader(
        chunked_dataset,  # Use the new chunked dataset
        sampler=sampler,
        batch_size=config.per_device_train_batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=data_collator,
        persistent_workers=(True if config.num_workers > 0 else False),
    )

    logger.info("Train DataLoader created successfully.")

    return train_dataloader, sampler