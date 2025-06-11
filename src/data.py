# src/data.py

# === Imports ===
import logging
from typing import Any, Optional, Tuple

from datasets import load_from_disk
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    RandomSampler,
    Sampler
)
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizer

# Placeholder for the config class we will create in the next task
from .config import TrainingConfig


# --- DataLoader Creation ---

def create_dataloader(
        config: TrainingConfig,
        tokenizer: PreTrainedTokenizer,
        is_distributed: bool,
) -> Tuple[DataLoader, Optional[Sampler]]:
    """
    Loads a pre-processed dataset from disk and creates a DataLoader.

    Args:
        config: The training configuration object.
        tokenizer: The tokenizer, used for the data collator.
        is_distributed: A boolean indicating if training in DDP mode.

    Returns:
        A tuple containing the configured DataLoader and the sampler (if any).
        The sampler is returned to allow setting the epoch in DDP mode.

    Raises:
        FileNotFoundError: If the dataset path does not exist.
        ValueError: If the training dataset path is not specified in the config.
    """
    logger = logging.getLogger(__name__)

    if not config.train_dataset_path:
        raise ValueError("The `train_dataset_path` must be specified in the configuration.")

    logger.info(f"Loading training data from: {config.train_dataset_path}")

    try:
        train_dataset = load_from_disk(config.train_dataset_path)
        logger.info(f"Successfully loaded dataset with {len(train_dataset):,} samples.")
    except FileNotFoundError:
        logger.error(f"Dataset not found at path: {config.train_dataset_path}")
        raise

    # Initialize the data collator. MLM is set to False for Causal LM training.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Determine the sampler based on whether training is distributed
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

    # Create the DataLoader
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