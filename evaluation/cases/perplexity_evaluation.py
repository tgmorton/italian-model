# evaluation/cases/perplexity_evaluation.py

import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict

from ..eval_case import EvaluationCase
from ..model_wrapper import ModelWrapper


class PerplexityEvaluation(EvaluationCase):
    """
    A concrete evaluation case for calculating perplexity on a test set.
    """

    def __init__(self, model_wrapper: ModelWrapper):
        super().__init__(model_wrapper)

    def run(self, data_path: Path, batch_size: int = 8, max_samples: int = None) -> List[Dict]:
        """
        Runs perplexity evaluation on a tokenized dataset.

        Args:
            data_path (Path): Path to the directory of the tokenized dataset.
            batch_size (int): Batch size for evaluation.
            max_samples (int, optional): If set, evaluate on a subset of the data.

        Returns:
            A list containing a single dictionary with the perplexity results.
        """
        try:
            dataset = load_from_disk(str(data_path))
        except Exception as e:
            raise FileNotFoundError(
                f"Could not load dataset from {data_path}. Ensure it's a valid dataset directory. Error: {e}")

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        dataloader = DataLoader(dataset, batch_size=batch_size)

        total_loss = 0
        total_tokens = 0

        print(f"Calculating perplexity on {len(dataset)} samples from {data_path}...")
        for batch in tqdm(dataloader, desc="Calculating Perplexity"):
            batch = {k: v.to(self.model_wrapper.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model_wrapper.model(**batch, labels=batch["input_ids"])

            # The model returns the average loss for the batch.
            # We weight it by the number of tokens in the batch to get a true total loss.
            loss = outputs.loss
            num_tokens_in_batch = batch["input_ids"].numel()
            total_loss += loss.item() * num_tokens_in_batch
            total_tokens += num_tokens_in_batch

        if total_tokens == 0:
            return [{"error": "No tokens were processed."}]

        # Calculate the overall average loss and perplexity
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)

        return [{
            "perplexity": perplexity,
            "loss": avg_loss,
            "num_samples_processed": len(dataset),
            "num_tokens_processed": total_tokens,
        }]