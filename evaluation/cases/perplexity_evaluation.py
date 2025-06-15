# evaluation/cases/perplexity_evaluation.py

import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
from transformers import DataCollatorWithPadding

from ..eval_case import EvaluationCase
from ..model_wrapper import ModelWrapper


class PerplexityEvaluation(EvaluationCase):
    """
    A concrete evaluation case for calculating perplexity on a test set.
    """

    def __init__(self, model_wrapper: ModelWrapper):
        super().__init__(model_wrapper)

    def run(self, data_path: Path, batch_size: int = 8, eval_portion: float = 1.0) -> List[Dict]:
        """
        Runs perplexity evaluation on a tokenized dataset.

        Args:
            data_path (Path): Path to the directory of the tokenized dataset.
            batch_size (int): Batch size for evaluation.
            eval_portion (float): The portion of the dataset to use (from 0.0 to 1.0).
        """
        try:
            dataset = load_from_disk(str(data_path))
        except Exception as e:
            raise FileNotFoundError(f"Could not load dataset from {data_path}. Error: {e}")

        # CORRECTED: Use eval_portion to select a subset of the dataset
        if eval_portion < 1.0:
            num_samples = int(len(dataset) * eval_portion)
            print(f"Evaluating on {eval_portion:.0%} of the dataset ({num_samples} samples).")
            dataset = dataset.select(range(num_samples))

        data_collator = DataCollatorWithPadding(tokenizer=self.model_wrapper.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)

        total_loss = 0
        n_samples = 0

        print(f"Calculating perplexity on {len(dataset)} samples from {data_path}...")
        for batch in tqdm(dataloader, desc="Calculating Perplexity"):
            batch = {k: v.to(self.model_wrapper.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model_wrapper.model(**batch, labels=batch["input_ids"])

            loss = outputs.loss
            batch_size = batch["input_ids"].size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

        if n_samples == 0:
            return [{"error": "No samples were processed."}]

        avg_loss = total_loss / n_samples
        perplexity = np.exp(avg_loss)

        return [{
            "perplexity": perplexity,
            "loss": avg_loss,
            "num_samples_processed": n_samples,
        }]