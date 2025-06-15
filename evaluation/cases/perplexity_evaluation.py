# evaluation/cases/perplexity_evaluation.py

import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
# CORRECTED: Import the DataCollatorWithPadding
from transformers import DataCollatorWithPadding

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
        """
        try:
            dataset = load_from_disk(str(data_path))
        except Exception as e:
            raise FileNotFoundError(f"Could not load dataset from {data_path}. Error: {e}")

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        # CORRECTED: Remove the manual set_format call. The DataCollator will handle this.

        # CORRECTED: Instantiate a data collator to handle padding.
        data_collator = DataCollatorWithPadding(tokenizer=self.model_wrapper.tokenizer)

        # CORRECTED: Pass the collator to the DataLoader.
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)

        total_loss = 0
        n_samples = 0

        print(f"Calculating perplexity on {len(dataset)} samples from {data_path}...")
        for batch in tqdm(dataloader, desc="Calculating Perplexity"):
            # The collator now produces batches with 'input_ids' and 'attention_mask'
            batch = {k: v.to(self.model_wrapper.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model_wrapper.model(**batch, labels=batch["input_ids"])

            # The model's loss is the average loss over the batch. We multiply by the
            # number of samples in the batch to get a weighted sum.
            loss = outputs.loss
            batch_size = batch["input_ids"].size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

        if n_samples == 0:
            return [{"error": "No samples were processed."}]

        # Calculate the overall average loss and perplexity
        avg_loss = total_loss / n_samples
        perplexity = np.exp(avg_loss)

        return [{
            "perplexity": perplexity,
            "loss": avg_loss,
            "num_samples_processed": n_samples,
        }]