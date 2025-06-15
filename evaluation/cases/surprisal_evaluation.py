# evaluation/cases/surprisal_evaluation.py

import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple

from ..eval_case import EvaluationCase
from ..model_wrapper import ModelWrapper


def find_sublist_indices(main_list: List, sub_list: List) -> Tuple[int, int]:
    """Finds the start and end indices of a sublist within a main list."""
    for i in range(len(main_list) - len(sub_list) + 1):
        if main_list[i:i + len(sub_list)] == sub_list:
            return i, i + len(sub_list)
    return -1, -1


class SurprisalEvaluation(EvaluationCase):
    """
    A concrete evaluation case for calculating surprisal on null vs. overt pronouns.
    """

    def __init__(self, model_wrapper: ModelWrapper):
        super().__init__(model_wrapper)

    def _analyze_sentence(self, context: str, target: str, hotspot: str) -> Dict:
        """Analyzes a single sentence (context + target) for surprisal."""
        full_text = f"{context.strip()} {target.strip()}"
        tokens, surprisals = self.model_wrapper.get_surprisals(full_text)

        # Tokenize hotspot separately to find its representation
        hotspot_tokens = self.model_wrapper.tokenizer.tokenize(hotspot.strip())
        start_idx, end_idx = find_sublist_indices(tokens, hotspot_tokens)

        hotspot_results = {}
        if start_idx != -1:
            hotspot_surprisals = surprisals[start_idx:end_idx]
            hotspot_results = {
                "avg_surprisal": np.mean(hotspot_surprisals).item(),
                "num_tokens": len(hotspot_surprisals),
            }

        return {
            "full_tokens": tokens,
            "full_surprisals": surprisals,
            "hotspot_analysis": hotspot_results
        }

    def run(self, data: pd.DataFrame) -> List[Dict]:
        results = []
        for _, row in tqdm(data.iterrows(), total=len(data), desc="Processing stimuli"):
            null_results = self._analyze_sentence(
                row["context"], row["null_sentence"], row["hotspot"]
            )
            overt_results = self._analyze_sentence(
                row["context"], row["overt_sentence"], row["hotspot"]
            )

            # Calculate difference score
            diff_score = None
            if "avg_surprisal" in null_results["hotspot_analysis"] and \
                    "avg_surprisal" in overt_results["hotspot_analysis"]:
                diff_score = (
                        null_results["hotspot_analysis"]["avg_surprisal"] -
                        overt_results["hotspot_analysis"]["avg_surprisal"]
                )

            results.append({
                "item_id": row["item_id"],
                "source_file": source_filename,
                "context": row["context"],
                "hotspot_text": row["hotspot"],
                "null_sentence_analysis": null_results,
                "overt_sentence_analysis": overt_results,
                "hotspot_difference_score": diff_score,
            })
        return results