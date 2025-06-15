# evaluation/cases/surprisal_evaluation.py

import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict

from ..eval_case import EvaluationCase
from ..model_wrapper import ModelWrapper


class SurprisalEvaluation(EvaluationCase):
    """
    A concrete evaluation case for calculating surprisal on null vs. overt pronouns.
    """

    def __init__(self, model_wrapper: ModelWrapper):
        super().__init__(model_wrapper)

    def _analyze_sentence(self, context: str, target: str, hotspot: str) -> Dict:
        """Analyzes a single sentence (context + target) for surprisal."""
        clean_context = context.strip()
        clean_target = target.strip()

        full_text = f"{clean_context} {clean_target}"
        tokens, surprisals, offset_mapping = self.model_wrapper.get_surprisals(full_text)

        # CORRECTED: Search for the stripped version of the hotspot to robustly find its position.
        stripped_hotspot = hotspot.strip()
        hotspot_char_start_in_target = clean_target.find(stripped_hotspot)

        hotspot_indices = []
        hotspot_results = {}

        if hotspot_char_start_in_target != -1:
            context_len = len(clean_context) + 1
            hotspot_char_start_in_full = context_len + hotspot_char_start_in_target
            # CORRECTED: Use the length of the stripped hotspot for calculating the end.
            hotspot_char_end_in_full = hotspot_char_start_in_full + len(stripped_hotspot)

            for i, (token_start, token_end) in enumerate(offset_mapping):
                if token_end > hotspot_char_start_in_full and token_start < hotspot_char_end_in_full:
                    hotspot_indices.append(i)

            if hotspot_indices:
                hotspot_surprisals = [surprisals[i] for i in hotspot_indices]
                hotspot_results = {
                    "avg_surprisal": np.mean(hotspot_surprisals).item(),
                    "num_tokens": len(hotspot_surprisals),
                    "tokens": [tokens[i] for i in hotspot_indices]
                }

        return {
            "full_tokens": tokens,
            "full_surprisals": surprisals,
            "hotspot_analysis": hotspot_results
        }

    def run(self, data: pd.DataFrame, source_filename: str = "unknown") -> List[Dict]:
        # ... (run method is unchanged) ...
        results = []
        for _, row in tqdm(data.iterrows(), total=len(data), desc=f"Processing {source_filename}"):
            null_results = self._analyze_sentence(
                row["context"], row["null_sentence"], row["hotspot"]
            )
            overt_results = self._analyze_sentence(
                row["context"], row["overt_sentence"], row["hotspot"]
            )

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