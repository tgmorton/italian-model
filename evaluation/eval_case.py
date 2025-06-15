# evaluation/eval_case.py

import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict

from .model_wrapper import ModelWrapper

class EvaluationCase(ABC):
    """
    Abstract base class for an evaluation case.
    """
    def __init__(self, model_wrapper: ModelWrapper):
        self.model_wrapper = model_wrapper

    @abstractmethod
    def run(self, data: pd.DataFrame) -> List[Dict]:
        """
        Runs the evaluation on the provided data.

        Args:
            data (pd.DataFrame): The data loaded by the DataLoader.

        Returns:
            List[Dict]: A list of dictionaries, where each dictionary
                        contains the results for a single item.
        """
        pass