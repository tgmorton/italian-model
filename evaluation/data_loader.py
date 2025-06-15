# evaluation/data_loader.py

import pandas as pd
from pathlib import Path

class DataLoader:
    """
    Handles loading and validation of evaluation data from CSV files.
    It expects specific columns for the Italian stimuli and renames
    them to generic names for use in the evaluation pipeline.
    """
    # The required columns from the source CSV file
    REQUIRED_COLUMNS = [
        "item",
        "c_italian",
        "t_null_italian",
        "t_overt_italian",
        "hotspot_italian",
    ]

    # Mapping from the source column names to the internal, generic names
    COLUMN_MAPPING = {
        "item": "item_id",
        "c_italian": "context",
        "t_null_italian": "null_sentence",
        "t_overt_italian": "overt_sentence",
        "hotspot_italian": "hotspot",
    }

    def __init__(self, file_path: Path):
        """
        Initializes the DataLoader with the path to the evaluation file.

        Args:
            file_path (Path): The path to the .csv evaluation file.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Evaluation file not found at: {file_path}")
        self.file_path = file_path
        self.data = None

    def _validate_columns(self) -> None:
        """
        Validates that the loaded DataFrame contains the required source columns.
        """
        missing_cols = [
            col for col in self.REQUIRED_COLUMNS if col not in self.data.columns
        ]
        if missing_cols:
            raise ValueError(
                f"Evaluation file '{self.file_path.name}' is missing required columns: {missing_cols}"
            )

    def load_data(self) -> pd.DataFrame:
        """
        Loads the CSV, validates it, and returns a DataFrame with standardized column names.

        Returns:
            pd.DataFrame: A DataFrame with columns renamed according to COLUMN_MAPPING.
        """
        self.data = pd.read_csv(self.file_path)
        self._validate_columns()

        # Keep only the required columns and rename them
        relevant_data = self.data[self.REQUIRED_COLUMNS].copy()
        renamed_data = relevant_data.rename(columns=self.COLUMN_MAPPING)

        print(f"Successfully loaded and validated '{self.file_path.name}'.")
        return renamed_data