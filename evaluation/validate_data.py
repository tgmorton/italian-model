# evaluation/validate_data.py

import argparse
import sys
from pathlib import Path

# Add the parent directory to the path to allow sibling imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from evaluation.data_loader import DataLoader

def main():
    """
    Main function to run the validation script.
    """
    parser = argparse.ArgumentParser(
        description="Validate an evaluation CSV file for compatibility with the evaluation framework.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "file_path",
        type=Path,
        help="Path to the evaluation .csv file to validate."
    )

    args = parser.parse_args()

    try:
        # The DataLoader now encapsulates all the validation logic.
        loader = DataLoader(args.file_path)
        loader.load_data() # This will raise an error if validation fails.

        print("\nValidation successful!")
        print(f"File '{args.file_path.name}' has the required columns: {loader.REQUIRED_COLUMNS}")
    except (FileNotFoundError, ValueError) as e:
        print(f"\nValidation failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()