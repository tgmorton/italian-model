# evaluation/monitor.py

import argparse
from pathlib import Path
import time
import json
from tqdm import tqdm

from .model_wrapper import ModelWrapper
from .cases.surprisal_evaluation import SurprisalEvaluation
from .cases.perplexity_evaluation import PerplexityEvaluation
from .data_loader import DataLoader

# A registry of all available evaluation cases
EVAL_CASES_REGISTRY = {
    'surprisal': SurprisalEvaluation,
    'perplexity': PerplexityEvaluation,
}


def main():
    parser = argparse.ArgumentParser(description="Finds and evaluates model checkpoints.")
    parser.add_argument("--model_parent_dir", type=Path, required=True,
                        help="Path to the model size directory (e.g., /models/10M).")
    parser.add_argument("--output_base_dir", type=Path, required=True,
                        help="Base directory to save all evaluation results.")

    # MODIFIED: Now expects a directory for surprisal data
    parser.add_argument("--surprisal_data_dir", type=Path,
                        help="Path to the directory with CSV files for surprisal evaluation.")
    parser.add_argument("--perplexity_data_base_path", type=Path,
                        help="Path to the base dir for tokenized data (e.g., /data/tokenized).")

    # Optional: run only specific cases
    parser.add_argument(
        "--eval_cases", type=str, nargs='*', default=list(EVAL_CASES_REGISTRY.keys()),
        help=f"Space-separated list of cases to run. Default: all. Available: {list(EVAL_CASES_REGISTRY.keys())}"
    )
    args = parser.parse_args()

    print(f"Starting evaluation monitor.")
    print(f"Searching for model checkpoints under: {args.model_parent_dir}")

    # Find all checkpoint directories within the specific model_parent_dir
    all_checkpoints = sorted(list(args.model_parent_dir.glob("checkpoint-*")))
    print(f"Found {len(all_checkpoints)} total checkpoints to evaluate.")

    if not all_checkpoints:
        print("No checkpoints found. Exiting.")
        return

    # Process each checkpoint sequentially
    for checkpoint_path in all_checkpoints:
        try:
            print(f"\n{'=' * 80}")
            print(f"Processing checkpoint: {checkpoint_path}")

            # Infer model size and tokenizer path from checkpoint path
            # e.g., .../models/10M/checkpoint-5000 -> tokenizer is in .../tokenizer/10M
            model_size_tag = checkpoint_path.parent.name
            # The tokenizer directory is expected to be parallel to the models directory
            tokenizer_path = checkpoint_path.parents[1] / "tokenizer" / model_size_tag

            if not tokenizer_path.exists():
                print(f"  [ERROR] Tokenizer not found at expected path: {tokenizer_path}. Skipping checkpoint.")
                continue

            # --- Load Model (once per checkpoint) ---
            model_wrapper = ModelWrapper(checkpoint_path, tokenizer_path)
            output_dir = args.output_base_dir / model_size_tag
            output_dir.mkdir(parents=True, exist_ok=True)

            # --- Loop through and run selected evaluation cases ---
            for case_name in args.eval_cases:
                evaluator_class = EVAL_CASES_REGISTRY.get(case_name)
                print(f"\n--- Running evaluation case: {case_name} ---")
                evaluator = evaluator_class(model_wrapper)
                results = []
                data_name_for_filename = "unknown"

                try:
                    if case_name == 'surprisal':
                        if not args.surprisal_data_dir:
                            raise ValueError("Surprisal evaluation requires --surprisal_data_dir.")

                        # Find all CSV files in the specified directory
                        surprisal_files = sorted(list(args.surprisal_data_dir.glob('*.csv')))
                        if not surprisal_files:
                            raise FileNotFoundError(f"No .csv files found in {args.surprisal_data_dir}")

                        print(f"Found {len(surprisal_files)} surprisal files to process.")

                        all_surprisal_results = []
                        for csv_file in surprisal_files:
                            data_loader = DataLoader(csv_file)
                            eval_data = data_loader.load_data()
                            current_results = evaluator.run(eval_data, source_filename=csv_file.name)
                            all_surprisal_results.extend(current_results)

                        results = all_surprisal_results
                        data_name_for_filename = "surprisal_stimuli"

                    elif case_name == 'perplexity':
                        if not args.perplexity_data_base_path:
                            raise ValueError("Perplexity evaluation requires --perplexity_data_base_path.")
                        perplexity_data_path = args.perplexity_data_base_path / model_size_tag / "test"
                        if not perplexity_data_path.exists():
                            raise FileNotFoundError(
                                f"Perplexity data not found for {model_size_tag} at {perplexity_data_path}")
                        results = evaluator.run(data_path=perplexity_data_path)
                        data_name_for_filename = perplexity_data_path.name

                    # Save results for the current case
                    output_filename = f"{checkpoint_path.name}_{data_name_for_filename}_{case_name}_results.json"
                    output_path = output_dir / output_filename

                    print(f"Saving {case_name} results to {output_path}...")
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(results, f, ensure_ascii=False, indent=4)

                except Exception as e:
                    print(f"  [ERROR] running case '{case_name}' for checkpoint {checkpoint_path.name}: {e}")
                    continue

        except Exception as e:
            print(f"  [FATAL ERROR] processing checkpoint {checkpoint_path}. Moving to next. Error: {e}")
            continue

    print(f"\n{'=' * 80}")
    print("Evaluation monitor has completed processing all found checkpoints.")


if __name__ == "__main__":
    main()