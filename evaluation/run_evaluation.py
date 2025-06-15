# run_evaluation.py

import argparse
import json
from pathlib import Path

from evaluation.data_loader import DataLoader
from evaluation.model_wrapper import ModelWrapper
from evaluation.cases.surprisal_evaluation import SurprisalEvaluation
from evaluation.cases.perplexity_evaluation import PerplexityEvaluation

# A registry of all available evaluation cases
EVAL_CASES_REGISTRY = {
    'surprisal': SurprisalEvaluation,
    'perplexity': PerplexityEvaluation,
}


def main():
    parser = argparse.ArgumentParser(description="Run evaluation cases on a model checkpoint.")
    parser.add_argument("--checkpoint_path", type=Path, required=True, help="Path to the model checkpoint directory.")
    parser.add_argument("--tokenizer_path", type=Path, required=True, help="Path to the tokenizer directory.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to save the results JSON file.")

    # Argument to select specific cases. If not provided, all cases from the registry will run.
    parser.add_argument(
        "--eval_cases",
        type=str,
        nargs='*',  # Expect 0 or more case names
        default=list(EVAL_CASES_REGISTRY.keys()),  # Default to running all cases
        help=f"Space-separated list of evaluation cases to run. Available: {list(EVAL_CASES_REGISTRY.keys())}. If not provided, all cases are run."
    )

    # Specific data paths for each evaluation case
    parser.add_argument("--surprisal_data_path", type=Path, help="Path to the CSV file for surprisal evaluation.")
    parser.add_argument("--perplexity_data_path", type=Path,
                        help="Path to the tokenized dataset directory for perplexity evaluation.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="For perplexity, max number of samples to evaluate on.")

    args = parser.parse_args()

    # --- 1. Load Model (once per checkpoint) ---
    print(f"Loading model and tokenizer for checkpoint: {args.checkpoint_path.name}...")
    model_wrapper = ModelWrapper(args.checkpoint_path, args.tokenizer_path)

    # --- 2. Loop Through and Run Selected Evaluation Cases ---
    for case_name in args.eval_cases:
        evaluator_class = EVAL_CASES_REGISTRY.get(case_name)
        if not evaluator_class:
            print(f"Warning: Unknown evaluation case '{case_name}'. Skipping.")
            continue

        print(f"\n--- Running evaluation case: {case_name} ---")
        evaluator = evaluator_class(model_wrapper)
        results = []
        data_name_for_filename = "unknown"

        try:
            if case_name == 'surprisal':
                if not args.surprisal_data_path:
                    raise ValueError("Surprisal evaluation requires --surprisal_data_path.")
                data_loader = DataLoader(args.surprisal_data_path)
                eval_data = data_loader.load_data()
                results = evaluator.run(eval_data)
                data_name_for_filename = args.surprisal_data_path.stem

            elif case_name == 'perplexity':
                if not args.perplexity_data_path:
                    raise ValueError("Perplexity evaluation requires --perplexity_data_path.")
                results = evaluator.run(data_path=args.perplexity_data_path, max_samples=args.max_samples)
                data_name_for_filename = args.perplexity_data_path.name

            # --- 3. Save Results for the current case ---
            output_filename = f"{args.checkpoint_path.name}_{data_name_for_filename}_{case_name}_results.json"
            output_path = args.output_dir / output_filename
            args.output_dir.mkdir(exist_ok=True, parents=True)

            print(f"Saving {case_name} results to {output_path}...")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"Error running evaluation case '{case_name}': {e}")
            continue

    print("\nAll specified evaluations complete for this checkpoint.")


if __name__ == "__main__":
    main()