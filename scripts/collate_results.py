# scripts/collate_results.py

import argparse
import json
import pandas as pd
from pathlib import Path


def process_surprisal_file(file_path: Path, checkpoint: str) -> list:
    """
    Processes a single surprisal JSON file and extracts the relevant data
    for each item into a flat list of dictionaries.
    """
    rows = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        # Get surprisal values, defaulting to None if hotspot analysis failed
        null_surprisal = item.get("null_sentence_analysis", {}).get("hotspot_analysis", {}).get("avg_surprisal")
        overt_surprisal = item.get("overt_sentence_analysis", {}).get("hotspot_analysis", {}).get("avg_surprisal")

        row = {
            "checkpoint": checkpoint,
            "item_id": item.get("item_id"),
            "source_file": item.get("source_file"),
            "hotspot_text": item.get("hotspot_text"),
            "null_hotspot_surprisal": null_surprisal,
            "overt_hotspot_surprisal": overt_surprisal,
            "difference_score": item.get("hotspot_difference_score"),
        }
        rows.append(row)
    return rows


def process_perplexity_file(file_path: Path, checkpoint: str) -> list:
    """
    Processes a single perplexity JSON file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        # Perplexity JSON is a list with a single results dictionary
        data = json.load(f)[0]

    row = {
        "checkpoint": checkpoint,
        "perplexity": data.get("perplexity"),
        "loss": data.get("loss"),
        "num_samples_processed": data.get("num_samples_processed"),
    }
    return [row]


def main():
    parser = argparse.ArgumentParser(description="Collate evaluation JSON results into CSV files for analysis.")
    parser.add_argument("model_size_tag", type=str, help="The model size tag to process (e.g., '10M').")
    parser.add_argument("--results_base_dir", type=Path, default=Path("results"),
                        help="Base directory where results are stored.")
    parser.add_argument("--output_base_dir", type=Path, default=Path("analysis"),
                        help="Directory to save the final CSV files.")
    args = parser.parse_args()

    input_dir = args.results_base_dir / args.model_size_tag
    args.output_base_dir.mkdir(exist_ok=True, parents=True)

    if not input_dir.exists():
        print(f"Error: Input directory not found at {input_dir}")
        return

    all_json_files = list(input_dir.glob("*.json"))
    if not all_json_files:
        print(f"No JSON files found in {input_dir}")
        return

    print(f"Found {len(all_json_files)} JSON files to process for model '{args.model_size_tag}'.")

    all_surprisal_data = []
    all_perplexity_data = []

    for file_path in all_json_files:
        # Extract checkpoint name from filename, e.g., "checkpoint-128"
        checkpoint_name = file_path.name.split('_')[0]

        if "_surprisal_results.json" in file_path.name:
            all_surprisal_data.extend(process_surprisal_file(file_path, checkpoint_name))
        elif "_perplexity_results.json" in file_path.name:
            all_perplexity_data.extend(process_perplexity_file(file_path, checkpoint_name))

    # --- Save Surprisal CSV ---
    if all_surprisal_data:
        df_surprisal = pd.DataFrame(all_surprisal_data)
        # Optional: Extract numeric part of checkpoint for easier sorting/plotting in R
        df_surprisal['checkpoint_step'] = df_surprisal['checkpoint'].str.extract(r'(\d+)').astype(int)

        output_path = args.output_base_dir / f"{args.model_size_tag}_surprisal_analysis.csv"
        df_surprisal.to_csv(output_path, index=False)
        print(f"Surprisal data collated for {len(df_surprisal)} items. Saved to: {output_path}")
    else:
        print("No surprisal data found to collate.")

    # --- Save Perplexity CSV ---
    if all_perplexity_data:
        df_perplexity = pd.DataFrame(all_perplexity_data)
        df_perplexity['checkpoint_step'] = df_perplexity['checkpoint'].str.extract(r'(\d+)').astype(int)
        df_perplexity = df_perplexity.sort_values('checkpoint_step')

        output_path = args.output_base_dir / f"{args.model_size_tag}_perplexity_analysis.csv"
        df_perplexity.to_csv(output_path, index=False)
        print(f"Perplexity data collated for {len(df_perplexity)} checkpoints. Saved to: {output_path}")
    else:
        print("No perplexity data found to collate.")


if __name__ == "__main__":
    main()