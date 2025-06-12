# src/analysis/generate_aligned_schedule.py

import math
from pathlib import Path
from typing import Dict, List, Set

import typer
import yaml
from datasets import load_from_disk
from pydantic import BaseModel, PositiveInt, field_validator
from transformers import AutoTokenizer  # Added to load tokenizer for chunk size


# --- Configuration Models ---

class ScheduleConfig(BaseModel):
    per_device_train_batch_size: PositiveInt
    gradient_accumulation_steps: PositiveInt
    num_epochs: PositiveInt
    target_checkpoints: Dict[str, int]

    @field_validator("target_checkpoints", mode="before")
    @classmethod
    def parse_target_string(cls, v: str) -> Dict[str, int]:
        if not isinstance(v, str): return v
        try:
            return {item.split(':')[0]: int(item.split(':')[1]) for item in v.split(',')}
        except (ValueError, IndexError):
            raise ValueError("Invalid format. Use '10M:20,25M:45,...'")


# --- Core Generation Function ---

def generate_checkpoint_list(
        mandatory_steps: List[int], total_steps: int, target_checkpoints: int
) -> List[int]:
    """
    Generates a list of checkpoints by layering log-steps on top of mandatory
    steps, and then filling the remaining gaps proportionally.
    """
    key_points: Set[int] = set(mandatory_steps) | {total_steps}
    first_milestone = min((s for s in key_points if s > 1), default=total_steps)
    log_step = 1
    while log_step < first_milestone:
        key_points.add(log_step)
        log_step *= 2

    if len(key_points) >= target_checkpoints:
        return sorted(list(key_points))

    num_to_add = target_checkpoints - len(key_points)
    sorted_keys = sorted(list(key_points))

    intervals = [{"start": sorted_keys[i], "end": sorted_keys[i + 1], "length": sorted_keys[i + 1] - sorted_keys[i]} for
                 i in range(len(sorted_keys) - 1) if sorted_keys[i + 1] > sorted_keys[i]]
    total_interval_length = sum(iv['length'] for iv in intervals)

    if total_interval_length == 0:
        return sorted_keys

    points_to_distribute = num_to_add
    checkpoints_in_interval_float = []
    for iv in intervals:
        proportion = iv['length'] / total_interval_length if total_interval_length > 0 else 0
        checkpoints_in_interval_float.append(proportion * points_to_distribute)

    num_in_each_interval = [int(n) for n in checkpoints_in_interval_float]
    remainders = [f - i for f, i in zip(checkpoints_in_interval_float, num_in_each_interval)]

    num_to_distribute_remainder = num_to_add - sum(num_in_each_interval)
    remainder_indices = sorted(range(len(remainders)), key=lambda k: remainders[k], reverse=True)

    for i in range(num_to_distribute_remainder):
        num_in_each_interval[remainder_indices[i]] += 1

    for i, iv in enumerate(intervals):
        num_in_interval = num_in_each_interval[i]
        if num_in_interval > 0:
            step_size = (iv['end'] - iv['start']) / (num_in_interval + 1)
            for j in range(1, num_in_interval + 1):
                new_point = round(iv['start'] + j * step_size)
                if new_point > iv['start'] and new_point < iv['end']:
                    key_points.add(new_point)

    return sorted(list(key_points))


# --- Typer App for CLI ---

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def main(
        base_data_dir: Path = typer.Option("/Users/thomasmorton/Italian-Model/data/tokenized", help="Base directory for tokenized datasets."),
        base_tokenizer_dir: Path = typer.Option("/Users/thomasmorton/Italian-Model/tokenizer", help="Base directory for tokenizers."),
        per_device_train_batch_size: int = typer.Option(8, help="Batch size per device."),
        gradient_accumulation_steps: int = typer.Option(16, help="Gradient accumulation steps."),
        num_epochs: int = typer.Option(1, help="Total training epochs to simulate."),
        target_checkpoints: str = typer.Option(
            "10M:40,25M:100,50M:100,100M:200",
            help="Comma-separated list of target checkpoint counts per size."
        ),
):
    """
    Generates aligned, cascading checkpoint schedules for each dataset size,
    accurately simulating the dataloader's chunking logic.
    """
    config = ScheduleConfig(**locals())
    all_sizes = ['10M', '25M', '50M', '100M']
    milestone_steps: Dict[str, int] = {}
    effective_batch_size = config.per_device_train_batch_size * config.gradient_accumulation_steps

    print("--- Step 1: Calculating Milestone Step Counts (with 100% accurate chunking) ---")

    for size in all_sizes:
        dataset_path = base_data_dir / size / "train"
        tokenizer_path = base_tokenizer_dir / size

        if not dataset_path.exists() or not tokenizer_path.exists():
            print(f"- Skipping {size} (data or tokenizer not found).")
            continue

        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        chunk_size = tokenizer.model_max_length
        if not chunk_size: chunk_size = 1024  # Fallback

        dataset = load_from_disk(str(dataset_path))

        # --- NEW: Accurately simulate the chunking logic from data.py ---
        num_chunks = sum(math.ceil(len(ids) / chunk_size) for ids in dataset['input_ids'])

        steps_per_epoch = math.ceil(num_chunks / effective_batch_size)
        milestone_steps[size] = steps_per_epoch * config.num_epochs
        print(f"  - Milestone for {size}: {milestone_steps[size]:,} steps (from {num_chunks:,} chunks)")

    print("\n--- Step 2: Generating Cascading Layered Schedules ---")

    previous_schedule: List[int] = []
    for current_run_size in all_sizes:
        if current_run_size not in milestone_steps: continue

        total_steps_for_this_run = milestone_steps[current_run_size]
        target_count = config.target_checkpoints.get(current_run_size, 20)
        mandatory = set(previous_schedule) | {total_steps_for_this_run}

        schedule = generate_checkpoint_list(sorted(list(mandatory)), total_steps_for_this_run, target_count)

        print(f"\n--- Recommended Schedule for {current_run_size} Training Run ---")
        print(f"  Targeting ~{target_count} checkpoints. Generated {len(schedule)} total.")
        print(f"  This schedule PRESERVES the {len(previous_schedule)} points from the previous schedule.")
        print("\n  >>> Generated Checkpoint List (for Python):")
        print(f"  {schedule}")
        print("\n  >>> YAML format (for config file):")
        yaml_dict = {"checkpoint_schedule": schedule}
        yaml_output = yaml.dump(yaml_dict, indent=2, default_flow_style=False)
        indented_yaml_output = "\n".join(["    " + line for line in yaml_output.splitlines()])
        print(indented_yaml_output)

        previous_schedule = schedule


if __name__ == "__main__":
    app()