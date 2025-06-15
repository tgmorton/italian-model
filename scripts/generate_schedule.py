# src/analysis/generate_aligned_schedule.py

import math
from pathlib import Path
from typing import Dict, List, Set

import typer
import yaml
from datasets import load_from_disk
from pydantic import BaseModel, field_validator
from transformers import AutoTokenizer


# --- Configuration Models ---

class ScheduleConfig(BaseModel):
    target_checkpoints: Dict[str, int]

    @field_validator("target_checkpoints", mode="before")
    @classmethod
    def parse_target_string(cls, v: str) -> Dict[str, int]:
        if not isinstance(v, str):
            return v
        try:
            return {item.split(':')[0]: int(item.split(':')[1]) for item in v.split(',')}
        except (ValueError, IndexError):
            raise ValueError("Invalid format for target_checkpoints. Use '10M:20,25M:45,...'")


# --- Core Generation Function ---

def generate_checkpoint_list(
        mandatory_steps: List[int], total_steps: int, target_checkpoints: int
) -> List[int]:
    """
    Generates a list of checkpoints by layering log-steps on top of mandatory
    steps, and then filling the remaining gaps proportionally.
    """
    # Ensure total_steps is included in the key points
    key_points: Set[int] = set(mandatory_steps) | {total_steps}

    # Add powers of 2 up to the first milestone for dense early checkpoints
    first_milestone = min((s for s in key_points if s > 1), default=total_steps)
    log_step = 1
    while log_step < first_milestone:
        key_points.add(log_step)
        log_step *= 2

    # If we already have enough points, we're done
    if len(key_points) >= target_checkpoints:
        return sorted(list(key_points))

    # Calculate how many points we need to add
    num_to_add = target_checkpoints - len(key_points)
    sorted_keys = sorted(list(key_points))

    # Identify the intervals between existing key points
    intervals = [
        {"start": sorted_keys[i], "end": sorted_keys[i + 1], "length": sorted_keys[i + 1] - sorted_keys[i]}
        for i in range(len(sorted_keys) - 1) if sorted_keys[i + 1] > sorted_keys[i]
    ]
    total_interval_length = sum(iv['length'] for iv in intervals)

    if total_interval_length == 0:
        return sorted_keys

    # Distribute the needed points proportionally across the intervals
    points_to_distribute = num_to_add
    checkpoints_in_interval_float = []
    for iv in intervals:
        proportion = iv['length'] / total_interval_length if total_interval_length > 0 else 0
        checkpoints_in_interval_float.append(proportion * points_to_distribute)

    # Convert float distributions to integer counts and handle remainders
    num_in_each_interval = [int(n) for n in checkpoints_in_interval_float]
    remainders = [f - i for f, i in zip(checkpoints_in_interval_float, num_in_each_interval)]
    num_to_distribute_remainder = num_to_add - sum(num_in_each_interval)
    remainder_indices = sorted(range(len(remainders)), key=lambda k: remainders[k], reverse=True)
    for i in range(num_to_distribute_remainder):
        num_in_each_interval[remainder_indices[i]] += 1

    # Add the new points within each interval
    for i, iv in enumerate(intervals):
        num_in_interval = num_in_each_interval[i]
        if num_in_interval > 0:
            step_size = (iv['end'] - iv['start']) / (num_in_interval + 1)
            for j in range(1, num_in_interval + 1):
                new_point = round(iv['start'] + j * step_size)
                if iv['start'] < new_point < iv['end']:
                    key_points.add(new_point)

    return sorted(list(key_points))


# --- Typer App for CLI ---

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def main(
        base_data_dir: Path = typer.Option("/Users/thomasmorton/Italian-Model/data/tokenized",
                                           help="Base directory for tokenized datasets."),
        base_tokenizer_dir: Path = typer.Option("/Users/thomasmorton/Italian-Model/tokenizer",
                                                help="Base directory for tokenizers."),
        batch_size: int = typer.Option(8, help="Batch size per device."),
        gradient_accumulation_steps: int = typer.Option(16, help="Gradient accumulation steps."),
        num_epochs: int = typer.Option(10, help="Total training epochs to simulate."),
        chunk_size: int = typer.Option(1024, help="The chunk size used for tokenizing."),
        target_checkpoints: str = typer.Option(
            "10M:20,25M:50,50M:100,100M:200",
            help="Comma-separated list of target checkpoint counts PER EPOCH. Format: 'SIZE:COUNT,...'"
        ),
):
    """
    Generates aligned, cascading checkpoint schedules for each dataset size,
    accurately simulating the dataloader's chunking logic.
    """
    config = ScheduleConfig(target_checkpoints=target_checkpoints)

    if batch_size <= 0 or gradient_accumulation_steps <= 0 or num_epochs <= 0 or chunk_size <= 0:
        print("Error: Batch size, accumulation steps, epochs, and chunk size must be positive integers.")
        raise typer.Exit(code=1)

    all_sizes = ['10M', '25M', '50M', '100M']
    steps_per_epoch_map: Dict[str, int] = {}
    effective_batch_size = batch_size * gradient_accumulation_steps

    print("--- Step 1: Calculating Steps per Epoch (with 100% accurate chunking) ---")

    for size in all_sizes:
        dataset_path = base_data_dir / size / "train"
        if not dataset_path.exists():
            print(f"- Skipping {size} (data not found).")
            continue

        dataset = load_from_disk(str(dataset_path))
        num_chunks = sum(math.ceil(len(ids) / chunk_size) for ids in dataset['input_ids'])
        steps_per_epoch = math.ceil(num_chunks / effective_batch_size)
        steps_per_epoch_map[size] = steps_per_epoch
        print(f"  - Steps/Epoch for {size}: {steps_per_epoch_map[size]:,} steps (from {num_chunks:,} chunks)")

    print("\n--- Step 2: Generating Cascading Layered Schedules (Epoch-by-Epoch) ---")

    # MODIFICATION: This now correctly tracks only the first-epoch pattern for cascading.
    previous_first_epoch_schedule: List[int] = []

    for current_run_size in all_sizes:
        if current_run_size not in steps_per_epoch_map:
            continue

        steps_per_epoch = steps_per_epoch_map[current_run_size]
        total_steps_for_this_run = steps_per_epoch * num_epochs
        target_count_per_epoch = config.target_checkpoints.get(current_run_size, 20)

        # 1. Generate the schedule for the FIRST epoch, seeded by the previous run's *first epoch* pattern.
        # This ensures alignment without inheriting the full multi-epoch schedule.
        mandatory_for_first_epoch = {s for s in previous_first_epoch_schedule if s <= steps_per_epoch}

        first_epoch_schedule = generate_checkpoint_list(
            sorted(list(mandatory_for_first_epoch)),
            steps_per_epoch,
            target_count_per_epoch
        )

        # 2. Extend this single-epoch schedule across all subsequent epochs.
        full_schedule_set = set()
        for epoch in range(num_epochs):
            offset = epoch * steps_per_epoch
            for step in first_epoch_schedule:
                # Avoid adding step 0 to subsequent epochs
                if step == 0 and epoch > 0:
                    continue
                full_schedule_set.add(step + offset)

        # 3. Ensure the final step of the training run is always included.
        full_schedule_set.add(total_steps_for_this_run)

        final_schedule_for_current_run = sorted(list(full_schedule_set))

        print(f"\n--- Recommended Schedule for {current_run_size} Training Run ({num_epochs} epochs) ---")
        print(
            f"  Targeting ~{target_count_per_epoch} checkpoints per epoch. Generated {len(final_schedule_for_current_run)} total.")
        print(
            f"  This schedule's pattern PRESERVES the {len(previous_first_epoch_schedule)} points from the previous run's pattern.")
        print("\n  >>> Generated Checkpoint List (for Python):")
        print(f"  {final_schedule_for_current_run}")
        print("\n  >>> YAML format (for config file):")
        yaml_dict = {"checkpoint_schedule": final_schedule_for_current_run}
        yaml_output = yaml.dump(yaml_dict, indent=2, default_flow_style=False)
        indented_yaml_output = "\n".join([f"    {line}" for line in yaml_output.splitlines()])
        print(indented_yaml_output)

        # MODIFICATION: For the next iteration, the "previous" schedule is now correctly set
        # to be the first-epoch pattern of the CURRENT run.
        previous_first_epoch_schedule = first_epoch_schedule


if __name__ == "__main__":
    app()
