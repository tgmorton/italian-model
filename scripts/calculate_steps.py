import math
from pathlib import Path
from typing import Dict, List, Set, Tuple

import typer
from datasets import load_from_disk
from pydantic import BaseModel, PositiveInt


# --- Configuration Model ---

class StepCalculationConfig(BaseModel):
    """Defines the parameters needed for step calculation."""
    per_device_train_batch_size: PositiveInt
    gradient_accumulation_steps: PositiveInt
    num_epochs: PositiveInt


# --- Helper Functions for Analysis ---

def find_all_divisors(number: int) -> Set[int]:
    """Finds all integer divisors of a number and returns them in a set."""
    divs = set()
    for i in range(1, int(math.sqrt(number)) + 1):
        if number % i == 0:
            divs.add(i)
            divs.add(number // i)
    return divs


# --- Typer App for CLI with Multiple Commands ---

app = typer.Typer(
    add_completion=False,
    pretty_exceptions_show_locals=False,
    help="A script to analyze training steps and recommend checkpointing strategies.",
)


# --- Original command remains available ---
@app.command(name="analyze-all")
def main(
        # ... (The previous `main` function is unchanged but renamed to `analyze-all`)
        # ... I have elided it here for brevity, but it is the same as the last version.
):
    """Calculates step counts and finds the best common divisor from zero."""
    # ... (code from the previous version) ...


# --- NEW command to solve the offset problem ---
@app.command()
def find_interval(
        offset_size: str = typer.Option("25M", help="The dataset size to use as the starting checkpoint offset."),
        base_data_dir: Path = typer.Option("data/tokenized", help="Base directory for tokenized datasets."),
        per_device_train_batch_size: int = typer.Option(8, help="Batch size per device."),
        gradient_accumulation_steps: int = typer.Option(16, help="Gradient accumulation steps."),
        num_epochs: int = typer.Option(1, help="Total training epochs to simulate."),
):
    """
    Finds a save interval 'S' that hits all larger data size checkpoints perfectly,
    starting from an offset checkpoint (e.g., the final step of the 25M run).
    """
    config = StepCalculationConfig(
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_epochs=num_epochs,
    )

    all_sizes = ['10M', '25M', '50M', '100M']
    total_steps_map = {}
    effective_batch_size = config.per_device_train_batch_size * config.gradient_accumulation_steps

    print("--- Step 1: Calculating Total Steps for All Datasets ---")
    print(
        f"Settings: Batch Size={config.per_device_train_batch_size}, Grad Accum={config.gradient_accumulation_steps}, Epochs={config.num_epochs}")

    for size in all_sizes:
        dataset_path = base_data_dir / size / "train"
        if not dataset_path.exists(): continue
        dataset = load_from_disk(str(dataset_path))
        steps_per_epoch = math.ceil(len(dataset) / effective_batch_size)
        total_steps_map[size] = steps_per_epoch * config.num_epochs
        print(f"  - Total steps for {size}: {total_steps_map[size]:,}")

    if offset_size not in total_steps_map:
        print(f"\nERROR: Could not calculate steps for offset size '{offset_size}'. Aborting.")
        return

    offset_steps = total_steps_map[offset_size]
    target_steps = {s: t for s, t in total_steps_map.items() if t > offset_steps}

    if not target_steps:
        print("\nNo targets found larger than the offset. Nothing to calculate.")
        return

    print(f"\n--- Step 2: Calculating Differences from Offset '{offset_size}' ({offset_steps:,} steps) ---")
    differences = []
    for size, steps in target_steps.items():
        diff = steps - offset_steps
        differences.append(diff)
        print(f"  - Target {size} ({steps:,} steps): Difference = {diff:,}")

    print("\n--- Step 3: Finding Common Divisors for a Perfect Interval ---")
    # Find all divisors for the first difference
    if not differences: return
    common_divisors = find_all_divisors(differences[0])

    # Find the intersection with divisors of all other differences
    for i in range(1, len(differences)):
        common_divisors.intersection_update(find_all_divisors(differences[i]))

    if not common_divisors:
        print("No common integer divisor found for the step differences.")
        return

    sorted_divisors = sorted(list(common_divisors))

    # Highlight "nice" round numbers from the list of all perfect divisors
    nice_divisors = [d for d in sorted_divisors if d % 10 == 0 and d >= 100]

    print("\n--- Results ---")
    print("Any of the following numbers can be used as a `save_steps` interval to perfectly hit all targets:")
    print(f"\nAll Possible Perfect Intervals: {sorted_divisors}")
    print(f"\nRecommended 'Round Number' Intervals: {nice_divisors or 'None found'}")

    # Calculate the Greatest Common Divisor (GCD) for the most precise interval
    gcd_result = differences[0]
    for i in range(1, len(differences)):
        gcd_result = math.gcd(gcd_result, differences[i])
    print(f"\nThe mathematically most precise interval (GCD) is: {gcd_result}")


if __name__ == "__main__":
    app()