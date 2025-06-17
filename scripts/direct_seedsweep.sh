#!/bin/bash

# --- Direct Execution Script for Model Training Seed Sweep ---
# This script is designed to be run directly on a Linux machine with a GPU,
# assuming Singularity and CUDA are pre-installed.
# It executes a range of seeds for the model training script.

# Exit on any error
set -e

# --- Script Usage ---
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <start_seed> <end_seed>"
    echo "Example: ./direct_seed_sweep.sh 1 9"
    echo "This will run seeds from 1 to 9 (inclusive)."
    exit 1
fi

# --- Argument Parsing ---
START_SEED=$1
END_SEED=$2

# === Environment Setup ===
echo "========================================================"
echo "Script Started: $(date)"
echo "Running Seeds from ${START_SEED} to ${END_SEED}"
echo "Note: Running directly on this machine."
echo "========================================================"

# --- Load necessary system modules ---
echo "Attempting to load system modules (ensure these are available in your environment)..."
# These module commands are typically for HPC environments.
# If running on a personal machine, you might just need to ensure Singularity and CUDA
# are in your PATH and configured correctly, or adjust these lines.
module load singularity/4.1.1 || echo "Warning: singularity/4.1.1 module not found. Ensure Singularity is in your PATH."
module load cuda/11.8 || echo "Warning: cuda/11.8 module not found. Ensure CUDA is correctly configured."

# --- Define Paths ---
HOST_PROJECT_DIR="/home/AD/thmorton/italian-model"
HOST_SIF_PATH="/home/AD/thmorton/italian-model/italian_llm_env.sif"

# --- Define the experiment config file ---
CONFIG_FILE="configs/10M_experiment.yaml"

# --- Preparations ---
echo "Project Directory (Host): ${HOST_PROJECT_DIR}"
echo "SIF Image Path (Host): ${HOST_SIF_PATH}"
echo "Config File: ${CONFIG_FILE}"
if [ ! -f "$HOST_SIF_PATH" ]; then echo "ERROR: Singularity image not found at $HOST_SIF_PATH"; exit 1; fi
mkdir -p "${HOST_PROJECT_DIR}/logs"

# --- Training Script Execution Loop ---
# Loop through the specified range of seeds
for SEED in $(seq "${START_SEED}" "${END_SEED}"); do
    echo "--------------------------------------------------------"
    echo "Processing Seed: ${SEED}"

    # --- Define dynamic parameters for the current seed ---
    OUTPUT_DIR="models/seed_sweep/10M-seed_${SEED}"
    echo "Output Directory for Seed ${SEED}: ${OUTPUT_DIR}"

    # Ensure the output directory for the current seed exists
    mkdir -p "${HOST_PROJECT_DIR}/${OUTPUT_DIR}"

    echo "Starting Python training script inside Singularity container for Seed ${SEED}..."
    echo "Using config file: ${CONFIG_FILE}"

    # Set PyTorch CUDA Allocator Config (can be outside loop if consistent, but safer inside for clarity)
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    # Execute the training script inside the container, overriding parameters
    singularity exec --nv \
        --bind "${HOST_PROJECT_DIR}":/workspace \
        "${HOST_SIF_PATH}" \
        bash -c "cd /workspace && python -m src.train \
            --config-file ${CONFIG_FILE} \
            --seed ${SEED} \
            --output-dir ${OUTPUT_DIR}"

    echo "Finished processing Seed: ${SEED}"
done

echo "========================================================"
echo "Script Finished: $(date)"
echo "All specified seeds have been processed."
echo "========================================================"
