#!/bin/bash

# --- Direct Execution Script for Model Training Seed Sweep ---
# This script is designed to be run directly on a Linux machine with a GPU,
# assuming Singularity and CUDA are pre-installed.
# It executes a single seed of the model training script.

# Exit on any error
set -e

# --- Script Usage ---
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <seed_number>"
    echo "Example: ./direct_seed_sweep.sh 1"
    echo "Example: ./direct_seed_sweep.sh 5"
    exit 1
fi

# --- Argument Parsing ---
SEED=$1

# === Environment Setup ===
echo "========================================================"
echo "Script Started: $(date)"
echo "Running Seed: ${SEED}"
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

# --- Define dynamic parameters based on the provided SEED ---
OUTPUT_DIR="models/seed_sweep/10M-seed_${SEED}"

# --- Preparations ---
echo "Project Directory (Host): ${HOST_PROJECT_DIR}"
echo "SIF Image Path (Host): ${HOST_SIF_PATH}"
echo "Config File: ${CONFIG_FILE}"
echo "Output Directory: ${OUTPUT_DIR}"
if [ ! -f "$HOST_SIF_PATH" ]; then echo "ERROR: Singularity image not found at $HOST_SIF_PATH"; exit 1; fi
mkdir -p "${HOST_PROJECT_DIR}/logs"

# --- Training Script Execution ---
echo "Starting Python training script inside Singularity container..."
echo "Using config file: ${CONFIG_FILE}"

# Set PyTorch CUDA Allocator Config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Execute the training script inside the container, overriding parameters
singularity exec --nv \
    --bind "${HOST_PROJECT_DIR}":/workspace \
    "${HOST_SIF_PATH}" \
    bash -c "cd /workspace && python -m src.train \
        --config-file ${CONFIG_FILE} \
        --seed ${SEED} \
        --output-dir ${OUTPUT_DIR}"

echo "========================================================"
echo "Script Finished: $(date)"
echo "========================================================"
