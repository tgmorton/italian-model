#!/bin/bash

# --- Continue Model Training from Checkpoint Script ---
# This script is designed to be run directly on a Linux machine with a GPU,
# assuming Singularity and CUDA are pre-installed.
# It continues training for a single model from a given checkpoint.

# Exit on any error
set -e

# --- Define Paths (Constants) ---
HOST_PROJECT_DIR="/home/AD/thmorton/italian-model"
HOST_SIF_PATH="/home/AD/thmorton/italian-model/italian_llm_env.sif"

# --- Script Usage ---
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <relative_config_file_path> <absolute_checkpoint_path>"
    echo "Example: ./continue_training.sh 'configs/10M_experiment.yaml' '/home/AD/thmorton/italian-model/models/seed_sweep/10M-seed_1/checkpoint-XYZ'"
    echo "The relative_config_file_path should be relative to your HOST_PROJECT_DIR (e.g., 'configs/my_config.yaml')."
    echo "The absolute_checkpoint_path must be the full path to the specific checkpoint directory on the host."
    exit 1
fi

# --- Argument Parsing ---
CONFIG_FILE_REL_PATH=$1 # e.g., configs/10M_experiment.yaml
CHECKPOINT_FULL_PATH=$2 # e.g., /home/AD/thmorton/italian-model/models/seed_sweep/10M-seed_1/checkpoint-XYZ

# === Environment Setup ===
echo "========================================================"
echo "Script Started: $(date)"
echo "Using Config File: ${CONFIG_FILE_REL_PATH}"
echo "Continuing training from checkpoint: ${CHECKPOINT_FULL_PATH}"
echo "Note: Running directly on this machine."
echo "========================================================"

# --- Load necessary system modules ---
echo "Attempting to load system modules (ensure these are available in your environment)..."
# These module commands are typically for HPC environments.
# If running on a personal machine, you might just need to ensure Singularity and CUDA
# are in your PATH and configured correctly, or adjust these lines.
module load singularity/4.1.1 || echo "Warning: singularity/4.1.1 module not found. Ensure Singularity is in your PATH."
module load cuda/11.8 || echo "Warning: cuda/11.8 module not found. Ensure CUDA is correctly configured."

# --- Derive dynamic parameters from the checkpoint path ---
# Extract the parent directory of the checkpoint, which is the model's specific training directory (e.g., 10M-seed_1)
MODEL_TRAINING_DIR_HOST=$(dirname "${CHECKPOINT_FULL_PATH}")
MODEL_DIR_NAME_FROM_CHECKPOINT=$(basename "${MODEL_TRAINING_DIR_HOST}") # e.g., 10M-seed_1

# Derive the output directory for src.train relative to /workspace inside the container.
# This assumes the structure: /workspace/models/seed_sweep/MODEL_DIR_NAME_FROM_CHECKPOINT
OUTPUT_DIR_REL_TO_WORKSPACE="models/seed_sweep/${MODEL_DIR_NAME_FROM_CHECKPOINT}"

# Extract the seed number from the model directory name
# This relies on the convention "*-seed_NUMBER" in the directory name.
if [[ "${MODEL_DIR_NAME_FROM_CHECKPOINT}" =~ seed_([0-9]+) ]]; then
    SEED="${BASH_REMATCH[1]}"
    echo "Detected Seed: ${SEED}"
else
    echo "ERROR: Could not extract seed from model directory name: ${MODEL_DIR_NAME_FROM_CHECKPOINT}"
    echo "Expected directory name format like '...-seed_N' (e.g., 10M-seed_1)."
    exit 1
fi

# Determine the checkpoint path inside the container
# By replacing the host project directory with /workspace
CHECKPOINT_CONTAINER_PATH="${CHECKPOINT_FULL_PATH/${HOST_PROJECT_DIR}/\/workspace}"

# --- Preparations ---
echo "Project Directory (Host): ${HOST_PROJECT_DIR}"
echo "SIF Image Path (Host): ${HOST_SIF_PATH}"
echo "Config File (Relative): ${CONFIG_FILE_REL_PATH}"
echo "Derived Output Directory (relative to /workspace): ${OUTPUT_DIR_REL_TO_WORKSPACE}"
echo "Checkpoint Path (inside container): ${CHECKPOINT_CONTAINER_PATH}"

if [ ! -f "$HOST_SIF_PATH" ]; then echo "ERROR: Singularity image not found at $HOST_SIF_PATH"; exit 1; fi
if [ ! -d "${CHECKPOINT_FULL_PATH}" ]; then echo "ERROR: Checkpoint directory not found at ${CHECKPOINT_FULL_PATH}"; exit 1; fi
if [ ! -f "${HOST_PROJECT_DIR}/${CONFIG_FILE_REL_PATH}" ]; then echo "ERROR: Config file not found at ${HOST_PROJECT_DIR}/${CONFIG_FILE_REL_PATH}"; exit 1; fi

mkdir -p "${HOST_PROJECT_DIR}/logs"
# Ensure the output directory for the current model exists on the host
mkdir -p "${HOST_PROJECT_DIR}/${OUTPUT_DIR_REL_TO_WORKSPACE}"


# --- Training Script Execution ---
echo "Starting Python training script inside Singularity container..."
echo "Using config file: ${CONFIG_FILE_REL_PATH}"

# Set PyTorch CUDA Allocator Config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Execute the training script inside the container, passing the derived parameters
singularity exec --nv \
    --bind "${HOST_PROJECT_DIR}":/workspace \
    "${HOST_SIF_PATH}" \
    bash -c "cd /workspace && python -m src.train \
        --config-file ${CONFIG_FILE_REL_PATH} \
        --seed ${SEED} \
        --output-dir ${OUTPUT_DIR_REL_TO_WORKSPACE} \
        --checkpoint_path \"${CHECKPOINT_CONTAINER_PATH}\""

echo "========================================================"
echo "Script Finished: $(date)"
echo "Training for model from checkpoint ${CHECKPOINT_FULL_PATH} has completed."
echo "========================================================"
