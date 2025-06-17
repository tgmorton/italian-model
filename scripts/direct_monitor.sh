#!/bin/bash

# --- Direct Execution Script for Titan-X with Tokenizer Argument ---
# This script is designed to be run directly on a Linux machine with a Titan-X GPU,
# assuming Singularity and CUDA are pre-installed.
# It mimics the execution logic of your SBATCH script but without Slurm directives.

# Exit on error
set -e

# --- Script Usage ---
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <model_dir_name> <tokenizer_name> [perplexity_eval_portion]"
    echo "Example (full run): ./run_monitor_titan_x_tokenizer.sh 10M_10epoch 10M"
    echo "Example (33% of perplexity test set): ./run_monitor_titan_x_tokenizer.sh 10M_10epoch 10M 0.33"
    exit 1
fi

# --- Argument Parsing ---
MODEL_DIR_NAME=$1
TOKENIZER_NAME=$2
PERPLEXITY_PORTION=$3


# === Environment Setup ===
echo "=== Script Started: $(date) ==="
echo "Target Model Directory: ${MODEL_DIR_NAME}"
echo "Using Tokenizer: ${TOKENIZER_NAME}"
echo "Note: Running directly on this machine."

# --- Load necessary system modules ---
echo "Attempting to load system modules (ensure these are available in your environment)..."
# These module commands are typically for HPC environments.
# If running on a personal machine, you might just need to ensure Singularity and CUDA
# are in your PATH and configured correctly, or adjust these lines.
module load singularity/4.1.1 || echo "Warning: singularity/4.1.1 module not found. Ensure Singularity is in your PATH."
module load cuda/11.8 || echo "Warning: cuda/11.8 module not found. Ensure CUDA is correctly configured."

# --- Define Host and Container Paths ---
HOST_PROJECT_DIR="/home/AD/thmorton/italian-model"
HOST_SIF_PATH="${HOST_PROJECT_DIR}/italian_llm_env.sif"
HOST_DATA_DIR="${HOST_PROJECT_DIR}/data"
HOST_MODELS_DIR="${HOST_PROJECT_DIR}/models"
HOST_RESULTS_DIR="${HOST_PROJECT_DIR}/results"
HOST_LOGS_DIR="${HOST_PROJECT_DIR}/logs"
HOST_TOKENIZER_DIR="${HOST_PROJECT_DIR}/tokenizer"
HOST_SURPRISAL_DIR="${HOST_PROJECT_DIR}/evaluation/data/italian_null_subject"

CONTAINER_WORKSPACE="/workspace"
CONTAINER_DATA_DIR="/data"
CONTAINER_MODELS_DIR="/models"
CONTAINER_RESULTS_DIR="/results"
CONTAINER_TOKENIZER_DIR="/workspace/tokenizer"
CONTAINER_SURPRISAL_DIR="/surprisal_data"

# --- Process Optional Argument ---
PERPLEXITY_ARG=""
if [ -n "$PERPLEXITY_PORTION" ]; then
    PERPLEXITY_ARG="--perplexity_eval_portion ${PERPLEXITY_PORTION}"
fi

# --- Preparations ---
echo "Project Directory (Host): ${HOST_PROJECT_DIR}"
# Check for user-specified model and tokenizer directories on the host
if [ ! -d "${HOST_MODELS_DIR}/${MODEL_DIR_NAME}" ]; then echo "ERROR: Target model directory not found at ${HOST_MODELS_DIR}/${MODEL_DIR_NAME}"; exit 1; fi
if [ ! -d "${HOST_TOKENIZER_DIR}/${TOKENIZER_NAME}" ]; then echo "ERROR: Tokenizer directory not found at ${HOST_TOKENIZER_DIR}/${TOKENIZER_NAME}"; exit 1; fi
if [ ! -d "${HOST_SURPRISAL_DIR}" ]; then echo "ERROR: Surprisal data directory not found at ${HOST_SURPRISAL_DIR}"; exit 1; fi
mkdir -p "${HOST_RESULTS_DIR}"
mkdir -p "${HOST_LOGS_DIR}"

# === Monitor Script Execution ===
echo "Starting Python monitor.py script inside Singularity container..."

# Execute the Python script within the Singularity container, leveraging the GPU (--nv)
# and binding the necessary host directories to their container counterparts.
# The critical bind mount for the tokenizer is now:
# -B "${HOST_TOKENIZER_DIR}/${TOKENIZER_NAME}":"${CONTAINER_TOKENIZER_DIR}/${TOKENIZER_NAME}"
# This maps the specific host tokenizer directory (e.g., /home/.../tokenizer/10M)
# to the path inside the container that the python script now explicitly expects (e.g., /workspace/tokenizer/10M).
singularity exec --nv \
    -B "${HOST_PROJECT_DIR}":"${CONTAINER_WORKSPACE}" \
    -B "${HOST_DATA_DIR}":"${CONTAINER_DATA_DIR}" \
    -B "${HOST_MODELS_DIR}":"${CONTAINER_MODELS_DIR}" \
    -B "${HOST_RESULTS_DIR}":"${CONTAINER_RESULTS_DIR}" \
    -B "${HOST_SURPRISAL_DIR}":"${CONTAINER_SURPRISAL_DIR}" \
    -B "${HOST_TOKENIZER_DIR}":"${CONTAINER_TOKENIZER_DIR}" \
    -B "${HOST_TOKENIZER_DIR}/${TOKENIZER_NAME}":"${CONTAINER_TOKENIZER_DIR}/${TOKENIZER_NAME}" \
    "${HOST_SIF_PATH}" \
    bash -c "cd ${CONTAINER_WORKSPACE} && python3 -m evaluation.monitor \
        --model_parent_dir \"${CONTAINER_MODELS_DIR}/${MODEL_DIR_NAME}\" \
        --output_base_dir \"${CONTAINER_RESULTS_DIR}\" \
        --tokenizer_base_dir \"${CONTAINER_TOKENIZER_DIR}\" \
        --tokenizer_name \"${TOKENIZER_NAME}\" \
        --surprisal_data_dir \"${CONTAINER_SURPRISAL_DIR}\" \
        --perplexity_data_base_path \"${CONTAINER_DATA_DIR}/tokenized\" \
        ${PERPLEXITY_ARG}"

# === Script Completion ===
echo "=== Script Finished: $(date) ==="
