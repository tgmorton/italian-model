#!/bin/bash

# === SBATCH Directives ===
#SBATCH --job-name=italian_eval_monitor
#SBATCH --partition=general_gpu_p6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=7-0:00:00
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err

# Exit on error
set -e

# --- Script Usage ---
if [ "$#" -ne 1 ]; then
    echo "Usage: sbatch $0 <model_size_tag>"
    echo "Example: sbatch $0 10M"
    exit 1
fi
MODEL_SIZE_TAG=$1

# === Environment Setup ===
echo "=== Job Started: $(date) ==="
echo "Target Model Size: ${MODEL_SIZE_TAG}"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"

# --- Load necessary system modules ---
echo "Loading system modules..."
module load singularity/4.1.1 cuda/11.8 # Assumed versions

# --- Define Host and Container Paths ---
# Assumes this script is run from the project's root directory.
HOST_PROJECT_DIR="../$(pwd)"
HOST_SIF_PATH="${HOST_PROJECT_DIR}/italian_llm_env.sif" # <<< IMPORTANT: UPDATE SIF FILENAME IF NEEDED
HOST_DATA_DIR="${HOST_PROJECT_DIR}/data"
HOST_MODELS_DIR="${HOST_PROJECT_DIR}/models" # Parent models directory
HOST_RESULTS_DIR="${HOST_PROJECT_DIR}/results"
HOST_LOGS_DIR="${HOST_PROJECT_DIR}/logs"

CONTAINER_WORKSPACE="/workspace"
CONTAINER_DATA_DIR="/data"
CONTAINER_MODELS_DIR="/models"
CONTAINER_RESULTS_DIR="/results"

# --- Preparations ---
echo "Project Directory (Host): ${HOST_PROJECT_DIR}"
if [ ! -d "${HOST_MODELS_DIR}/${MODEL_SIZE_TAG}" ]; then echo "ERROR: Target model directory not found at ${HOST_MODELS_DIR}/${MODEL_SIZE_TAG}"; exit 1; fi
mkdir -p "${HOST_RESULTS_DIR}"
mkdir -p "${HOST_LOGS_DIR}"

# === Monitor Script Execution ===
echo "Starting Python monitor.py script inside Singularity container..."

singularity exec --nv \
    -B "${HOST_PROJECT_DIR}":"${CONTAINER_WORKSPACE}" \
    -B "${HOST_DATA_DIR}":"${CONTAINER_DATA_DIR}" \
    -B "${HOST_MODELS_DIR}":"${CONTAINER_MODELS_DIR}" \
    -B "${HOST_RESULTS_DIR}":"${CONTAINER_RESULTS_DIR}" \
    "${HOST_SIF_PATH}" \
    python3 "${CONTAINER_WORKSPACE}/evaluation/monitor.py" \
        --model_parent_dir "${CONTAINER_MODELS_DIR}/${MODEL_SIZE_TAG}" \
        --output_base_dir "${CONTAINER_RESULTS_DIR}" \
        --surprisal_data_dir "${CONTAINER_DATA_DIR}/eval" \
        --perplexity_data_base_path "${CONTAINER_DATA_DIR}/tokenized"

# === Job Completion ===
echo "=== Job Finished: $(date) ==="