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
if [ "$#" -eq 0 ] || [ "$#" -gt 2 ]; then
    echo "Usage: sbatch $0 <model_size_tag> [perplexity_eval_portion]"
    echo "Example (full run): sbatch $0 10M"
    echo "Example (33% of perplexity test set): sbatch $0 10M 0.33"
    exit 1
fi
MODEL_SIZE_TAG=$1
PERPLEXITY_PORTION=$2

# === Environment Setup ===
echo "=== Job Started: $(date) ==="
echo "Target Model Size: ${MODEL_SIZE_TAG}"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"

# --- Load necessary system modules ---
echo "Loading system modules..."
module load singularity/4.1.1 cuda/11.8 # Assumed versions

# --- Define Host and Container Paths ---
HOST_PROJECT_DIR="/home/AD/thmorton/italian-model"
HOST_SIF_PATH="${HOST_PROJECT_DIR}/italian_llm_env.sif"
HOST_DATA_DIR="${HOST_PROJECT_DIR}/data"
HOST_MODELS_DIR="${HOST_PROJECT_DIR}/models"
HOST_RESULTS_DIR="${HOST_PROJECT_DIR}/results"
HOST_LOGS_DIR="${HOST_PROJECT_DIR}/logs"
HOST_TOKENIZER_DIR="${HOST_PROJECT_DIR}/tokenizer"
# CORRECTED: Point to the actual surprisal data directory
HOST_SURPRISAL_DIR="${HOST_PROJECT_DIR}/evaluation/data/italian_null_subject"

CONTAINER_WORKSPACE="/workspace"
CONTAINER_DATA_DIR="/data"
CONTAINER_MODELS_DIR="/models"
CONTAINER_RESULTS_DIR="/results"
CONTAINER_TOKENIZER_DIR="/workspace/tokenizer"
# CORRECTED: Define a new container path for the surprisal data
CONTAINER_SURPRISAL_DIR="/surprisal_data"

# --- Process Optional Argument ---
PERPLEXITY_ARG=""
if [ -n "$PERPLEXITY_PORTION" ]; then
    PERPLEXITY_ARG="--perplexity_eval_portion ${PERPLEXITY_PORTION}"
fi

# --- Preparations ---
echo "Project Directory (Host): ${HOST_PROJECT_DIR}"
if [ ! -d "${HOST_MODELS_DIR}/${MODEL_SIZE_TAG}" ]; then echo "ERROR: Target model directory not found at ${HOST_MODELS_DIR}/${MODEL_SIZE_TAG}"; exit 1; fi
if [ ! -d "${HOST_SURPRISAL_DIR}" ]; then echo "ERROR: Surprisal data directory not found at ${HOST_SURPRISAL_DIR}"; exit 1; fi
mkdir -p "${HOST_RESULTS_DIR}"
mkdir -p "${HOST_LOGS_DIR}"

# === Monitor Script Execution ===
echo "Starting Python monitor.py script inside Singularity container..."

singularity exec --nv \
    -B "${HOST_PROJECT_DIR}":"${CONTAINER_WORKSPACE}" \
    -B "${HOST_DATA_DIR}":"${CONTAINER_DATA_DIR}" \
    -B "${HOST_MODELS_DIR}":"${CONTAINER_MODELS_DIR}" \
    -B "${HOST_RESULTS_DIR}":"${CONTAINER_RESULTS_DIR}" \
    -B "${HOST_SURPRISAL_DIR}":"${CONTAINER_SURPRISAL_DIR}" \
    "${HOST_SIF_PATH}" \
    bash -c "cd ${CONTAINER_WORKSPACE} && python3 -m evaluation.monitor \
        --model_parent_dir \"${CONTAINER_MODELS_DIR}/${MODEL_SIZE_TAG}\" \
        --output_base_dir \"${CONTAINER_RESULTS_DIR}\" \
        --tokenizer_base_dir \"${CONTAINER_TOKENIZER_DIR}\" \
        --surprisal_data_dir \"${CONTAINER_SURPRISAL_DIR}\" \
        --perplexity_data_base_path \"${CONTAINER_DATA_DIR}/tokenized\" \
        ${PERPLEXITY_ARG}"

# === Job Completion ===
echo "=== Job Finished: $(date) ==="