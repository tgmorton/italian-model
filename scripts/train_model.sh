#!/bin/bash
#SBATCH --job-name=italian-llm-train    # A descriptive job name
#SBATCH --partition=general_gpu_a5000   # Your cluster's GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=7-0:00:00                # Time limit (D-HH:MM:SS)
#SBATCH --output=../logs/%x-%j.out         # Standard output log
#SBATCH --error=../logs/%x-%j.err          # Standard error log

# Exit on any error
set -e

# --- Configuration ---
# MODIFICATION: The script now takes a config NAME, not a full path.
if [ -z "$1" ]; then
    echo "ERROR: No config name provided."
    echo "Usage: sbatch $0 <config_name_without_yaml_extension>"
    exit 1
fi
CONFIG_NAME="$1"

# --- Environment Setup ---
echo "========================================================"
echo "Job Started: $(date)"
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "========================================================"

# --- Load necessary system modules ---
# Ensure these versions match what's available on your cluster
echo "Loading system modules..."
module load singularity/4.1.1 cuda/11.8

# --- Define Paths ---
# !!! IMPORTANT: UPDATE THESE PATHS TO MATCH YOUR ENVIRONMENT !!!
HOST_PROJECT_DIR="/home/AD/thmorton/italian-model"
HOST_SIF_PATH="/home/AD/thmorton/italian-model/italian_llm_env.sif"

# MODIFICATION: Construct the full path to the config file from the name.
# This path is for the host machine, used to verify the file exists.
HOST_CONFIG_FILE="${HOST_PROJECT_DIR}/configs/${CONFIG_NAME}.yaml"
# This path is relative to the /workspace inside the container.
CONTAINER_CONFIG_FILE="configs/${CONFIG_NAME}.yaml"

# --- Preparations ---
echo "Project Directory (Host): ${HOST_PROJECT_DIR}"
echo "SIF Image Path (Host): ${HOST_SIF_PATH}"

# MODIFICATION: Check for the config file using the fully-qualified host path.
if [ ! -f "$HOST_CONFIG_FILE" ]; then
    echo "ERROR: Config file not found at $HOST_CONFIG_FILE"
    exit 1
fi
if [ ! -f "$HOST_SIF_PATH" ]; then
    echo "ERROR: Singularity image not found at $HOST_SIF_PATH"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p "${HOST_PROJECT_DIR}/logs"

# --- Training Script Execution ---
# MODIFICATION: Use the container-relative path when calling the python script.
echo "Starting Python training script inside Singularity container..."
echo "Using config file: ${CONTAINER_CONFIG_FILE}"

# Set PyTorch CUDA Allocator Config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Execute the training script inside the container
# --nv: Enables NVIDIA GPU access
# --bind: Mounts the project directory into /workspace inside the container
srun singularity exec --nv \
    --bind "${HOST_PROJECT_DIR}":/workspace \
    "${HOST_SIF_PATH}" \
    bash -c "cd /workspace && python -m src.train --config-file ${CONTAINER_CONFIG_FILE}"

# --- To override a parameter from the YAML file, add it to the command ---
# Example:
# srun singularity exec --nv \
#     --bind "${HOST_PROJECT_DIR}":/workspace \
#     "${HOST_SIF_PATH}" \
#     bash -c "cd /workspace && python -m src.train --config-file ${CONTAINER_CONFIG_FILE} --learning-rate 1e-4"

echo "========================================================"
echo "Job Finished: $(date)"
echo "========================================================"
