#!/bin/bash
#SBATCH --job-name=italian-llm-lr-sweep    # A descriptive job name
#SBATCH --partition=general_gpu_p6000      # The general_gpu_a500 partition
#SBATCH --array=0-5                        # Create a job array for the learning rate indices
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=7-0:00:00                   # Time limit (D-HH:MM:SS)
#SBATCH --output=../logs/%x-%A_%a.out      # Unique standard output log for each array task
#SBATCH --error=../logs/%x-%A_%a.err       # Unique standard error log for each array task

# Exit on any error
set -e

# --- Define Learning Rates to Sweep ---
LEARNING_RATES=(1e-5 3e-5 1e-4 3e-4 5e-4 6e-4)
CURRENT_LR=${LEARNING_RATES[$SLURM_ARRAY_TASK_ID]}

# --- Environment Setup ---
echo "========================================================"
echo "Job Started: $(date)"
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Learning Rate: ${CURRENT_LR}"
echo "========================================================"

# --- Load necessary system modules ---
echo "Loading system modules..."
module load singularity/4.1.1 cuda/11.8

# --- Define Paths ---
HOST_PROJECT_DIR="/home/AD/thmorton/italian-model"
HOST_SIF_PATH="/home/AD/thmorton/italian-model/italian_llm_env.sif"

# --- Define the experiment config file ---
CONFIG_FILE="configs/25M_experiment.yaml"

# --- Define dynamic and fixed parameters ---
SEED=10
OUTPUT_DIR="models/lr_sweep/25M-lr_${CURRENT_LR}"

# --- Preparations ---
echo "Project Directory (Host): ${HOST_PROJECT_DIR}"
echo "SIF Image Path (Host): ${HOST_SIF_PATH}"
echo "Config File: ${CONFIG_FILE}"
echo "Seed: ${SEED}"
echo "Learning Rate: ${CURRENT_LR}"
echo "Output Directory: ${OUTPUT_DIR}"
if [ ! -f "$HOST_SIF_PATH" ]; then echo "ERROR: Singularity image not found at $HOST_SIF_PATH"; exit 1; fi
mkdir -p "${HOST_PROJECT_DIR}/logs"

# --- Training Script Execution ---
echo "Starting Python training script inside Singularity container..."

# Set PyTorch CUDA Allocator Config
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Execute the training script inside the container, overriding parameters
srun singularity exec --nv \
    --bind "${HOST_PROJECT_DIR}":/workspace \
    "${HOST_SIF_PATH}" \
    bash -c "cd /workspace && python -m src.train \
        --config-file ${CONFIG_FILE} \
        --seed ${SEED} \
        --learning-rate ${CURRENT_LR} \
        --output_dir ${OUTPUT_DIR}"

echo "========================================================"
echo "Job Finished: $(date)"
echo "========================================================"