# src/utils.py

# === Imports ===
import logging
import os
import random
import sys
import traceback
from typing import Tuple

import numpy as np
import torch
import torch.distributed


# --- Helper Function Definitions ---

def setup_logging(log_level: int = logging.INFO, rank: int = 0) -> None:
    """
    Configures the root logger for the application.

    Only rank 0 will log at the specified level. Other ranks will be suppressed
    to avoid cluttered logs in a distributed environment.

    Args:
        log_level: The logging level (e.g., logging.INFO).
        rank: The process rank in a distributed setup.
    """
    # Only configure logging for the master process (rank 0)
    # Other processes will be silent unless there's a critical error.
    level = log_level if rank == 0 else logging.CRITICAL + 1

    # Use force=True to allow reconfiguration if logging was already set up.
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    logger = logging.getLogger(__name__)
    if rank == 0:
        logger.info(f"Logging setup complete. Log level set for rank 0.")


def set_seed(seed_value: int) -> None:
    """
    Sets the random seeds for reproducibility across all relevant libraries.

    Args:
        seed_value: The integer value for the seed.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

    logger = logging.getLogger(__name__)
    # Log the seed setting only from the main process.
    if int(os.environ.get("RANK", "0")) == 0:
        logger.info(f"Set random seed: {seed_value}")


def get_device() -> torch.device:
    """
    Determines and returns the appropriate torch.device for training.

    Prioritizes CUDA, then MPS (for Apple Silicon), then CPU.
    In a DDP context, it uses the LOCAL_RANK environment variable to select the correct GPU.

    Returns:
        The selected torch.device.
    """
    logger = logging.getLogger(__name__)

    if torch.cuda.is_available():
        try:
            # In a DDP setup, LOCAL_RANK will be set.
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            if local_rank >= torch.cuda.device_count():
                logger.warning(
                    f"LOCAL_RANK {local_rank} is out of bounds. Defaulting to device 0."
                )
                local_rank = 0

            device = torch.device(f"cuda:{local_rank}")
            device_name = torch.cuda.get_device_name(device)
            logger.info(f"Using CUDA GPU: {local_rank} - {device_name}")
            return device
        except Exception as e:
            logger.error(f"Error setting CUDA device: {e}. Falling back to CPU.")
            return torch.device("cpu")

    elif torch.backends.mps.is_available():
        logger.info("CUDA not available. Using Apple MPS.")
        return torch.device("mps")

    else:
        logger.info("CUDA and MPS not available. Using CPU.")
        return torch.device("cpu")


def setup_distributed() -> Tuple[bool, int, int, int]:
    """
    Sets up the distributed data parallel (DDP) environment if applicable.

    Reads environment variables (WORLD_SIZE, RANK, LOCAL_RANK) to initialize
    the process group.

    Returns:
        A tuple containing:
        - is_distributed (bool): True if DDP is enabled.
        - rank (int): The global rank of the current process.
        - world_size (int): The total number of processes.
        - local_rank (int): The rank of the process on the local machine.
    """
    logger = logging.getLogger(__name__)
    is_dist, rank, world_size, local_rank = False, 0, 1, 0

    if 'WORLD_SIZE' in os.environ and int(os.environ.get('WORLD_SIZE', 1)) > 1:
        is_dist = True
        try:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ["LOCAL_RANK"])

            if not torch.cuda.is_available():
                raise RuntimeError("Distributed training requires CUDA.")

            # Initialize the process group
            torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=world_size)
            torch.cuda.set_device(local_rank)

            msg = f"DDP Initialized: Rank {rank}/{world_size}, LocalRank {local_rank}, Device: cuda:{local_rank}"
            logger.info(msg)

            # Barrier to ensure all processes are synchronized after setup
            torch.distributed.barrier()

        except Exception as e:
            logger.critical(f"Critical error during DDP initialization: {e}")
            traceback.print_exc()
            raise
    else:
        logger.info("DDP not enabled (WORLD_SIZE not set or is 1).")

    return is_dist, rank, world_size, local_rank