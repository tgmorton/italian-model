# src/trainer.py

import gc
import json
import logging
import random
from pathlib import Path
from typing import Dict, Optional, Union, List, Set

import numpy as np
import torch
import torch.distributed
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from .config import TrainingConfig


class Trainer:
    """
    Encapsulates the entire training process for a Hugging Face model.
    """

    def __init__(
        self,
        config: TrainingConfig,
        model: Union[PreTrainedModel, DDP],
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler,
        train_dataloader: DataLoader,
        train_sampler: Optional[Sampler],
        device: torch.device,
        tokenizer: PreTrainedTokenizer,
        num_training_steps: int,
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.train_sampler = train_sampler
        self.device = device
        self.tokenizer = tokenizer
        self.num_training_steps = num_training_steps

        self.logger = logging.getLogger(__name__)
        self.scaler = GradScaler(enabled=self.config.use_amp)
        self.is_main_process = (
                not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
        )

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.total_loss_since_logging = 0.0
        self.steps_since_logging = 0

        self.checkpoint_schedule_set: Optional[Set[int]] = None

        # Prioritize the custom schedule if provided
        if self.config.checkpoint_schedule:
            self.checkpoint_schedule_set = set(self.config.checkpoint_schedule)
            if self.is_main_process:
                self.logger.info(
                    f"Using a custom checkpoint schedule with {len(self.checkpoint_schedule_set)} specific steps.")
        else:
            # Fallback to log-step + periodic saving if no schedule is given
            self.log_save_steps = set()
            if self.config.save_steps > 0:
                current_log_step = 1
                while current_log_step < self.config.save_steps:
                    self.log_save_steps.add(current_log_step)
                    next_log_step = current_log_step * 2
                    if next_log_step <= current_log_step: break
                    current_log_step = next_log_step
            if self.is_main_process:
                self.logger.info(
                    f"Using periodic saving every {self.config.save_steps} steps, with log-step checkpoints at: {sorted(list(self.log_save_steps))}")

        if self.config.checkpoint_path:
            self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        """Loads training state from a checkpoint."""
        if not self.config.checkpoint_path:
            return

        state_file = self.config.checkpoint_path / "training_state.pt"
        if not state_file.is_file():
            self.logger.warning(
                f"Checkpoint path specified, but {state_file} not found. Starting fresh."
            )
            return

        self.logger.info(f"Loading training state from: {state_file}")
        try:
            ckpt = torch.load(state_file, map_location="cpu")

            # Load model state
            model_to_load = self.model.module if isinstance(self.model, DDP) else self.model
            model_to_load.load_state_dict(ckpt["model"])
            self.logger.info("Model state loaded successfully.")

            # Load trainer and hardware state
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            if self.scaler.is_enabled() and "scaler" in ckpt:
                self.scaler.load_state_dict(ckpt["scaler"])

            # Load training progress
            self.current_epoch = ckpt.get("epoch", 0) + 1  # Start next epoch
            self.global_step = ckpt.get("global_step", 0)

            # Restore RNG states for reproducibility
            random.setstate(ckpt["python_rng_state"])
            np.random.set_state(ckpt["numpy_rng_state"])
            torch.set_rng_state(ckpt["torch_rng_state"])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(ckpt["torch_cuda_rng_state_all"])

            self.logger.info(
                f"Resuming training from epoch {self.current_epoch}, global step {self.global_step}."
            )
            del ckpt
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint from {state_file}: {e}", exc_info=True)
            # Reset state to be safe
            self.current_epoch = 0
            self.global_step = 0

    def _save_checkpoint(self) -> None:
        """Saves the complete training state to a checkpoint directory."""
        if not self.is_main_process:
            return

        ckpt_dir = self.config.output_dir / f"checkpoint-{self.global_step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Saving checkpoint to {ckpt_dir}")

        unwrapped_model = self.model.module if isinstance(self.model, DDP) else self.model

        # Create state dictionary
        state = {
            "model": unwrapped_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "config": self.config.model_dump(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

        # Save the training state
        torch.save(state, ckpt_dir / "training_state.pt")

        # Save the model and tokenizer using Hugging Face's format
        unwrapped_model.save_pretrained(ckpt_dir)
        self.tokenizer.save_pretrained(ckpt_dir)
        self.logger.info(f"Checkpoint {self.global_step} saved successfully.")

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Performs a single training step."""
        with torch.autocast(
                device_type=self.device.type, enabled=self.config.use_amp
        ):
            batch_on_device = {
                k: v.to(self.device, non_blocking=True) for k, v in batch.items()
            }
            outputs = self.model(**batch_on_device)
            loss = outputs.loss

        # loss is a scalar, but scaled_loss is divided by accumulation steps
        scaled_loss = loss / self.config.gradient_accumulation_steps
        self.scaler.scale(scaled_loss).backward()

        return loss.item()

    def _log_metrics(self) -> None:
        """Logs training metrics."""
        if not self.is_main_process or self.steps_since_logging == 0:
            return

        avg_loss = self.total_loss_since_logging / self.steps_since_logging
        learning_rate = self.lr_scheduler.get_last_lr()[0]

        self.logger.info(
            f"Step: {self.global_step} | "
            f"Epoch: {self.current_epoch + 1} | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"LR: {learning_rate:.6e}"
        )
        # Reset counters
        self.total_loss_since_logging = 0.0
        self.steps_since_logging = 0

    def train(self) -> None:
        """The main training loop."""
        self.logger.info("***** Starting Training *****")
        self.logger.info(f"  Config: {self.config.model_dump_json(indent=2)}")

        max_steps = self.config.max_steps
        num_epochs = self.config.num_train_epochs

        progress_bar = tqdm(
            total=self.num_training_steps,  # Use the correct total number of optimizer steps
            initial=self.global_step,  # Start the bar from the current global step (for resuming)
            disable=not self.is_main_process,
            desc="Training"
        )

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            self.model.train()
            if self.train_sampler and hasattr(self.train_sampler, "set_epoch"):
                self.train_sampler.set_epoch(epoch)

            for step, batch in enumerate(self.train_dataloader):
                loss = self._train_step(batch)
                self.total_loss_since_logging += loss
                self.steps_since_logging += 1

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Unscale and clip gradients
                    if self.config.max_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                    # Optimizer and scheduler step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1
                    progress_bar.update(1)

                    # --- MODIFIED SAVE CONDITION ---
                    should_save = False
                    if self.checkpoint_schedule_set:
                        # Use the custom schedule if it exists
                        if self.global_step in self.checkpoint_schedule_set:
                            should_save = True
                    else:
                        # Fallback to the old periodic/log-step logic
                        is_log_save_step = self.global_step in self.log_save_steps
                        is_regular_save_step = (
                                    self.config.save_steps > 0 and self.global_step > 0 and self.global_step % self.config.save_steps == 0)
                        if is_log_save_step or is_regular_save_step:
                            should_save = True

                    if should_save:
                        self._save_checkpoint()

                if max_steps > 0 and self.global_step >= max_steps:
                    break
            if max_steps > 0 and self.global_step >= max_steps:
                break

        progress_bar.close()
        self.logger.info("***** Training Finished *****")
        self._save_checkpoint()  # Save final checkpoint

    def save_final_model(self) -> None:
        """Saves the final model and config to a 'final_model' directory."""
        if not self.is_main_process:
            return

        final_dir = self.config.output_dir / "final_model"
        final_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Saving final model to {final_dir}")

        unwrapped_model = self.model.module if isinstance(self.model, DDP) else self.model
        unwrapped_model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)

        # Save the final config used for this run
        with open(final_dir / "training_config.json", "w") as f:
            f.write(self.config.model_dump_json(indent=2))

        self.logger.info("Final model and config saved successfully.")