# report_model_params.py

import argparse
import logging
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoConfig

# --- Basic Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def report_model_parameters(checkpoint_path: str):
    """
    Loads a model from a checkpoint and reports its key parameters.

    Args:
        checkpoint_path (str): The path to the directory containing the saved model
                               (e.g., the output of the Hugging Face Trainer).
    """
    checkpoint_dir = Path(checkpoint_path)

    if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
        logger.error(f"Error: Checkpoint path not found or is not a directory: {checkpoint_path}")
        return

    try:
        # --- Load Model and its Configuration ---
        logger.info(f"Loading model from checkpoint: '{checkpoint_path}'")

        # The config contains all the architectural details.
        config = AutoConfig.from_pretrained(checkpoint_dir)

        # Load the model weights. We don't need the whole model instance for reporting,
        # but loading it confirms the checkpoint is valid.
        model = AutoModelForCausalLM.from_pretrained(checkpoint_dir)
        logger.info("Model loaded successfully.")

        # --- Parameter Extraction and Calculation ---
        # Get the total number of parameters.
        # The `num_parameters()` method provides a convenient way to count them.
        total_params = model.num_parameters()
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Extract key architectural details from the config object.
        # Using .get() provides a safe way to access attributes that might not exist.
        vocab_size = config.vocab_size
        n_positions = config.n_positions
        n_layer = getattr(config, 'n_layer', 'N/A')  # For GPT-2 style
        n_head = getattr(config, 'n_head', 'N/A')  # For GPT-2 style
        n_embd = getattr(config, 'n_embd', 'N/A')  # For GPT-2 style

        # For other architectures like Llama, names might differ
        if n_layer == 'N/A':
            n_layer = getattr(config, 'num_hidden_layers', 'N/A')
        if n_head == 'N/A':
            n_head = getattr(config, 'num_attention_heads', 'N/A')
        if n_embd == 'N/A':
            n_embd = getattr(config, 'hidden_size', 'N/A')

        # --- Reporting ---
        print("\n" + "=" * 50)
        print(f"Model Report for: {checkpoint_path}")
        print("=" * 50)

        print("\n--- Parameter Count ---")
        print(f"Total Parameters:     {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")

        print("\n--- Architecture Details ---")
        print(f"Vocabulary Size (vocab_size):      {vocab_size}")
        print(f"Max Sequence Length (n_positions): {n_positions}")
        print(f"Number of Layers (n_layer):        {n_layer}")
        print(f"Number of Attention Heads (n_head):{n_head}")
        print(f"Embedding Dimension (n_embd):      {n_embd}")
        print(f"Model Architecture Type:           {config.model_type}")

        print("\n" + "=" * 50 + "\n")

    except Exception as e:
        logger.error(f"An error occurred while processing the checkpoint: {e}", exc_info=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Load a Hugging Face model from a checkpoint and report its parameters."
    )
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to the model checkpoint directory."
    )

    args = parser.parse_args()
    report_model_parameters(args.checkpoint_path)

