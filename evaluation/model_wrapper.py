# evaluation/model_wrapper.py

import torch
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel
from typing import List, Tuple, Union


class ModelWrapper:
    """
    A wrapper for a Hugging Face Causal LM to handle loading and surprisal calculation.
    """

    def __init__(self, checkpoint_path: Union[str, Path], tokenizer_path: Union[str, Path]):
        """
        Initializes and loads the model and tokenizer.

        Args:
            checkpoint_path (Union[str, Path]): Path to the model checkpoint directory.
            tokenizer_path (Union[str, Path]): Path to the tokenizer directory.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.checkpoint_path = Path(checkpoint_path)
        self.tokenizer_path = Path(tokenizer_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {self.checkpoint_path}")
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer directory not found: {self.tokenizer_path}")

        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Loads the tokenizer and ensures a padding token is set."""
        print(f"Loading tokenizer from: {self.tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

        # CORRECTED: Add logic to set a pad_token if one doesn't exist.
        if tokenizer.pad_token is None:
            if tokenizer.eos_token:
                tokenizer.pad_token = tokenizer.eos_token
                print(f"  > tokenizer.pad_token was None. Set to eos_token: '{tokenizer.eos_token}'")
            else:
                # Add a new pad token if no eos token exists either
                new_pad_token = '[PAD]'
                tokenizer.add_special_tokens({'pad_token': new_pad_token})
                print(f"  > tokenizer.pad_token and eos_token were None. Added new pad_token: '{new_pad_token}'")

        return tokenizer

    def _load_model(self) -> PreTrainedModel:
        """
        Loads the model's config, updates vocab size to match the tokenizer,
        and then loads the model weights into the corrected architecture.
        """
        print(f"Loading model from: {self.checkpoint_path}")

        # 1. Load the configuration from the checkpoint.
        model_config = AutoConfig.from_pretrained(self.checkpoint_path)

        # 2. Check for and fix vocabulary size mismatch BEFORE loading the model weights.
        tokenizer_vocab_size = len(self.tokenizer)
        if model_config.vocab_size != tokenizer_vocab_size:
            print(
                f"  > Vocab size mismatch detected. Model config: {model_config.vocab_size}, Tokenizer: {tokenizer_vocab_size}.")
            print(f"  > Updating model config vocab_size to {tokenizer_vocab_size} before loading weights.")
            model_config.vocab_size = tokenizer_vocab_size

        # 3. Load the model weights using the (potentially corrected) configuration.
        #    This forces the model to have the correct output dimensions from the start.
        model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path,
            config=model_config
        )

        # The previous resize_token_embeddings call is no longer needed
        # as this new method is more robust.

        model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def get_surprisals(self, text: str) -> Tuple[List[str], List[float]]:
        """
        Calculates the token-level surprisal for a given string of text.
        """
        # We tokenize twice: once for the model, once to get raw token strings for matching.
        # This is the most reliable way to ensure the strings match what `tokenize` produces.
        raw_tokens = self.tokenizer.tokenize(text)

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids

        # We must ensure the number of tokens matches the number of input_ids.
        # The high-level tokenizer call may add special tokens (like <s>) that .tokenize() does not.
        # We will use convert_ids_to_tokens as the most reliable source of truth.
        tokens_for_matching = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().tolist())

        outputs = self.model(**inputs, labels=input_ids)
        logits = outputs.logits

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        surprisals_tensor = loss_fct(shift_logits.view(-1, self.model.config.vocab_size), shift_labels.view(-1))

        surprisals = surprisals_tensor.cpu().numpy().tolist()

        full_surprisals = [0.0] + surprisals

        return tokens_for_matching, full_surprisals