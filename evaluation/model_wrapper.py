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

        Surprisal is the negative log probability of a token given the preceding context,
        i.e., -log(P(token_i | token_0, ..., token_{i-1})).

        Args:
            text (str): The input text.

        Returns:
            A tuple containing:
            - A list of tokens (strings).
            - A corresponding list of surprisal values (floats).
              The first token's surprisal is defined as 0.0.
        """
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids

        # Get the model's logits (predictions)
        outputs = self.model(**inputs, labels=input_ids)
        logits = outputs.logits

        # The cross_entropy loss function calculates -log(softmax(logits)).
        # By using reduction='none', we get the loss for each token individually.
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        # Shift logits and labels for next-token prediction
        # The logits at position i are used to predict the token at position i+1
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        # Calculate loss (surprisal) for each token
        surprisals_tensor = loss_fct(shift_logits.view(-1, self.model.config.vocab_size), shift_labels.view(-1))

        # Convert tensor to a list of floats
        surprisals = surprisals_tensor.cpu().numpy().tolist()

        # The first token has no preceding context, so its surprisal is undefined.
        # We assign it 0.0 for practical purposes.
        full_surprisals = [0.0] + surprisals

        # Decode the tokens back to strings for readability
        tokens = [
            self.tokenizer.decode(token_id) for token_id in input_ids.squeeze().tolist()
        ]

        return tokens, full_surprisals