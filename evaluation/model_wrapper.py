# evaluation/model_wrapper.py

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel
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
        """Loads the tokenizer from the specified path."""
        print(f"Loading tokenizer from: {self.tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        return tokenizer

    def _load_model(self) -> PreTrainedModel:
        """Loads the model from the specified checkpoint and moves it to the device."""
        print(f"Loading model from: {self.checkpoint_path}")
        model = AutoModelForCausalLM.from_pretrained(self.checkpoint_path)
        model.to(self.device)
        model.eval()  # Set model to evaluation mode
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