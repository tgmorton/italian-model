# 02b_wrap_tokenizer.py
import os
from transformers import LlamaTokenizer


def main():
    """
    Loads the raw sentencepiece models and saves them in the Hugging Face
    transformers format. This creates the necessary config files for the
    training script to load the tokenizer correctly.
    """
    print("\n--- Running Script 02b: Wrapping SentencePiece Tokenizers for Transformers ---")
    base_dir = os.path.expanduser('~/Italian-Model')
    tokenizer_dir = os.path.join(base_dir, 'tokenizer')
    print(f"Reading from and writing to: {tokenizer_dir}\n")

    sizes = ['10M', '25M', '50M', '100M']

    for size in sizes:
        print(f"----- Wrapping tokenizer for size: {size} -----")

        sp_model_path = os.path.join(tokenizer_dir, size, f'tokenizer_{size}.model')

        if not os.path.exists(sp_model_path):
            print(f"  - WARNING: SentencePiece model not found at '{sp_model_path}'. Skipping.")
            continue

        print(f"  - Loading SentencePiece model: {sp_model_path}")

        # 1. Load the raw sentencepiece model into a LlamaTokenizer object
        try:
            tokenizer = LlamaTokenizer(vocab_file=sp_model_path)
        except Exception as e:
            print(f"  - ERROR: Failed to load tokenizer for size {size}: {e}")
            continue

        # 2. Save the tokenizer using save_pretrained. This creates the necessary files.
        output_dir = os.path.join(tokenizer_dir, size)
        print(f"  - Saving in Transformers format to: {output_dir}")
        tokenizer.save_pretrained(output_dir)
        print(f"  - Successfully wrapped tokenizer for {size}.\n")

    print("\n----- All tokenizers have been wrapped. -----")


if __name__ == '__main__':
    main()