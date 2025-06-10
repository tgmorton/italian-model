# train_tokenizers.py
import os
import sentencepiece as spm
from glob import glob


def main():
    """
    Trains a SentencePiece tokenizer for each dataset size (10M, 25M, 50M, 100M).
    The base directory is hardcoded to '~/Italian-Model'.
    """
    # Hardcode the base directory and expand the '~' to the user's home directory
    base_dir = os.path.expanduser('~/Italian-Model')
    print(f"Using base directory: {base_dir}")

    # Define relative paths for data and tokenizer output
    raw_data_dir = os.path.join(base_dir, 'data', 'raw')

    # List of dataset sizes to process
    sizes = ['10M', '25M', '50M', '100M']
    vocab_size = 32000

    for size in sizes:
        print(f"\n----- Training Tokenizer for {size} -----")

        tokenizer_output_dir = os.path.join(base_dir, 'tokenizer', size)
        os.makedirs(tokenizer_output_dir, exist_ok=True)

        # Gather the training files for this size and the common test files
        train_files = glob(os.path.join(raw_data_dir, size, '*.train'))
        test_files = glob(os.path.join(raw_data_dir, 'test_data', '*.text'))

        input_files = train_files + test_files
        input_files_str = ",".join(input_files)

        if not input_files:
            print(f"Warning: No input files found for size {size}. Skipping.")
            continue

        print(f"Found {len(input_files)} files for training tokenizer.")
        print(f"Output will be saved in: {tokenizer_output_dir}")

        # Define the output model prefix
        model_prefix = os.path.join(tokenizer_output_dir, f'tokenizer_{size}')

        # Train the SentencePiece model
        try:
            spm.SentencePieceTrainer.train(
                f'--input={input_files_str} '
                f'--model_prefix={model_prefix} '
                f'--vocab_size={vocab_size} '
                '--model_type=unigram '
                '--character_coverage=1.0 '
                '--hard_vocab_limit=false'
            )
            print(f"Tokenizer for {size} trained successfully.")
        except Exception as e:
            print(f"Error training tokenizer for {size}: {e}")

    print("\n----- All tokenizer training complete! -----")


if __name__ == '__main__':
    main()