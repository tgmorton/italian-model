# 02_train_tokenizers.py
import os
import sentencepiece as spm
from glob import glob


def main():
    print("\n--- Running Script 02: Training Tokenizers ---")
    base_dir = os.path.expanduser('~/Italian-Model')
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    tokenizer_dir = os.path.join(base_dir, 'tokenizer')
    print(f"Reading processed from:   {processed_dir}")
    print(f"Writing tokenizers to:    {tokenizer_dir}\n")

    sizes = ['10M', '25M', '50M', '100M']

    for size in sizes:
        print(f"----- Preparing Tokenizer for size: {size} -----")

        # CORRECTED: Now looks for .test files
        train_files_pattern = os.path.join(processed_dir, size, '*.train')
        test_files_pattern = os.path.join(processed_dir, 'test_data', '*.test')

        train_files = glob(train_files_pattern)
        test_files = glob(test_files_pattern)

        print(f"  - Found {len(train_files)} training files for {size}.")
        print(f"  - Found {len(test_files)} common test files to include.")

        all_input_files = train_files + test_files

        if not all_input_files:
            print(f"  - WARNING: No files found for size {size}. Skipping.")
            continue

        output_model_dir = os.path.join(tokenizer_dir, size)
        os.makedirs(output_model_dir, exist_ok=True)
        model_prefix = os.path.join(output_model_dir, f'tokenizer_{size}')

        args = {
            'input': ",".join(all_input_files),
            'model_prefix': model_prefix,
            'vocab_size': 32000,
            'model_type': 'unigram',
            'max_sentence_length': 8192,
            'character_coverage': 1.0,
            'hard_vocab_limit': 'false',
        }
        arg_string = ' '.join([f'--{key}={value}' for key, value in args.items()])

        print(f"  - Starting training for tokenizer_{size}.model...")
        spm.SentencePieceTrainer.train(arg_string)
        print(f"  - Successfully trained tokenizer for {size}.\n")

    print("\n----- All tokenizer training complete. -----")


if __name__ == '__main__':
    main()