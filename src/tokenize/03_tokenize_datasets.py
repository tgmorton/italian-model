# 03_tokenize_datasets.py
import os
import sentencepiece as spm
from datasets import load_dataset
from glob import glob


def tokenize_and_save(tokenizer, files_to_tokenize, output_dir, split_name):
    if not files_to_tokenize:
        print(f"  - WARNING: No data files found for '{split_name}' split. Skipping.")
        return

    print(f"  - Tokenizing {len(files_to_tokenize)} file(s) for '{split_name}' split...")
    print(f"  - Output directory: {output_dir}")

    dataset = load_dataset('text', data_files=files_to_tokenize, split='train')

    def tokenize_function(examples):
        return {'input_ids': tokenizer.encode(examples['text'], out_type=int)}

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, num_proc=os.cpu_count(), remove_columns=['text']
    )
    os.makedirs(output_dir, exist_ok=True)
    tokenized_dataset.save_to_disk(output_dir)
    print(f"  - Successfully saved '{split_name}' split.\n")


def main():
    print("\n--- Running Script 03: Tokenizing Datasets to Arrow Format ---")
    base_dir = os.path.expanduser('~/Italian-Model')
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    tokenizer_dir = os.path.join(base_dir, 'tokenizer')
    tokenized_dir = os.path.join(base_dir, 'data', 'tokenized')
    print(f"Reading processed from:  {processed_dir}")
    print(f"Reading tokenizers from: {tokenizer_dir}")
    print(f"Writing tokenized to:    {tokenized_dir}\n")

    sizes = ['10M', '25M', '50M', '100M']

    for size in sizes:
        print(f"----- Processing dataset for size: {size} -----")

        tokenizer_model_path = os.path.join(tokenizer_dir, size, f'tokenizer_{size}.model')
        if not os.path.exists(tokenizer_model_path):
            print(f"  - FATAL ERROR: Tokenizer not found at '{tokenizer_model_path}'. Skipping.")
            continue

        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(tokenizer_model_path)
        print(f"  - Loaded tokenizer: {os.path.basename(tokenizer_model_path)}")

        # --- 1. Process the TRAINING split for this size ---
        train_files = glob(os.path.join(processed_dir, size, '*.train'))
        output_train_dir = os.path.join(tokenized_dir, size, 'train')
        tokenize_and_save(tokenizer, train_files, output_train_dir, 'train')

        # --- 2. Process the common TEST split using this size's tokenizer ---
        # CORRECTED: Now looks for .test files
        test_files = glob(os.path.join(processed_dir, 'test_data', '*.test'))
        output_test_dir = os.path.join(tokenized_dir, size, 'test')
        tokenize_and_save(tokenizer, test_files, output_test_dir, 'test')

    print("\n----- All datasets have been tokenized and saved. -----")


if __name__ == '__main__':
    main()