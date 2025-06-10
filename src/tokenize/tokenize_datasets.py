# tokenize_datasets.py
import os
import sentencepiece as spm
from datasets import load_dataset
from glob import glob


def tokenize_and_save(tokenizer, data_files, output_dir, split_name):
    """
    Tokenizes raw text files and saves them as a Hugging Face Arrow dataset.
    """
    if not data_files:
        print(f"No data files found for {split_name} at {output_dir}. Skipping.")
        return

    print(f"Tokenizing {split_name} split, saving to {output_dir}...")

    # 1. Load raw text files into a Hugging Face Dataset
    dataset = load_dataset('text', data_files=data_files, split='train')

    # 2. Define the tokenization function
    def tokenize_function(examples):
        return {'input_ids': tokenizer.encode(examples['text'], out_type=int)}

    # 3. Tokenize the dataset using multiple processes
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=['text']
    )

    # 4. Save the tokenized dataset to disk
    os.makedirs(output_dir, exist_ok=True)
    tokenized_dataset.save_to_disk(output_dir)
    print(f"Successfully saved tokenized {split_name} split.")


def main():
    """
    Uses the pre-trained tokenizers to convert the raw Italian text data
    into tokenized Arrow files. The base directory is hardcoded to '~/Italian-Model'.
    """
    # Hardcode the base directory and expand the '~' to the user's home directory
    base_dir = os.path.expanduser('~/Italian-Model')
    print(f"Using base directory: {base_dir}")

    # Define relative paths
    raw_data_dir = os.path.join(base_dir, 'data', 'raw')
    tokenized_data_dir = os.path.join(base_dir, 'data', 'tokenized')
    tokenizer_base_dir = os.path.join(base_dir, 'tokenizer')

    # List of dataset sizes to process
    sizes = ['10M', '25M', '50M', '100M']

    for size in sizes:
        print(f"\n----- Processing dataset for size: {size} -----")

        # Path to the tokenizer model for the current size
        tokenizer_model_path = os.path.join(tokenizer_base_dir, size, f'tokenizer_{size}.model')

        if not os.path.exists(tokenizer_model_path):
            print(f"ERROR: Tokenizer model not found at {tokenizer_model_path}")
            print("Please run train_tokenizers.py first.")
            continue

        # Load the specific tokenizer for this size
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(tokenizer_model_path)

        # --- Process the training set for this size ---
        train_files = glob(os.path.join(raw_data_dir, size, '*.train'))
        output_train_dir = os.path.join(tokenized_data_dir, size, 'train')
        tokenize_and_save(tokenizer, train_files, output_train_dir, 'train')

        # --- Process the common test set using this size's tokenizer ---
        test_files = glob(os.path.join(raw_data_dir, 'test_data', '*.text'))
        output_test_dir = os.path.join(tokenized_data_dir, size, 'test')
        tokenize_and_save(tokenizer, test_files, output_test_dir, 'test')

    print("\n----- All datasets have been tokenized and saved! -----")


if __name__ == '__main__':
    main()