# Italian Language Model Training Pipeline

This repository contains a flexible and modular pipeline for training Italian language models from scratch. It is designed to facilitate experimentation with different model sizes, architectures, and training datasets.

The entire workflow, from text preprocessing and tokenizer training to the final model training, is handled by a series of easy-to-use scripts.

## Features

* **End-to-End Workflow**: A complete pipeline from raw text files to a trained model.
* **Custom Tokenizer Training**: Scripts to train your own `sentencepiece` tokenizers on specific data sizes.
* **Modular Training Script**: The core training logic is built with a modern, object-oriented structure that is easy to read and maintain.
* **Flexible Configuration**: Centrally manage experiments using YAML files. Any parameter can be easily overridden from the command line for quick tests and sweeps.
* **Custom Architectures**: Define your own model architectures (number of layers, heads, etc.) in a simple YAML file and train them from scratch.
* **Modern Libraries**: Built on top of PyTorch, Hugging Face `transformers`, `datasets`, `pydantic`, and `typer`.

## Project Structure

```
├── configs/
│ ├── model_architectures.yaml # Defines custom model sizes (e.g., 'gpt2-10m')
│ └── 10M_experiment.yaml # Example experiment config file
│
├── data/
│ ├── raw/ # Place your initial raw text files here
│ ├── processed/ # Cleaned, sentence-split text files are saved here
│ └── tokenized/ # Final tokenized Arrow datasets for training
│
├── models/
│ └── # Output directory for saved model checkpoints and final models
│
├── src/
│ ├── tokenize/ # Scripts for the data preparation pipeline
│ │ ├── 01_preprocess.py
│ │ ├── 02_train_tokenizers.py
│ │ └── 03_tokenize_datasets.py
│ │
│ ├── config.py # Pydantic model for type-safe configuration
│ ├── data.py # Handles dataset loading
│ ├── model.py # Handles model creation
│ ├── trainer.py # The core training engine
│ ├── train.py # Main entry point for starting a training run
│ └── utils.py # Helper functions
│
└── tokenizer/
└── # Output directory for trained tokenizers
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the required dependencies:**
    *(A `requirements.txt` file should be created for this. Based on the project, it would include at least these packages.)*
    ```bash
    pip install torch transformers datasets sentencepiece protobuf pyyaml pydantic "typer[all]" tqdm numpy nltk
    ```

## Workflow: From Raw Data to Trained Model

Follow these steps in order to run the full pipeline.

#### Step 1: Place Raw Data

Place your raw Italian text files inside the `data/raw/{datasize}` directories (e.g., `data/raw/10M/`). The scripts are designed to find files ending in `.train` for training data and `.test` for test data.

#### Step 2: Preprocess Text

This script cleans the text, converts it to lowercase, and splits it into sentences.
```bash
python src/tokenize/01_preprocess.py
```

#### Step 3: Train Custom Tokenizers

This script trains a `sentencepiece` tokenizer for each data size defined in the script.
```bash
python src/tokenize/02_train_tokenizers.py
```

#### Step 4: Wrap Tokenizers for Transformers

This essential step takes the raw `sentencepiece` models and saves them in a format that the Hugging Face `transformers` library can load directly.
```bash
python src/tokenize/02b_wrap_tokenizer.py
```

#### Step 5: Tokenize Datasets

The final data preparation step uses the newly created tokenizers to convert the processed text into efficient Arrow datasets for training.
```bash
python src/tokenize/03_tokenize_datasets.py
```

#### Step 6: Configure and Run Training

Training runs are controlled by YAML files in the `configs/` directory.

1.  **Create an experiment file** (e.g., `configs/my_experiment.yaml`) or modify the existing one to point to the correct data, tokenizer, and output paths. Define your desired hyperparameters.

2.  **Launch the training** by pointing the script to your config file:
    ```bash
    python -m src.train --config-file configs/my_experiment.yaml
    ```

3.  **(Optional) Override Parameters:** For quick experiments, you can override any setting from your YAML file directly on the command line.
    ```bash
    # Use the experiment file but run for only 3 epochs with a different learning rate
    python -m src.train --config-file configs/my_experiment.yaml --num-train-epochs 3 --learning-rate 1e-4
    ```