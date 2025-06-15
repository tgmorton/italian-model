# CSV to UTF-8 Converter
#
# This script searches for all .csv files in the directory where it is located,
# reads them, and then saves them again with standard 'utf-8' encoding.
# This process ensures the files are in a universally compatible format.
#
# How to use:
# 1. Place this script in the folder containing the .csv files you want to convert.
# 2. Run the script from your terminal: python convert_to_utf8.py
#
# Note: This script will overwrite the original files with the newly encoded versions.
# It's always a good idea to have a backup of your files before running.
#
# Requirements:
# No special libraries are needed; this uses standard Python modules.

import os


def convert_csv_to_utf8():
    """
    Finds all .csv files in the current directory and re-saves them
    with standard utf-8 encoding.
    """
    # Get the directory where the script is running
    current_directory = os.getcwd()
    print(f"Searching for .csv files to convert in: {current_directory}\n")

    # Find all files ending with .csv
    csv_files = [f for f in os.listdir(current_directory) if f.endswith('.csv')]

    if not csv_files:
        print("No .csv files found in this directory.")
        return

    for csv_file in csv_files:
        file_path = os.path.join(current_directory, csv_file)
        print(f"Processing '{csv_file}'...")

        try:
            # --- Read the file content ---
            # Try to open with standard UTF-8 first. If that fails, try a common
            # fallback encoding like 'latin-1', as it might be the original encoding.
            content = None
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(f"  - '{csv_file}' is already in UTF-8. Re-saving to ensure standard format.")
            except UnicodeDecodeError:
                print(f"  - Could not read as UTF-8. Trying 'latin-1' encoding.")
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()

            if content is None:
                print(f"  - Could not read file '{csv_file}'. Skipping.")
                continue

            # --- Write the file content back with standard utf-8 ---
            # This makes the file compatible with a wide range of systems and tools.
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"  - Successfully re-saved with UTF-8 encoding.\n")

        except Exception as e:
            print(f"Could not process {csv_file}. Reason: {e}\n")

    print("All files processed.")


if __name__ == "__main__":
    convert_csv_to_utf8()
