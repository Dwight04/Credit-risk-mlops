import os
import zipfile
import kaggle
from pathlib import Path

# Define paths
RAW_DATA_DIR = Path("data/raw")
COMPETITION_NAME = "GiveMeSomeCredit"

def download_data():
    # Create data directory if it doesn't exist
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading data from Kaggle competition: {COMPETITION_NAME}")

    # Kaggle API authenticates automatically via
    # KAGGLE_USERNAME and KAGGLE_KEY environment variables
    kaggle.api.authenticate()

    # Download competition data
    kaggle.api.competition_download_files(
        competition=COMPETITION_NAME,
        path=RAW_DATA_DIR,
        quiet=False
    )

    # Unzip downloaded files
    for zip_file in RAW_DATA_DIR.glob("*.zip"):
        print(f"Unzipping {zip_file.name}...")
        with zipfile.ZipFile(zip_file, "r") as z:
            z.extractall(RAW_DATA_DIR)
        zip_file.unlink()  # delete zip after extraction

    print(f"Data downloaded and saved to {RAW_DATA_DIR}")
    print("Files available:")
    for f in RAW_DATA_DIR.iterdir():
        print(f"  {f.name}")

if __name__ == "__main__":
    download_data()
