import os
import zipfile
from pathlib import Path

RAW_DATA_DIR = Path("data/raw")
COMPETITION_NAME = "GiveMeSomeCredit"


def download_data():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Set Kaggle API token from environment variable
    kaggle_token = os.environ.get("KAGGLE_KEY")
    kaggle_username = os.environ.get("KAGGLE_USERNAME")

    if not kaggle_token or not kaggle_username:
        raise ValueError("KAGGLE_USERNAME and KAGGLE_KEY environment variables must be set")

    # Write kaggle.json for the API client
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"
    kaggle_json.write_text(
        f'{{"username":"{kaggle_username}","key":"{kaggle_token}"}}'
    )
    kaggle_json.chmod(0o600)

    print(f"Downloading data from Kaggle competition: {COMPETITION_NAME}")

    import kaggle
    kaggle.api.authenticate()
    kaggle.api.competition_download_files(
        competition=COMPETITION_NAME,
        path=RAW_DATA_DIR,
        quiet=False
    )

    for zip_file in RAW_DATA_DIR.glob("*.zip"):
        print(f"Unzipping {zip_file.name}...")
        with zipfile.ZipFile(zip_file, "r") as z:
            z.extractall(RAW_DATA_DIR)
        zip_file.unlink()

    print(f"Data downloaded to {RAW_DATA_DIR}")
    for f in RAW_DATA_DIR.iterdir():
        print(f"  {f.name}")


if __name__ == "__main__":
    download_data()
