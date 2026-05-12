import os
import zipfile
import subprocess
from pathlib import Path

RAW_DATA_DIR = Path("data/raw")


def download_data():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    kaggle_token = os.environ.get("KAGGLE_KEY")

    if not kaggle_token:
        raise ValueError("KAGGLE_KEY environment variable must be set")

    print("Downloading data from Kaggle...")

    url = "https://www.kaggle.com/api/v1/competitions/data/download-all/GiveMeSomeCredit"
    output_path = RAW_DATA_DIR / "data.zip"

    result = subprocess.run([
        "curl", "-L",
        "-H", f"Authorization: Bearer {kaggle_token}",
        "-o", str(output_path),
        url
    ], capture_output=True, text=True)

    print("CURL stdout:", result.stdout)
    print("CURL stderr:", result.stderr)

    # Print first 500 chars of downloaded file to diagnose
    with open(output_path, "rb") as f:
        content = f.read(500)
    print("Downloaded content preview:", content)

    # Check if it's actually a zip
    if not zipfile.is_zipfile(output_path):
        raise ValueError(
            f"Downloaded file is not a zip. "
            f"Content: {content.decode('utf-8', errors='ignore')}"
        )

    print("Unzipping...")
    with zipfile.ZipFile(output_path, "r") as z:
        z.extractall(RAW_DATA_DIR)
    output_path.unlink()

    print("Files available:")
    for f in RAW_DATA_DIR.iterdir():
        print(f"  {f.name}")


if __name__ == "__main__":
    download_data()
