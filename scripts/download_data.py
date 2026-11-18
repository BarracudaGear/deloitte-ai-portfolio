"""Download Brazilian E-Commerce (Olist) and PaySim fraud datasets via kagglehub."""

import os
from pathlib import Path
import kagglehub

def download_dataset(handle: str, target_dir: Path) -> None:
    """Download a Kaggle dataset to the target directory and print file info."""
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {handle} to {target_dir} ...")
    
    # kagglehub.dataset_download supports output_dir in recent versions
    downloaded_path = kagglehub.dataset_download(handle, output_dir=str(target_dir))
    print(f"Downloaded to: {downloaded_path}")
    
    # Print all files with sizes
    print("\nFiles downloaded:")
    for root, _, files in os.walk(downloaded_path):
        for file in sorted(files):
            filepath = Path(root) / file
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  {file}: {size_mb:.2f} MB")
    print()

def main():
    base = Path("./data/raw")
    
    # Brazilian E-Commerce dataset
    download_dataset("olistbr/brazilian-ecommerce", base / "olist")
    
    # PaySim fraud dataset
    download_dataset("ealaxi/paysim1", base / "paysim")
    
    print("All datasets downloaded successfully.")

if __name__ == "__main__":
    main()
