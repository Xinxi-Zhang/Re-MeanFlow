import argparse
import tarfile
import requests
from pathlib import Path
from tqdm import tqdm

def download_file(url, dest_path):
    """Download large files with streaming and progress bar."""
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(dest_path, 'wb') as file, tqdm(
        desc=dest_path.name,
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024 * 1024):
            size = file.write(data)
            bar.update(size)

def extract_tar(tar_path, dest_dir):
    """Extract a tar file and delete it."""
    print(f"Extracting {tar_path.name}...")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=dest_dir)
    tar_path.unlink()

def setup_imagenet_dataset(dataset_dir: Path, url: str):
    """
    Downloads and extracts the ImageNet 2012 train split such that the final structure is:
    {dataset_dir}/raw/train/<classfolder>/<imagefiles>
    """
    dataset_dir = dataset_dir.resolve()
    raw_dir = dataset_dir / "raw"
    train_dir = raw_dir / "train"
    tar_file = raw_dir / "ILSVRC2012_img_train.tar"

    # Step 1: Create dataset directories
    raw_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)

    # Step 2: Download ImageNet tarball
    print(f"Starting ImageNet download...\nTarget: {tar_file}")
    if not tar_file.exists():
        download_file(url, tar_file)
    else:
        print(f"Tar file already exists at {tar_file}, skipping download.")

    # Step 3: Extract main tar file into train_dir
    print(f"\nExtracting main archive to {train_dir} ...")
    extract_tar(tar_file, train_dir)

    # Step 4: Extract each per-class archive into its own folder in train_dir
    print("\nExtracting per-class archives into training folders (this may take a while)...")
    per_class_tars = list(train_dir.glob("*.tar"))
    for class_tar in tqdm(per_class_tars, desc="Classes"):
        class_name = class_tar.stem
        class_folder = train_dir / class_name
        class_folder.mkdir(exist_ok=True)
        extract_tar(class_tar, class_folder)

    print(f"\nImageNet dataset successfully extracted at: {train_dir}")

def main():
    parser = argparse.ArgumentParser(
        description="Download and extract the ImageNet 2012 dataset automatically (pure Python version)."
    )
    parser.add_argument(
        "--path", type=str, required=True,
        help="Path to store the dataset (e.g., ./dataset)"
    )
    parser.add_argument(
        "--url", type=str, required=True,
        help="Direct URL to the ILSVRC2012_img_train.tar file"
    )
    args = parser.parse_args()

    setup_imagenet_dataset(Path(args.path), args.url)

if __name__ == "__main__":
    main()