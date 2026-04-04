"""Download and prepare Labeled Faces in the Wild (LFW) dataset.

LFW standard benchmark:
- 13,233 images of 5,749 people
- 6,000 pairs (3,000 same person, 3,000 different) for verification

Usage:
    python data/sources/download_lfw.py --output-dir data/raw/lfw
"""

import argparse
import tarfile
import urllib.request
from pathlib import Path

LFW_IMAGES_URL = "http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz"
LFW_PAIRS_URL = "http://vis-www.cs.umass.edu/lfw/pairs.txt"


def download(output_dir: str = "data/raw/lfw") -> str:
    """
    Download LFW funneled images and pairs.txt.

    Returns:
        Path to the dataset directory.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pairs_path = out / "pairs.txt"
    images_dir = out / "lfw_funneled"

    if not pairs_path.exists():
        print(f"Downloading pairs.txt...")
        urllib.request.urlretrieve(LFW_PAIRS_URL, pairs_path)
        print(f"Saved to {pairs_path}")
    else:
        print(f"pairs.txt already exists, skipping.")

    tgz_path = out / "lfw-funneled.tgz"
    if not images_dir.exists():
        if not tgz_path.exists():
            print(f"Downloading lfw-funneled.tgz (~200MB)...")
            urllib.request.urlretrieve(LFW_IMAGES_URL, tgz_path)
            print(f"Saved to {tgz_path}")

        print("Extracting...")
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(out)
        tgz_path.unlink()
        print(f"Extracted to {images_dir}")
    else:
        print(f"Images already exist at {images_dir}, skipping.")

    print(f"LFW dataset ready at: {out.resolve()}")
    return str(out.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/raw/lfw")
    args = parser.parse_args()
    download(args.output_dir)
