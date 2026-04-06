"""Download and prepare Labeled Faces in the Wild (LFW) dataset.

LFW standard benchmark:
- 13,233 images of 5,749 people
- 6,000 pairs (3,000 same person, 3,000 different) for verification

Downloads via scikit-learn (Figshare mirror) — more reliable than UMass direct.
Falls back to UMass if sklearn download fails.

Usage:
    python data/sources/download_lfw.py --output-dir data/raw/lfw
"""

import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path

LFW_PAIRS_URL = "http://vis-www.cs.umass.edu/lfw/pairs.txt"
LFW_IMAGES_URL = "http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz"


def _download_via_sklearn(out: Path) -> bool:
    """
    Download LFW funneled images via scikit-learn (uses Figshare mirror).
    Copies images to out/lfw_funneled/ to match expected structure.
    Returns True on success.
    """
    try:
        from sklearn.datasets import fetch_lfw_people
        print("Downloading LFW via scikit-learn (Figshare mirror)...")
        dataset = fetch_lfw_people(funneled=True, download_if_missing=True, min_faces_per_person=0)

        # sklearn stores images at ~/scikit_learn_data/lfw_home/lfw_funneled/
        import sklearn.datasets._base as _base
        sklearn_lfw_dir = Path(_base.get_data_home()) / "lfw_home" / "lfw_funneled"

        dest = out / "lfw_funneled"
        if dest.exists():
            print(f"Images already exist at {dest}, skipping copy.")
            return True

        if not sklearn_lfw_dir.exists():
            print("sklearn download completed but directory not found.")
            return False

        print(f"Copying images to {dest}...")
        shutil.copytree(sklearn_lfw_dir, dest)
        print("Done.")
        return True

    except Exception as e:
        print(f"sklearn download failed: {e}")
        return False


def _download_via_urllib(out: Path) -> bool:
    """Download LFW funneled images directly from UMass. Returns True on success."""
    try:
        tgz_path = out / "lfw-funneled.tgz"
        images_dir = out / "lfw_funneled"

        if not tgz_path.exists():
            print(f"Downloading lfw-funneled.tgz (~200MB) from UMass...")
            urllib.request.urlretrieve(LFW_IMAGES_URL, tgz_path)
            print(f"Saved to {tgz_path}")

        print("Extracting...")
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(out)
        tgz_path.unlink()
        print(f"Extracted to {images_dir}")
        return True

    except Exception as e:
        print(f"UMass download failed: {e}")
        return False


def _download_pairs(out: Path) -> bool:
    """Download pairs.txt. Returns True on success."""
    pairs_path = out / "pairs.txt"
    if pairs_path.exists():
        print("pairs.txt already exists, skipping.")
        return True

    # Try UMass
    try:
        print("Downloading pairs.txt...")
        urllib.request.urlretrieve(LFW_PAIRS_URL, pairs_path)
        print(f"Saved to {pairs_path}")
        return True
    except Exception:
        pass

    # Try sklearn's cached copy
    try:
        from sklearn.datasets._base import get_data_home
        sklearn_pairs = Path(get_data_home()) / "lfw_home" / "pairs.txt"
        if sklearn_pairs.exists():
            shutil.copy(sklearn_pairs, pairs_path)
            print(f"Copied pairs.txt from sklearn cache.")
            return True
    except Exception:
        pass

    print("Failed to download pairs.txt from all sources.")
    return False


def download(output_dir: str = "data/raw/lfw") -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    images_dir = out / "lfw_funneled"

    # Download images
    if not images_dir.exists():
        success = _download_via_sklearn(out)
        if not success:
            success = _download_via_urllib(out)
        if not success:
            raise RuntimeError(
                "Failed to download LFW dataset from all sources.\n"
                "Download manually from http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz\n"
                f"and extract to {out}/lfw_funneled/"
            )
    else:
        print(f"Images already exist at {images_dir}, skipping.")

    # Download pairs
    _download_pairs(out)

    print(f"\nLFW dataset ready at: {out.resolve()}")
    return str(out.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/raw/lfw")
    args = parser.parse_args()
    download(args.output_dir)
