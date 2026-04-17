"""LFW (Labeled Faces in the Wild) dataset adapter.

Parses `lfw-funneled.tgz` layout:
    data_path/
    ├── lfw_funneled/<name>/<name>_<idx:04d>.jpg
    └── pairs.txt

pairs.txt format:
    Line 1: <num_folds>\\t<pairs_per_fold>
    Same pair : <name>\\t<idx_a>\\t<idx_b>
    Diff pair : <name_a>\\t<idx_a>\\t<name_b>\\t<idx_b>

Output contract matches the `load_dataset` step:
    image_root    : absolute path to lfw_funneled/
    unique_images : sorted list of rel paths (e.g. "Abel_Pacheco/Abel_Pacheco_0001.jpg")
    pairs         : list of (rel_path_a, rel_path_b, is_same)
"""

from pathlib import Path


def _rel(name: str, idx: int) -> str:
    return f"{name}/{name}_{idx:04d}.jpg"


def load_lfw(data_path: str) -> tuple[str, list[str], list[tuple[str, str, bool]]]:
    data_dir = Path(data_path)
    image_root = data_dir / "lfw_funneled"
    pairs_file = data_dir / "pairs.txt"

    if not image_root.is_dir():
        raise FileNotFoundError(f"LFW images not found at {image_root}")
    if not pairs_file.is_file():
        raise FileNotFoundError(f"pairs.txt not found at {pairs_file}")

    pairs: list[tuple[str, str, bool]] = []
    lines = pairs_file.read_text().strip().splitlines()
    # lines[0] is "<num_folds>\t<pairs_per_fold>" — skip
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) == 3:
            name, idx_a, idx_b = parts
            pairs.append((_rel(name, int(idx_a)), _rel(name, int(idx_b)), True))
        elif len(parts) == 4:
            name_a, idx_a, name_b, idx_b = parts
            pairs.append((_rel(name_a, int(idx_a)), _rel(name_b, int(idx_b)), False))
        # Silently skip malformed lines (rare; LFW pairs.txt is well-formed)

    unique = sorted({p for a, b, _ in pairs for p in (a, b)})

    return str(image_root.resolve()), unique, pairs
