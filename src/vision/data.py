"""Dataset-yaml helpers shared by train/evaluate (no ultralytics import here)."""
import random
from pathlib import Path

import yaml

from src.common import rpath


def resolve_data_yaml(data_yaml="data/processed/data.yaml"):
    """Write <dir>/data_resolved.yaml with an absolute `path` for this machine and return its path.

    Ultralytics resolves a relative/missing `path` against its global DATASETS_DIR, not the yaml's folder.
    """
    src = rpath(data_yaml)
    if not src.is_file():
        raise FileNotFoundError(f"{src} not found - run scripts/convert_to_yolo_seg.py first")
    d = yaml.safe_load(src.read_text(encoding="utf-8"))
    d["path"] = str(src.parent.resolve())
    out = src.parent / "data_resolved.yaml"
    out.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    return out


def subset_yaml(resolved_yaml, out_dir, lists):
    """Write a yaml whose splits are text files of absolute image paths.

    lists: {split_name: [image paths]}. Ultralytics finds labels by swapping /images/ -> /labels/.
    """
    d = yaml.safe_load(Path(resolved_yaml).read_text(encoding="utf-8"))
    out_dir = rpath(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, paths in lists.items():
        txt = out_dir / f"{split}.txt"
        txt.write_text("\n".join(str(Path(p).resolve()) for p in paths) + "\n", encoding="utf-8")
        d[split] = str(txt)
    for split in ("train", "val", "test"):
        d.setdefault(split, d.get("val"))
    out = out_dir / "data.yaml"
    out.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    return out


def split_images(resolved_yaml, split):
    d = yaml.safe_load(Path(resolved_yaml).read_text(encoding="utf-8"))
    folder = Path(d["path"]) / d[split]
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})


def sample(paths, n, seed=42):
    paths = list(paths)
    return paths if len(paths) <= n else random.Random(seed).sample(paths, n)


def source_of(path):
    """Images are named <source>__<original-name>; see convert_to_yolo_seg.safe_stem."""
    return Path(path).name.split("__", 1)[0]
