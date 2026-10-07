"""Shared helpers: repo paths, YAML/JSON IO, seeding, markdown tables."""
import json
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["crack", "corrosion", "oil_leak", "surface_wear", "damage"]  # YOLO class ids 0..4
SEED = 42


def rpath(p):
    """Resolve a repo-relative path to an absolute Path."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_yaml(path):
    with open(rpath(path), encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(p):
    p = rpath(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def save_json(obj, path):
    path = rpath(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    return path


def df_to_md(df, floatfmt="{:.3f}"):
    """Minimal DataFrame -> markdown table (avoids a tabulate dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def save_table(df, stem, folder="results/tables"):
    """Write <stem>.csv and <stem>.md under results/tables."""
    folder = ensure_dir(folder)
    df.to_csv(folder / f"{stem}.csv", index=False)
    (folder / f"{stem}.md").write_text(df_to_md(df), encoding="utf-8")
    return folder / f"{stem}.csv"
