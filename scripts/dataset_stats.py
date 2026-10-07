"""A1: Dataset statistics for data/processed (run after convert_to_yolo_seg.py).

Writes results/figures/dataset_*.png, results/tables/dataset_*.csv|md, results/metrics/dataset_summary.json.
Usage: python scripts/dataset_stats.py [--data data/processed] [--grid 12]
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import CLASSES, SEED, rpath, save_json, save_table  # noqa: E402
from src.plotting import CLASS_COLORS, INK2, plt, savefig  # noqa: E402


def bar(ax, labels, values, color, title, xlabel):
    y = np.arange(len(labels))
    ax.barh(y, values, color=color, height=0.6)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    for yi, v in zip(y, values):
        ax.text(v, yi, f" {v:,}", va="center", color=INK2, fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--grid", type=int, default=12, help="number of sample images in the grid")
    ap.add_argument("--out", default="results", help="results root")
    args = ap.parse_args()
    d, R = rpath(args.data), rpath(args.out)
    man, inst = pd.read_csv(d / "manifest.csv"), pd.read_csv(d / "instances.csv")
    classes = [c for c in CLASSES if c in set(inst["cls"])] or CLASSES

    # tables
    per_src = man.pivot_table(index="source", columns="split", values="file", aggfunc="count", fill_value=0)
    per_src = per_src.reindex(columns=["train", "val", "test"], fill_value=0)
    per_src["total"] = per_src.sum(axis=1)
    per_src["share"] = (per_src["total"] / per_src["total"].sum()).round(3)
    save_table(per_src.reset_index(), "dataset_images_per_source", R / "tables")
    sxc = pd.crosstab(inst["source"], inst["cls"]).reindex(columns=CLASSES, fill_value=0)
    sxc["total"] = sxc.sum(axis=1)
    save_table(sxc.reset_index(), "dataset_source_by_class", R / "tables")
    area = inst.groupby("cls")["area_ratio"].describe(percentiles=[0.1, 0.5, 0.9]).reindex(classes)
    save_table(area.reset_index().rename(columns={"index": "cls"}), "dataset_mask_area_by_class", R / "tables")

    # figures
    fig, axes = plt.subplots(1, 2, figsize=(10, max(2.5, 0.45 * max(len(per_src), len(classes)) + 1)))
    bar(axes[0], per_src.index.tolist(), per_src["total"].tolist(), "#2a78d6", "Images per source", "images")
    counts = inst["cls"].value_counts().reindex(classes, fill_value=0)
    bar(axes[1], classes, counts.tolist(), [CLASS_COLORS[c] for c in classes], "Instances per class", "instances")
    savefig(fig, R / "figures" / "dataset_counts.png")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    data = [np.log10(inst.loc[inst["cls"] == c, "area_ratio"].clip(lower=1e-6)) for c in classes]
    bp = ax.boxplot(data, vert=False, widths=0.5, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], classes):
        patch.set(facecolor=CLASS_COLORS[c], alpha=0.6, edgecolor=CLASS_COLORS[c])
    ax.set_yticks(range(1, len(classes) + 1), classes)
    ax.set_xlabel("log10(mask area / image area)")
    ax.set_title("Mask-area distribution per class (whiskers 1.5 IQR, outliers hidden)", loc="left")
    savefig(fig, R / "figures" / "dataset_mask_area.png")

    # sample grid with masks (random, seed 42, at most one image per source per row cycle)
    rng = random.Random(SEED)
    files = man.groupby("source")["file"].apply(list).to_dict()
    picks, srcs = [], sorted(files)
    while len(picks) < min(args.grid, len(man)):
        for s in srcs:
            if files[s] and len(picks) < args.grid:
                picks.append((s, files[s].pop(rng.randrange(len(files[s])))))
    cols = 4
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows), squeeze=False)
    split_of = dict(zip(man["file"], man["split"]))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (s, f) in zip(axes.ravel(), picks):
        img = Image.open(d / "images" / split_of[f] / f).convert("RGB")
        w, h = img.size
        ax.imshow(img)
        for line in (d / "labels" / split_of[f] / (Path(f).stem + ".txt")).read_text().splitlines():
            t = line.split()
            c = CLASSES[int(t[0])]
            pts = np.array(t[1:], float).reshape(-1, 2) * [w, h]
            ax.add_patch(plt.Polygon(pts, closed=True, facecolor=CLASS_COLORS[c], alpha=0.45,
                                     edgecolor=CLASS_COLORS[c], linewidth=1))
        ax.set_title(s, fontsize=8, color=INK2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLORS[c]) for c in CLASSES]
    fig.legend(handles, CLASSES, loc="lower center", ncol=5)
    savefig(fig, R / "figures" / "dataset_sample_grid.png")

    save_json({"images": len(man), "instances": len(inst), "sources": per_src["total"].to_dict(),
               "instances_per_class": counts.to_dict(),
               "split_images": man["split"].value_counts().to_dict(),
               "median_area_ratio_per_class": inst.groupby("cls")["area_ratio"].median().to_dict()},
              R / "metrics" / "dataset_summary.json")
    print(f"[DONE] dataset figures/tables -> {R}")


if __name__ == "__main__":
    main()
