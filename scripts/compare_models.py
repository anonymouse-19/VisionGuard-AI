"""A2: Compare evaluated models (reads results/metrics/<model>.json and runs/segment/<model>/results.csv).

Usage: python scripts/compare_models.py [--models yolo11n-seg yolo11s-seg]
Outputs: results/tables/model_comparison.csv|md, results/figures/model_accuracy_vs_speed.png,
         results/figures/training_curves_loss.png, results/figures/training_curves_map.png
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import rpath, save_table  # noqa: E402
from src.plotting import INK2, SERIES, plt, savefig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="default: every results/metrics/*.json with box+mask keys")
    ap.add_argument("--runs", default="runs/segment")
    args = ap.parse_args()

    metrics = {}
    for f in sorted(rpath("results/metrics").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if "box" in d and "mask" in d and (not args.models or d["model"] in args.models):
            metrics[d["model"]] = d
    if not metrics:
        sys.exit("no model metrics found - run scripts/evaluate.py first")

    rows = [{"model": k, "params_M": d["params"] / 1e6, "gflops": d.get("gflops"),
             "box_mAP50": d["box"]["map50"], "box_mAP50_95": d["box"]["map50_95"],
             "mask_mAP50": d["mask"]["map50"], "mask_mAP50_95": d["mask"]["map50_95"],
             "mask_precision": d["mask"]["precision"], "mask_recall": d["mask"]["recall"],
             "latency_ms": d["latency_ms_batch1"]["mean"], "fps": d["fps_batch1"], "hardware": d["hardware"]}
            for k, d in metrics.items()]
    table = pd.DataFrame(rows).sort_values("latency_ms")
    save_table(table, "model_comparison")

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, r in enumerate(table.itertuples()):
        ax.scatter(r.latency_ms, r.mask_mAP50_95, s=60 + 25 * r.params_M, color=SERIES[i % len(SERIES)],
                   edgecolor="white", linewidth=2, zorder=3)
        ax.annotate(f"{r.model}\n{r.params_M:.1f}M params", (r.latency_ms, r.mask_mAP50_95),
                    textcoords="offset points", xytext=(8, 4), fontsize=8, color=INK2)
    ax.set_xlabel(f"batch-1 latency, ms ({table['hardware'].iloc[0]})")
    ax.set_ylabel("mask mAP50-95 (test)")
    ax.set_title("Accuracy vs speed", loc="left")
    savefig(fig, "results/figures/model_accuracy_vs_speed.png")

    curves = {}
    for k in metrics:
        f = rpath(args.runs) / k / "results.csv"
        if f.is_file():
            df = pd.read_csv(f)
            df.columns = [c.strip() for c in df.columns]
            curves[k] = df
        else:
            print(f"[WARN] {f} missing - no training curves for {k}")
    if curves:
        specs = {"loss": [("train/seg_loss", "-"), ("val/seg_loss", "--")],
                 "map": [("metrics/mAP50(M)", "--"), ("metrics/mAP50-95(M)", "-")]}
        for kind, cols in specs.items():
            fig, ax = plt.subplots(figsize=(6.5, 4))
            for i, (k, df) in enumerate(curves.items()):
                for col, ls in cols:
                    if col in df:
                        ax.plot(df["epoch"], df[col], ls, color=SERIES[i % len(SERIES)], label=f"{k} {col}")
            ax.set_xlabel("epoch")
            ax.set_title("Segmentation loss (solid train, dashed val)" if kind == "loss"
                         else "Validation mask mAP (solid 50-95, dashed 50)", loc="left")
            ax.legend(fontsize=7)
            savefig(fig, f"results/figures/training_curves_{kind}.png")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
