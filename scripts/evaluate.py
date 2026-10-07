"""A2: Evaluate trained weights on the TEST split (run on the GPU machine after train.py).

Reports box+mask mAP50 / mAP50-95, precision, recall, per-class AP, per-source AP, confusion matrix,
PR curves (Ultralytics plots), latency/FPS on the stated hardware and parameter count.
Usage: python scripts/evaluate.py --model yolo11n-seg [--weights runs/segment/yolo11n-seg/weights/best.pt]
Outputs: results/metrics/<model>.json, results/tables/<model>_per_class.csv|md, <model>_per_source.csv|md,
         results/figures/<model>/*.png (confusion matrix, PR/F1 curves), <model>_confusion_matrix.csv
"""
import argparse
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import load_yaml, rpath, save_json, save_table  # noqa: E402
from src.vision.data import resolve_data_yaml, sample, source_of, split_images, subset_yaml  # noqa: E402


def summarise(m):
    b, s = m.box, m.seg
    return {"box": {"map50": b.map50, "map50_95": b.map, "precision": b.mp, "recall": b.mr},
            "mask": {"map50": s.map50, "map50_95": s.map, "precision": s.mp, "recall": s.mr}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="run name, e.g. yolo11n-seg")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-timing", type=int, default=100, help="images for the batch-1 latency benchmark")
    ap.add_argument("--min-source-images", type=int, default=5)
    args = ap.parse_args()

    import torch
    import ultralytics
    from ultralytics import YOLO

    cfg = load_yaml(args.config)
    weights = rpath(args.weights or f"{cfg['project']}/{args.model}/weights/best.pt")
    if not weights.is_file():
        sys.exit(f"weights not found: {weights}")
    data_yaml = resolve_data_yaml(cfg["data"])
    model = YOLO(str(weights))
    names = model.names
    dev = {"device": args.device} if args.device is not None else {}
    common = dict(imgsz=cfg["imgsz"], batch=args.batch, verbose=False, **dev)

    # 1) full test split
    out_dir = rpath(f"runs/eval/{args.model}")
    m = model.val(data=str(data_yaml), split=args.split, plots=True, project=str(out_dir.parent),
                  name=out_dir.name, exist_ok=True, **common)
    res = {"model": args.model, "weights": str(weights), "split": args.split, **summarise(m)}
    res["speed_ms_per_image_val"] = m.speed

    idx = list(getattr(m, "ap_class_index", m.box.ap_class_index))
    rows = []
    for j, ci in enumerate(idx):
        rows.append({"class": names[int(ci)], "box_ap50": m.box.ap50[j], "box_ap50_95": m.box.ap[j],
                     "mask_ap50": m.seg.ap50[j], "mask_ap50_95": m.seg.ap[j]})
    per_class = pd.DataFrame(rows)
    save_table(per_class, f"{args.model}_per_class")
    res["per_class"] = per_class.set_index("class").to_dict(orient="index")

    cm_obj = getattr(m, "confusion_matrix", None)
    if cm_obj is not None:  # rows = predicted, cols = true (Ultralytics convention), last = background
        labels = [names[i] for i in range(len(names))] + ["background"]
        pd.DataFrame(cm_obj.matrix, index=[f"pred_{c}" for c in labels],
                     columns=[f"true_{c}" for c in labels]).to_csv(
            rpath(f"results/tables/{args.model}_confusion_matrix.csv"))
        res["confusion_matrix"] = {"labels": labels, "matrix": cm_obj.matrix}
    fig_dir = rpath(f"results/figures/{args.model}")
    fig_dir.mkdir(parents=True, exist_ok=True)
    for png in Path(getattr(m, "save_dir", out_dir)).glob("*.png"):
        shutil.copy2(png, fig_dir / png.name)

    # 2) per-source AP (images are named <source>__...)
    test_imgs = split_images(data_yaml, args.split)
    res["n_test_images"] = len(test_imgs)
    by_src = {}
    for p in test_imgs:
        by_src.setdefault(source_of(p), []).append(p)
    src_rows = []
    for src, paths in sorted(by_src.items()):
        if len(paths) < args.min_source_images:
            print(f"[WARN] {src}: only {len(paths)} test images - per-source AP skipped")
            continue
        y = subset_yaml(data_yaml, f"runs/eval/{args.model}_src/{src}", {"val": paths})
        ms = model.val(data=str(y), split="val", plots=False, project=str(out_dir.parent),
                       name=f"{args.model}_src_{src}", exist_ok=True, **common)
        s = summarise(ms)
        src_rows.append({"source": src, "n_images": len(paths), "box_map50": s["box"]["map50"],
                         "box_map50_95": s["box"]["map50_95"], "mask_map50": s["mask"]["map50"],
                         "mask_map50_95": s["mask"]["map50_95"]})
    if src_rows:
        save_table(pd.DataFrame(src_rows), f"{args.model}_per_source")
        res["per_source"] = {r["source"]: r for r in src_rows}

    # 3) batch-1 end-to-end latency (pre-process + inference + post-process), warm-up excluded
    timing = sample(test_imgs, args.n_timing, cfg["seed"])
    cuda = torch.cuda.is_available() and args.device != "cpu"
    for p in timing[:10]:
        model.predict(str(p), imgsz=cfg["imgsz"], verbose=False, **dev)
    lat = []
    for p in timing:
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.predict(str(p), imgsz=cfg["imgsz"], verbose=False, **dev)
        if cuda:
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
    lat = np.array(lat)
    res["latency_ms_batch1"] = {"mean": lat.mean(), "p50": np.median(lat), "p95": np.percentile(lat, 95),
                                "n_images": len(lat)}
    res["fps_batch1"] = 1000.0 / lat.mean()
    res["hardware"] = torch.cuda.get_device_name(0) if cuda else (platform.processor() or platform.machine())
    res["params"] = int(sum(p.numel() for p in model.model.parameters()))
    try:
        from ultralytics.utils.torch_utils import get_flops
        res["gflops"] = get_flops(model.model, cfg["imgsz"])
    except Exception:
        res["gflops"] = None
    res["versions"] = {"torch": torch.__version__, "ultralytics": ultralytics.__version__}
    save_json(res, f"results/metrics/{args.model}.json")
    print(f"[DONE] mask mAP50-95={res['mask']['map50_95']:.3f} box mAP50-95={res['box']['map50_95']:.3f} "
          f"latency={res['latency_ms_batch1']['mean']:.1f} ms on {res['hardware']}")


if __name__ == "__main__":
    main()
