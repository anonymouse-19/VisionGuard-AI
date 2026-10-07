"""A2: Train a YOLO segmentation model on data/processed (run on a GPU machine / Colab).

Usage:
  python scripts/train.py --model yolo11n-seg --smoke     # 1 epoch on 50 images (pipeline check)
  python scripts/train.py --model yolo11n-seg             # full run: 60 epochs, patience 15
  python scripts/train.py --model yolo11s-seg --batch -1  # auto batch size
Outputs: runs/segment/<model>/ (weights/best.pt, results.csv, plots) and train_config.json there.
"""
import argparse
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, load_yaml, rpath, save_json  # noqa: E402
from src.vision.data import resolve_data_yaml, sample, split_images, subset_yaml  # noqa: E402

MODELS = ["yolo11n-seg", "yolo11s-seg", "yolov8n-seg"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--data", default=None, help="override data yaml")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--device", default=None, help="e.g. 0, cpu (default: auto)")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--name", default=None, help="run name (default: model name)")
    ap.add_argument("--smoke", action="store_true", help="1 epoch on 50 training images")
    args = ap.parse_args()

    import torch
    import ultralytics
    from ultralytics import YOLO
    from src.vision.yolo_aug import install_brightness_contrast

    cfg = load_yaml(args.config)
    data_yaml = resolve_data_yaml(args.data or cfg["data"])
    name = args.name or args.model
    epochs, batch = args.epochs or cfg["epochs"], args.batch or cfg["batch"]
    if args.smoke:
        s = cfg["smoke"]
        data_yaml = subset_yaml(data_yaml, f"runs/smoke/{args.model}", {
            "train": sample(split_images(data_yaml, "train"), s["n_train"], cfg["seed"]),
            "val": sample(split_images(data_yaml, "val"), s["n_val"], cfg["seed"])})
        epochs, batch, name = s["epochs"], args.batch or s["batch"], f"{name}_smoke"

    bc_ok = install_brightness_contrast(**cfg["brightness_contrast"])
    if not bc_ok:
        print("[WARN] could not install BrightnessContrast; training continues with hsv_v brightness only")
    train_args = dict(data=str(data_yaml), imgsz=cfg["imgsz"], epochs=epochs, patience=cfg["patience"],
                      batch=batch, workers=args.workers or cfg["workers"], seed=cfg["seed"],
                      deterministic=cfg["deterministic"], optimizer=cfg["optimizer"], cos_lr=cfg["cos_lr"],
                      project=str(rpath(cfg["project"])), name=name, exist_ok=True, plots=True,
                      **cfg["augment"])
    if args.device is not None:
        train_args["device"] = args.device

    log = {"model": args.model, "run_name": name, "smoke": args.smoke, "train_args": train_args,
           "brightness_contrast": {**cfg["brightness_contrast"], "installed": bc_ok},
           "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor(),
           "torch": torch.__version__, "ultralytics": ultralytics.__version__}
    print("[INFO] batch size requested:", batch, "| augmentations:", cfg["augment"], cfg["brightness_contrast"])
    t0 = time.time()
    model = YOLO(f"{args.model}.pt")  # downloads COCO-pretrained weights on first use
    model.train(**train_args)
    tr = model.trainer
    log.update(batch_size_used=getattr(tr, "batch_size", batch), train_time_min=round((time.time() - t0) / 60, 2),
               best_weights=str(Path(tr.best).relative_to(ROOT)) if Path(tr.best).is_relative_to(ROOT) else str(tr.best),
               epochs_completed=getattr(tr, "epoch", -1) + 1)
    save_json(log, Path(tr.save_dir) / "train_config.json")
    save_json(log, f"results/logs/train_{name}.json")
    print(f"[DONE] {name}: best weights -> {tr.best}")


if __name__ == "__main__":
    main()
