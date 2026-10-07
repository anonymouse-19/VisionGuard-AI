"""Inference wrapper: image -> instance masks at original resolution + severity assessment."""
from pathlib import Path

import numpy as np

from src.common import rpath


def load_model(weights):
    """Return (model, None) or (None, reason) so callers can show a clear message."""
    w = rpath(weights)
    if not w.is_file():
        return None, f"weights not found at {w}"
    try:
        from ultralytics import YOLO
    except ImportError:
        return None, "ultralytics is not installed (pip install -r requirements.txt)"
    return YOLO(str(w)), None


def predict(model, image, conf=0.25, imgsz=640, device=None):
    """image: path or HxWx3 RGB array. Returns dict(masks (N,H,W) bool, classes [str], confs, plot RGB)."""
    src = str(image) if isinstance(image, (str, Path)) else np.ascontiguousarray(np.asarray(image)[..., ::-1])  # RGB->BGR
    kw = {"conf": conf, "imgsz": imgsz, "retina_masks": True, "verbose": False}
    if device is not None:
        kw["device"] = device
    r = model.predict(src, **kw)[0]
    h, w = r.orig_shape
    if r.masks is None or len(r.boxes) == 0:
        masks, cls_ids, confs = np.zeros((0, h, w), bool), [], np.zeros(0)
    else:
        masks = r.masks.data.cpu().numpy() > 0.5
        cls_ids = r.boxes.cls.cpu().numpy().astype(int).tolist()
        confs = r.boxes.conf.cpu().numpy()
    return {"masks": masks, "classes": [r.names[c] for c in cls_ids], "confs": confs,
            "boxes": r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4)),
            "plot": r.plot()[..., ::-1], "shape": (h, w)}
