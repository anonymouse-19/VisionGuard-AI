"""A3: defect severity -> equipment Health Score -> maintenance action (rules in configs/severity.yaml)."""
from functools import lru_cache

import numpy as np
import pandas as pd

from src.common import load_yaml


@lru_cache(maxsize=4)
def _load(path):
    return load_yaml(path)


def load_config(path="configs/severity.yaml"):
    return _load(str(path))


def area_factor(area_ratio, k):
    """f(a) = 1 - exp(-k a): monotone, 0 at a=0, saturates towards 1 for large defects."""
    return 1.0 - np.exp(-k * np.clip(np.asarray(area_ratio, dtype=float), 0.0, 1.0))


def location_weight(defect_mask=None, critical_zone=None, cfg=None):
    """w_location = default + (critical - default) * (share of defect pixels inside the critical zone)."""
    wl = (cfg or load_config())["w_location"]
    if defect_mask is None or critical_zone is None:
        return float(wl["default"])
    m = np.asarray(defect_mask, dtype=bool)
    if m.sum() == 0:
        return float(wl["default"])
    overlap = (m & np.asarray(critical_zone, dtype=bool)).sum() / m.sum()
    return float(wl["default"] + (wl["critical"] - wl["default"]) * overlap)


def severity(cls, area_ratio, confidence=1.0, w_location=1.0, cfg=None):
    """S = clip(w_type * f(area_ratio) * w_location * confidence, 0, 1). Scalars or arrays."""
    cfg = cfg or load_config()
    scalar = isinstance(cls, str)
    cls_arr = np.atleast_1d(np.asarray(cls, dtype=object))
    unknown = set(cls_arr) - set(cfg["w_type"])
    if unknown:
        raise ValueError(f"unknown defect class(es): {sorted(unknown)}")
    w = np.array([cfg["w_type"][c] for c in cls_arr], dtype=float)
    s = np.clip(w * area_factor(area_ratio, cfg["k"]) * np.asarray(w_location, float)
                * np.clip(np.asarray(confidence, float), 0, 1), 0.0, 1.0)
    return float(s.ravel()[0]) if scalar and s.size == 1 else s


def health_score(severities):
    """Health = 100 * prod(1 - S_i); 100 with no defects, 0 only if some S_i = 1. Always in [0, 100]."""
    s = np.clip(np.asarray(severities, dtype=float).ravel(), 0.0, 1.0)
    return float(100.0 * np.prod(1.0 - s)) if s.size else 100.0


def maintenance_action(health, cfg=None):
    for rule in (cfg or load_config())["actions"]:
        if health >= rule["min_health"]:
            return rule["action"]
    return (cfg or load_config())["actions"][-1]["action"]


def assess(classes, area_ratios, confidences, w_locations=None, cfg=None):
    """Per-defect severity table + overall health + action."""
    cfg = cfg or load_config()
    n = len(classes)
    wl = np.ones(n) * cfg["w_location"]["default"] if w_locations is None else np.asarray(w_locations, float)
    s = severity(list(classes), area_ratios, confidences, wl, cfg) if n else np.zeros(0)
    table = pd.DataFrame({"class": list(classes), "area_ratio": np.asarray(area_ratios, float),
                          "confidence": np.asarray(confidences, float), "w_location": wl, "severity": s})
    h = health_score(s)
    return table, h, maintenance_action(h, cfg)


def assess_masks(masks, classes, confidences, critical_zone=None, cfg=None):
    """masks: (N, H, W) bool instance masks at image resolution. area_ratio = mask pixels / image pixels."""
    cfg = cfg or load_config()
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3 or len(masks) == 0:
        return assess([], [], [], cfg=cfg)
    area = masks.reshape(len(masks), -1).mean(axis=1)
    wl = [location_weight(m, critical_zone, cfg) for m in masks]
    return assess(classes, area, confidences, wl, cfg)
