"""Extra brightness/contrast augmentation injected into Ultralytics training datasets.

Ultralytics has no contrast knob (hsv_v only scales brightness), so we subclass YOLODataset and insert a
pixel-only transform just before `Format`. Pinned against ultralytics==8.3.x (see requirements.txt):
`ultralytics.data.build.build_yolo_dataset` looks up the module-global `YOLODataset` at call time.
Classes live at module level so they pickle into DataLoader workers.
"""
import random

import numpy as np
from ultralytics.data.dataset import YOLODataset


class BrightnessContrast:
    def __init__(self, p=0.5, brightness=0.15, contrast=0.25):
        self.p, self.brightness, self.contrast = p, brightness, contrast

    def __call__(self, labels):
        if random.random() < self.p:
            alpha = 1.0 + random.uniform(-self.contrast, self.contrast)
            beta = 255.0 * random.uniform(-self.brightness, self.brightness)
            img = labels["img"].astype(np.float32) * alpha + beta
            labels["img"] = np.clip(img, 0, 255).astype(np.uint8)
        return labels

    def __repr__(self):
        return f"BrightnessContrast(p={self.p}, brightness={self.brightness}, contrast={self.contrast})"


class AugYOLODataset(YOLODataset):
    bc_params = {"p": 0.5, "brightness": 0.15, "contrast": 0.25}

    def build_transforms(self, hyp=None):
        t = super().build_transforms(hyp)
        if self.augment:  # train split only; also re-applied when close_mosaic rebuilds transforms
            t.transforms.insert(-1, BrightnessContrast(**self.bc_params))
        return t


def install_brightness_contrast(p, brightness, contrast):
    """Route Ultralytics dataset construction through AugYOLODataset. Returns True on success."""
    import ultralytics.data.build as build
    if not hasattr(build, "YOLODataset"):
        return False
    AugYOLODataset.bc_params = {"p": p, "brightness": brightness, "contrast": contrast}
    build.YOLODataset = AugYOLODataset
    return True
