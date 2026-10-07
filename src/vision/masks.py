"""Mask <-> polygon utilities (numpy/scipy; OpenCV used when installed, contourpy otherwise)."""
import numpy as np
from scipy import ndimage


def polygon_area(pts):
    """Shoelace area of an (N, 2) polygon, in the polygon's units."""
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _subsample(pts, max_points):
    if len(pts) <= max_points:
        return pts
    return pts[np.linspace(0, len(pts) - 1, max_points).astype(int)]


def _outer_contour(comp):
    """Outer boundary (x, y) of a single connected component. Holes are ignored."""
    try:
        import cv2
        cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        return cv2.approxPolyDP(c, 0.5, True).reshape(-1, 2).astype(float)
    except ImportError:
        import contourpy
        lines = contourpy.contour_generator(z=np.pad(comp.astype(float), 1)).lines(0.5)
        if not lines:
            return None
        return max(lines, key=len)[:, :2] - 1.0  # longest line = outer boundary; undo padding


def mask_to_polygons(mask, min_area_px=30, max_points=200):
    """Binary mask (H, W) -> list of (N, 2) pixel-coordinate polygons, one per connected component."""
    lab, _ = ndimage.label(np.asarray(mask) > 0, structure=np.ones((3, 3)))  # 8-connected: diagonal cracks stay whole
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        comp = lab[sl] == i
        if comp.sum() < min_area_px:
            continue
        pts = _outer_contour(comp)
        off = np.array([sl[1].start, sl[0].start], dtype=float)
        if pts is None or polygon_area(pts) < 0.5 * comp.sum():
            # 1-2 px wide structures (thin cracks) trace to ~zero-area polygons: dilate by 1 px first
            pts = _outer_contour(ndimage.binary_dilation(np.pad(comp, 1)))
            off -= 1.0
        if pts is None or len(pts) < 3:
            continue
        out.append(_subsample(pts + off, max_points))
    return out


def polygon_to_mask(pts_norm, h, w):
    """Normalised polygon -> boolean mask (H, W). Uses PIL to avoid an OpenCV dependency."""
    from PIL import Image, ImageDraw
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).polygon([(float(x) * w, float(y) * h) for x, y in pts_norm], fill=1)
    return np.asarray(img, dtype=bool)


def rle_decode_severstal(rle, h, w):
    """Severstal / Kaggle RLE: 1-indexed 'start length' pairs, column-major (top->bottom, then left->right)."""
    s = np.asarray(str(rle).split(), dtype=np.int64)
    flat = np.zeros(h * w, dtype=np.uint8)
    for start, length in zip(s[0::2] - 1, s[1::2]):
        flat[start:start + length] = 1
    return flat.reshape((h, w), order="F")


def rle_decode_coco(seg, h, w):
    """COCO RLE. Uncompressed (list counts) decoded here; compressed strings need pycocotools (else None)."""
    counts = seg.get("counts")
    if isinstance(counts, list):
        flat = np.zeros(h * w, dtype=np.uint8)
        pos, val = 0, 0
        for c in counts:
            flat[pos:pos + c] = val
            pos += c
            val = 1 - val
        return flat.reshape((h, w), order="F")
    try:
        from pycocotools import mask as mask_utils
        return mask_utils.decode({"size": seg["size"], "counts": counts.encode() if isinstance(counts, str) else counts})
    except ImportError:
        return None
