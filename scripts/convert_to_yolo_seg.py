"""A1: Convert public proxy defect datasets in data/raw/<source>/ into ONE YOLO-seg dataset.

PROXY DATA NOTICE: sources are public civil / industrial / automotive defect datasets used as a
stand-in for defence-equipment imagery, which is not available to this project.

Formats auto-detected per source folder (override with `format:` in configs/datasets.yaml):
  severstal_rle  CSV with an EncodedPixels column (Kaggle Severstal style)
  coco           JSON with images / annotations / categories (polygon or RLE segmentation)
  yolo_seg       labels/*.txt polygon rows "cls x1 y1 x2 y2 ..." (+ data.yaml names)
  mask_png       image files paired with mask images by file stem
Outputs in data/processed/: images/{train,val,test}, labels/{train,val,test}, data.yaml,
manifest.csv (one row per image), instances.csv (one row per polygon), conversion_report.json.

Usage: python scripts/convert_to_yolo_seg.py [--config configs/datasets.yaml] [--overwrite]
"""
import argparse
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import load_yaml, rpath, save_json  # noqa: E402
from src.vision.masks import mask_to_polygons, polygon_area, rle_decode_coco, rle_decode_severstal  # noqa: E402

IMG_EXT = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
MASK_TOKENS = {"mask", "masks", "gt", "groundtruth", "ground", "label", "labels", "lab",
               "annotation", "annotations", "segmentationclass", "seg"}
MASK_SUFFIXES = ("_mask", "-mask", "_gt", "_label", "_lab", "_seg")
LICENSE_RE = re.compile(r"(CC[- ]BY[A-Za-z\-]*[ -]?\d\.\d|CC0[ -]?1\.0|MIT License|Apache[- ]2\.0|AGPL[\w\-.]*|"
                        r"GPL[\w\-.]*|public domain|non[- ]commercial)", re.I)


@dataclass
class Record:
    image: Path
    w: int
    h: int
    polys: list = field(default_factory=list)  # [(class_name, (N, 2) normalised array)]


class Source:
    """Per-source context: config, counters and helpers shared by all parsers."""

    def __init__(self, name, path, cfg, gcfg):
        self.name, self.path, self.cfg, self.g = name, path, cfg or {}, gcfg
        self.stats, self.class_counts = Counter(), Counter()
        self._img_index = None

    def warn(self, msg):
        print(f"[WARN] {self.name}: {msg}")

    def resolve_class(self, raw):
        """raw class name/id -> unified class via class_map, keyword_map, default_class (None = drop)."""
        if raw is None:
            return None
        key = str(raw).strip().lower()
        cmap = {str(k).strip().lower(): v for k, v in (self.cfg.get("class_map") or {}).items()}
        if key in cmap:
            out = cmap[key]
        else:
            out = next((c for c, kws in self.g["keyword_map"].items() if any(k in key for k in kws)), None)
            out = out or self.cfg.get("default_class")
        if out is not None and out not in self.g["classes"]:
            raise ValueError(f"{self.name}: '{raw}' mapped to unknown class '{out}'")
        self.class_counts[f"{raw} -> {out}"] += 1
        return out

    @staticmethod
    def image_size(p):
        with Image.open(p) as im:
            return im.size  # (w, h)

    def find_image(self, name, near=None):
        for base in ([near] if near else []) + [self.path]:
            for cand in (base / name, base / "images" / name):
                if cand.is_file():
                    return cand
        if self._img_index is None:
            self._img_index = defaultdict(list)
            for p in self.path.rglob("*"):
                if p.suffix.lower() in IMG_EXT:
                    self._img_index[p.name.lower()].append(p)
        hits = self._img_index.get(Path(name).name.lower(), [])
        return hits[0] if hits else None

    def add_poly(self, rec, cls, pts_px):
        """Validate a pixel-coordinate polygon; append it normalised and clipped to [0, 1]."""
        if cls is None:
            self.stats["inst_unmapped_class"] += 1
            return
        pts = np.asarray(pts_px, dtype=float).reshape(-1, 2) / [rec.w, rec.h]
        if not np.isfinite(pts).all():
            self.stats["inst_invalid"] += 1
            return
        pts = np.clip(pts, 0.0, 1.0)
        pts = pts[np.r_[True, np.any(np.diff(pts, axis=0) != 0, axis=1)]]  # drop repeated vertices
        if len(pts) < 3 or polygon_area(pts) * rec.w * rec.h < self.g["min_instance_area_px"]:
            self.stats["inst_too_small_or_degenerate"] += 1
            return
        maxp = self.g["max_polygon_points"]
        if len(pts) > maxp:
            pts = pts[np.linspace(0, len(pts) - 1, maxp).astype(int)]
        rec.polys.append((cls, pts))

    def masks_to_polys(self, rec, cls, mask):
        if cls is None:
            self.stats["inst_unmapped_class"] += 1
            return
        for pts in mask_to_polygons(mask, self.g["min_instance_area_px"], self.g["max_polygon_points"]):
            self.add_poly(rec, cls, pts)


# ----------------------------------------------------------------------------- format detection
def _severstal_csvs(path):
    out = []
    for p in path.rglob("*.csv"):
        if "sample" in p.name.lower() or "submission" in p.name.lower():
            continue  # Kaggle sample_submission.csv also has an EncodedPixels column
        with open(p, encoding="utf-8", errors="ignore") as f:
            if "EncodedPixels" in f.readline():
                out.append(p)
    return sorted(out, key=lambda p: p.name.lower() != "train.csv")


def _load_coco(p):
    try:
        import json
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return None
    ok = isinstance(d, dict) and all(k in d for k in ("images", "annotations", "categories"))
    return d if ok else None


def _is_mask_file(p, root):
    if p.stem.lower().endswith(MASK_SUFFIXES):
        return True
    toks = set()
    for part in p.relative_to(root).parts[:-1]:
        toks.update(re.split(r"[_\-\s.]+", part.lower()))
    return bool(toks & MASK_TOKENS)


def _is_yolo_seg_label(p):
    try:
        rows = [r.split() for r in p.read_text(errors="ignore").splitlines() if r.strip()]
    except OSError:
        return False
    return any(len(r) >= 7 and len(r) % 2 == 1 for r in rows[:50])  # box-only rows may be mixed in


def detect_format(src):
    fmt = src.cfg.get("format", "auto")
    if fmt != "auto":
        return fmt
    if _severstal_csvs(src.path):
        return "severstal_rle"
    files = [p for p in src.path.rglob("*") if p.is_file()]
    if any(_load_coco(p) for p in files if p.suffix.lower() == ".json"):
        return "coco"
    if any(_is_yolo_seg_label(p) for p in files
           if p.suffix == ".txt" and "labels" in [x.lower() for x in p.parts]):
        return "yolo_seg"
    if any(_is_mask_file(p, src.path) for p in files if p.suffix.lower() in IMG_EXT):
        return "mask_png"
    return None


# ----------------------------------------------------------------------------- parsers
def parse_severstal(src):
    csv_path = _severstal_csvs(src.path)[0]
    df = pd.read_csv(csv_path)
    if "ImageId_ClassId" in df.columns:  # 2019 competition format
        df[["ImageId", "ClassId"]] = df["ImageId_ClassId"].str.rsplit("_", n=1, expand=True)
    df = df.dropna(subset=["EncodedPixels"])
    for img_id, g in df.groupby("ImageId"):
        p = src.find_image(img_id, near=csv_path.parent / "train_images")
        if p is None:
            src.stats["img_missing_file"] += 1
            continue
        w, h = src.image_size(p)
        rec = Record(p, w, h)
        for _, r in g.iterrows():
            cls = src.resolve_class(str(r["ClassId"]).split(".")[0])
            src.masks_to_polys(rec, cls, rle_decode_severstal(r["EncodedPixels"], h, w) if cls else None)
        yield rec


def parse_coco(src):
    for jf in sorted(src.path.rglob("*.json")):
        d = _load_coco(jf)
        if d is None:
            continue
        cats = {c["id"]: c["name"] for c in d["categories"]}
        anns = defaultdict(list)
        for a in d["annotations"]:
            anns[a["image_id"]].append(a)
        for im in d["images"]:
            p = src.find_image(im["file_name"], near=jf.parent)
            if p is None:
                src.stats["img_missing_file"] += 1
                continue
            w, h = im.get("width") or 0, im.get("height") or 0
            if not (w and h):
                w, h = src.image_size(p)
            rec = Record(p, int(w), int(h))
            for a in anns.get(im["id"], []):
                cls = src.resolve_class(cats.get(a["category_id"]))
                seg = a.get("segmentation")
                if cls is None:
                    src.stats["inst_unmapped_class"] += 1
                elif isinstance(seg, list) and seg:
                    for s in seg:
                        src.add_poly(rec, cls, s)
                elif isinstance(seg, dict):
                    m = rle_decode_coco(seg, rec.h, rec.w)
                    if m is None:
                        src.stats["inst_rle_needs_pycocotools"] += 1
                    else:
                        src.masks_to_polys(rec, cls, m)
                else:
                    src.stats["inst_box_only"] += 1
            yield rec


def _yolo_names(src):
    for y in sorted(src.path.rglob("*.y*ml")):
        try:
            d = yaml.safe_load(y.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        if isinstance(d, dict) and "names" in d:
            n = d["names"]
            return dict(enumerate(n)) if isinstance(n, list) else {int(k): v for k, v in n.items()}
    src.warn("no data.yaml with 'names'; raw class ids are used as class names")
    return {}


def _yolo_image_for_label(lf, src):
    rel = list(lf.relative_to(src.path).parts)
    rel[max(i for i, p in enumerate(rel) if p.lower() == "labels")] = "images"
    stem_dir, stem = src.path.joinpath(*rel[:-1]), Path(rel[-1]).stem
    for d in (stem_dir, lf.parent):
        for ext in IMG_EXT + [e.upper() for e in IMG_EXT]:
            if (d / (stem + ext)).is_file():
                return d / (stem + ext)
    return None


def parse_yolo_seg(src):
    names = _yolo_names(src)
    for lf in sorted(src.path.rglob("*.txt")):
        if "labels" not in [x.lower() for x in lf.parts]:
            continue
        img = _yolo_image_for_label(lf, src)
        if img is None:
            src.stats["img_missing_file"] += 1
            continue
        w, h = src.image_size(img)
        rec = Record(img, w, h)
        for line in lf.read_text(errors="ignore").splitlines():
            t = line.split()
            if not t:
                continue
            if len(t) == 5:
                src.stats["inst_box_only"] += 1
                continue
            try:
                assert len(t) >= 7 and len(t) % 2 == 1
                cid, coords = int(float(t[0])), np.array(t[1:], dtype=float).reshape(-1, 2)
            except (AssertionError, ValueError):
                src.stats["inst_invalid"] += 1
                continue
            src.add_poly(rec, src.resolve_class(names.get(cid, cid)), coords * [w, h])
        yield rec


def _norm_stem(s):
    s = s.lower()
    for suf in MASK_SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s


def parse_mask_png(src):
    files = [p for p in src.path.rglob("*") if p.suffix.lower() in IMG_EXT]
    exts = defaultdict(set)
    for p in files:
        exts[(p.parent, p.stem.lower())].add(p.suffix.lower())
    masks, images = [], []
    for p in files:
        png_beside_jpg = p.suffix.lower() == ".png" and exts[(p.parent, p.stem.lower())] & {".jpg", ".jpeg"}
        (masks if (_is_mask_file(p, src.path) or png_beside_jpg) else images).append(p)
    idx = defaultdict(list)
    for p in images:
        idx[p.stem.lower()].append(p)
    pairs = defaultdict(list)
    for m in masks:
        cands = idx.get(_norm_stem(m.stem), [])
        if not cands:
            src.stats["mask_without_image"] += 1
            continue
        img = max(cands, key=lambda c: (c.parent.name.lower() == m.parent.name.lower(),
                                        len(set(c.parts) & set(m.parts))))
        pairs[img].append(m)
    src.stats["img_without_mask"] += len(images) - len(pairs)
    mask_values = src.cfg.get("mask_values")
    for img, mlist in pairs.items():
        w, h = src.image_size(img)
        rec = Record(img, w, h)
        for mp in mlist:
            with Image.open(mp) as mi:
                mi = mi.convert("L")
                if mi.size != (w, h):
                    mi = mi.resize((w, h), Image.NEAREST)
                    src.stats["mask_resized_to_image"] += 1
                m = np.asarray(mi)
            if mask_values:
                for v, cls in mask_values.items():
                    if cls not in src.g["classes"]:
                        raise ValueError(f"{src.name}: mask value {v} -> unknown class {cls}")
                    src.masks_to_polys(rec, cls, m == int(v))
            else:
                cls = (src.cfg.get("mask_class") or src.resolve_class(mp.parent.name)
                       or src.resolve_class(src.name))
                src.masks_to_polys(rec, cls, m > (127 if m.max() > 127 else 0))
        yield rec


PARSERS = {"severstal_rle": parse_severstal, "coco": parse_coco,
           "yolo_seg": parse_yolo_seg, "mask_png": parse_mask_png}


# ----------------------------------------------------------------------------- helpers
def license_hint(path):
    files, found = [], None
    for p in sorted(path.rglob("*")):
        if p.is_file() and len(p.relative_to(path).parts) <= 3 and \
                any(h in p.name.lower() for h in ("license", "licence", "readme")):
            files.append(str(p.relative_to(path)))
            if found is None:
                m = LICENSE_RE.search(p.read_text(encoding="utf-8", errors="ignore")[:20000])
                found = m.group(0) if m else None
    return {"files": files, "detected": found or "check manually"}


def cap_sizes(sizes, frac):
    """Largest sources are reduced so none exceeds `frac` of the total (relaxed to 1/n if n < 1/frac)."""
    n = len(sizes)
    f = max(frac, 1.0 / n) if n else frac
    caps = dict(sizes)
    for _ in range(20):
        total = sum(caps.values())
        over = [k for k, v in caps.items() if v > f * total + 1e-9]
        if not over:
            break
        others = total - sum(caps[k] for k in over)
        c = int(f * others / max(1e-9, 1 - len(over) * f))
        for k in over:
            caps[k] = min(caps[k], c)
    return caps, f


def safe_stem(source, rel):
    s = "_".join(Path(rel).with_suffix("").parts)
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", s)[-150:]
    return f"{source}__{s}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/datasets.yaml")
    ap.add_argument("--overwrite", action="store_true", help="delete an existing data/processed first")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    raw, out = rpath(cfg["raw_dir"]), rpath(cfg["out_dir"])
    classes, scfgs = cfg["classes"], cfg.get("sources") or {}

    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            sys.exit(f"{out} is not empty. Re-run with --overwrite to rebuild it.")
        shutil.rmtree(out)
    if not raw.is_dir():
        sys.exit(f"No raw data folder at {raw}. Download sources into {raw}/<source>/ first (see docs/DATASETS.md).")

    present = {p.name: p for p in sorted(raw.iterdir()) if p.is_dir()}
    for name in scfgs:
        if name not in present:
            print(f"[WARN] {name}: configured but not found in {raw} - skipped")

    report, kept = {"sources": {}}, {}
    for name, path in present.items():
        src = Source(re.sub(r"_{2,}", "_", name), path, scfgs.get(name), cfg)
        fmt = detect_format(src)
        entry = {"format": fmt, "license": license_hint(path)}
        if fmt in (None, "skip"):
            reason = "format=skip in config (no masks)" if fmt == "skip" else "no supported annotation format found"
            src.warn(f"skipped: {reason}")
            entry["skipped"] = reason
            report["sources"][name] = entry
            continue
        recs, n_parsed = [], 0
        for rec in PARSERS[fmt](src):
            n_parsed += 1
            if rec.polys:
                recs.append(rec)
            else:
                src.stats["img_dropped_empty_after_filters"] += 1
        print(f"[INFO] {src.name}: format={fmt} parsed={n_parsed} kept={len(recs)}")
        entry.update(images_parsed=n_parsed, images_with_valid_masks=len(recs),
                     counters=dict(src.stats), class_mapping=dict(src.class_counts))
        report["sources"][name] = entry
        if recs:
            kept[src.name] = (name, path, sorted(recs, key=lambda r: str(r.image)))

    if not kept:
        sys.exit("No usable images found in any source.")
    caps, f_used = cap_sizes({k: len(v[2]) for k, v in kept.items()}, cfg["max_source_fraction"])
    if f_used > cfg["max_source_fraction"]:
        print(f"[WARN] only {len(kept)} source(s): per-source cap relaxed to {f_used:.0%}")

    sp = cfg["split"]
    manifest, instances, used = [], [], set()
    for split in sp:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    for sname, (folder, path, recs) in kept.items():
        rng = random.Random(f"{cfg['seed']}-{sname}")
        if caps[sname] < len(recs):
            print(f"[INFO] {sname}: capped {len(recs)} -> {caps[sname]} images")
            recs = sorted(rng.sample(recs, caps[sname]), key=lambda r: str(r.image))
        rng.shuffle(recs)
        n_tr, n_va = round(sp["train"] * len(recs)), round(sp["val"] * len(recs))
        splits = {"train": recs[:n_tr], "val": recs[n_tr:n_tr + n_va], "test": recs[n_tr + n_va:]}
        report["sources"][folder].update(images_after_cap=len(recs),
                                         split_counts={k: len(v) for k, v in splits.items()})
        for split, rs in splits.items():
            for rec in rs:
                stem = safe_stem(sname, rec.image.relative_to(path))
                while stem in used:
                    stem += "_x"
                used.add(stem)
                fname = stem + rec.image.suffix.lower()
                shutil.copy2(rec.image, out / "images" / split / fname)
                lines = [f"{classes.index(c)} " + " ".join(f"{v:.6f}" for v in pts.ravel()) for c, pts in rec.polys]
                (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
                manifest.append(dict(file=fname, source=sname, split=split, width=rec.w, height=rec.h,
                                     n_instances=len(rec.polys), original=str(rec.image.relative_to(raw))))
                for c, pts in rec.polys:
                    instances.append(dict(file=fname, source=sname, split=split, cls=c,
                                          area_ratio=polygon_area(pts), n_points=len(pts)))

    (out / "data.yaml").write_text(
        "# Generated by scripts/convert_to_yolo_seg.py. An absolute 'path' is added at train/eval time\n"
        "# (src/vision/data.py) so the folder can be moved between machines.\n"
        + yaml.safe_dump({"train": "images/train", "val": "images/val", "test": "images/test",
                          "names": dict(enumerate(classes))}, sort_keys=False), encoding="utf-8")
    pd.DataFrame(manifest).to_csv(out / "manifest.csv", index=False)
    inst = pd.DataFrame(instances)
    inst.to_csv(out / "instances.csv", index=False)
    report.update(total_images=len(manifest), total_instances=len(instances),
                  instances_per_class=inst["cls"].value_counts().to_dict(),
                  per_source_cap_fraction=f_used, seed=cfg["seed"], split=sp)
    save_json(report, out / "conversion_report.json")
    print(f"[DONE] {len(manifest)} images, {len(instances)} instances -> {out}")


if __name__ == "__main__":
    main()
