# Datasets (public proxies)

**No real defence-equipment imagery is used.** Public civil, industrial and automotive defect datasets stand in for it, so the model learns *proxy* defect appearance. Results do not transfer automatically to defence assets.

Put each source in `data/raw/<folder>/`. The folder name selects its block in `configs/datasets.yaml`. Unknown folders are still converted (`format: auto` plus keyword class mapping). `scripts/convert_to_yolo_seg.py` records the detected format, the class-mapping counts, dropped instances and any licence text it finds in `data/processed/conversion_report.json`.

Status on 2026-10-07: **no dataset has been downloaded yet.** The licence column must be confirmed from each source's own README/licence once downloaded. Hints marked *unverified* are from memory, not from files.

| Folder (suggested) | Source | Format | Raw class → unified class | Licence |
|---|---|---|---|---|
| `crack_seg` | Ultralytics / Roboflow crack-seg | YOLO-seg | crack → crack | check manually |
| `crack500` | Crack500 | mask PNG beside JPG | foreground → crack | check manually |
| `deepcrack` | DeepCrack | mask PNG (`*_lab/`) | foreground → crack | check manually |
| `<any corrosion set>` | Roboflow corrosion/rust segmentation | COCO or YOLO-seg | corrosion/rust → corrosion (keyword) | check manually (Roboflow sets are often CC BY 4.0, *unverified*) |
| `<any oil-leak set>` | Roboflow oil-leak / oil-spill segmentation | COCO or YOLO-seg | oil/leak/spill → oil_leak (keyword) | check manually |
| `severstal` | Kaggle Severstal Steel Defect Detection | RLE csv | ClassId 1–4 → surface_wear (organisers do not name the classes) | check manually (Kaggle competition rules, *unverified*) |
| `cardd` | CarDD | COCO polygons | dent, glass shatter, lamp broken, tire flat → damage; scratch → surface_wear; crack → crack | check manually (research-only, *unverified*) |
| `mvtec_ad` | MVTec AD | `ground_truth/<defect>/*_mask.png` | crack → crack; scratch → surface_wear; other defect folders → damage | check manually (CC BY-NC-SA 4.0, *unverified*) |
| `sdnet2018` | SDNET2018 | image-level labels only | **skipped: no masks** | n/a |
| `neu_det` | NEU-DET | bounding boxes only | **skipped: no masks** | n/a |

## Conversion rules
- Unified classes (YOLO ids 0–4): `crack, corrosion, oil_leak, surface_wear, damage`.
- A class name is mapped by the source's `class_map` first, then `keyword_map`, then `default_class`. If none match, the instance is **dropped and counted**, never guessed.
- Each connected component of a mask becomes one polygon (8-connected, outer boundary only, holes ignored). Masks 1–2 px wide are dilated by 1 px so they don't trace to zero-area polygons.
- The converter drops polygons under 30 px or with fewer than 3 distinct vertices, box-only rows, and images with no valid instance left. Healthy/background images are therefore excluded.
- No single source may exceed about 40% of the images. The cap relaxes to 1/n when there are fewer than 3 sources, with a warning.
- The 70/15/15 split is done per source (seed 42), so each split has every source in proportion. Images are renamed `<source>__<original path>`, which is how per-source AP is computed.
