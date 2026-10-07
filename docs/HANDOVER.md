# VisionGuard AI: handover (2026-10-07)

**For:** the next developer or AI agent, working on a machine with a GPU and internet access.
**Why a handover:** the original laptop has no GPU, cannot hold large datasets and was not allowed to download or train anything. Everything that runs on CPU was built and run there. Everything that needs data, weights or the heavy libraries is listed below as remaining work.

Pipeline: image → YOLO11-seg defect masks → severity → Health Score → RUL → maintenance action → defect-driven spare-parts forecast → Streamlit dashboard. **Proxy data only:** public defect datasets stand in for defence imagery, and the fleet timelines and spare-parts demand are simulated.

---

## 1. Project rules (keep following them)
1. **Never fabricate numbers.** Every figure or table must come from code that ran and saved its output under `results/`.
2. Seed 42, YAML configs in `configs/`, pinned `requirements.txt`.
3. After each stage, tick `docs/PROGRESS.md`. Add one line to `docs/ISSUES_LOG.md` for every real problem and its fix.
4. Simple tools only: Ultralytics, pandas, statsmodels, prophet, lightgbm, matplotlib, Streamlit. CSV/JSON storage; no API server or database.

## 2. Status by stage

| Stage | What exists | How far it was verified |
|---|---|---|
| A1 data | `scripts/convert_to_yolo_seg.py`, `scripts/dataset_stats.py`, `configs/datasets.yaml`, `docs/DATASETS.md` | **Tested on a synthetic fixture** covering all 4 formats (YOLO-seg, COCO, mask PNG, Severstal RLE), plus a skip source and the 40% cap. Polygon areas matched the drawn shapes. **Not yet run on real data.** |
| A2 train/eval | `scripts/train.py`, `scripts/evaluate.py`, `scripts/compare_models.py`, `src/vision/yolo_aug.py`, `src/vision/infer.py`, `notebooks/train_colab.ipynb` | **Compile-checked only** (ultralytics not installed). Never executed. |
| A3 severity | `src/severity/severity.py`, `configs/severity.yaml`, `tests/test_severity.py` | **Run:** 15/15 tests pass (`python -m pytest -q tests/`). |
| A4 RUL | `scripts/simulate_timelines.py`, `src/severity/rul.py`, `configs/simulation.yaml` | **Synthetic mode run.** `--from-images` mode written but not run (needs weights). |
| A5 forecast | `scripts/run_forecasting.py`, `configs/forecast.yaml` | **Partly run:** only naive, Croston and SBA (statsmodels, prophet, lightgbm missing). SARIMA, SARIMAX+severity, Prophet, LightGBM and LightGBM+severity were skipped, so **the sensitivity table does not exist yet**. |
| A6 dashboard | `src/dashboard/app.py` (navigation + styling), `src/dashboard/views.py` (pages), `.streamlit/config.toml` (theme) | **Run headless** (Streamlit AppTest): Fleet status, Alerts, Degradation & parts and Inspect image all render with no exceptions. The fleet filter gives row counts that match the summary numbers. Inspect image shows the "no model loaded" message as intended. |
| Stage B | — | **Not started** (needs trained weights). Specs are in §5. |

## 3. Real results produced so far (all synthetic; from code that ran)
- **RUL** (`results/tables/rul_summary.md`, `results/metrics/rul_summary.json`): 30 synthetic equipment, 309 inspections, 11 overhauls, 187 evaluation points, none censored.
  - Exponential (damage-growth) fit: MAE **295 d**, median abs error **77 d**.
  - Linear fit: MAE **808 d**, median **815 d**; 38% of linear predictions hit the 1825-day cap.
  - Error falls as more inspections are used (`rul_mae_by_k.*`). Figures: `results/figures/rul_sample_curves.png`, `rul_mae_by_k.png`.
  - The first parameter set was too slow for the 2-year window and was changed once. Both runs are recorded in ISSUES_LOG.
- **Forecasting baselines** (`results/tables/forecast_summary.md`): mean MASE over 6 SKUs and 3 replicates. SBA is best of the three baselines: 0.875 at zero coupling, rising to 1.422 at high coupling. Naive goes from 1.006 to 1.747. The demand uplift adds variance that univariate baselines cannot anticipate. **These files are partial and must be regenerated** once the full requirements are installed.
- Simulation outputs used by the dashboard and forecasting: `results/simulation/*.csv`.

## 4. What is left, in order (exact commands, run from the repo root)

**4.1 Environment**
```bash
pip install -r requirements.txt        # on GPU boxes install the CUDA torch build first (pytorch.org)
python -m pytest -q tests/             # expect 15 passed
```
Pins marked *unverified* in `requirements.txt` were never installed. If one does not resolve, pick the nearest release and log it. ultralytics must be a version that loads checkpoints under torch ≥ 2.6 (`weights_only` default).

**4.2 Data.** Download the sources into `data/raw/<folder>/`, using the suggested folder names in `docs/DATASETS.md` so the per-source config applies. Then:
```bash
python scripts/convert_to_yolo_seg.py --overwrite
python scripts/dataset_stats.py
```
- Read `data/processed/conversion_report.json`: per-source format, kept/dropped counters, class-mapping counts, licence hints. Fix mappings in `configs/datasets.yaml` if anything maps wrongly.
- Fill in the licence column of `docs/DATASETS.md`.
- Aim for at least 3 sources so the 40% cap holds, and check that corrosion and oil_leak are not empty.

**4.3 Training (GPU), locally or with `notebooks/train_colab.ipynb`**
```bash
python scripts/train.py --model yolo11n-seg --smoke     # 1 epoch / 50 images: pipeline check
python scripts/train.py --model yolo11n-seg
python scripts/train.py --model yolo11s-seg
# optional baseline: python scripts/train.py --model yolov8n-seg
```
Watch for `[WARN] could not install BrightnessContrast`. If it appears, training still runs with brightness jitter (hsv_v) only; log it.

**4.4 Evaluation**
```bash
python scripts/evaluate.py --model yolo11n-seg
python scripts/evaluate.py --model yolo11s-seg
python scripts/compare_models.py
```
These write `results/metrics/<model>.json`, per-class and per-source tables, confusion matrix and PR curves (`results/figures/<model>/`), `model_comparison.*`, and the accuracy-vs-speed and training-curve figures.

**4.5 Forecasting with all models:** `python scripts/run_forecasting.py`. This regenerates `forecast_*` and adds `forecast_sensitivity.*`. Report the zero-coupling result honestly: at zero coupling the severity features should give no gain, and a small loss is possible.

**4.6 Dashboard with real weights:** `streamlit run src/dashboard/app.py`. The default weights path is `configs/severity.yaml → inference.weights` (yolo11s-seg best.pt) and can be changed in the sidebar.

**4.7 Optional:** `python scripts/simulate_timelines.py --from-images inspections.csv --weights <best.pt>` with CSV columns `equipment_id,timestamp,image_path`. This mode produces RUL predictions only; there is no ground truth for it.

## 5. Stage B (not written yet; build after 4.4)
- **B1 `scripts/run_severity_experiment.py`** (the key novelty evidence). On the test split, compute:
  - ground-truth severity from GT polygons (`src/vision/masks.polygon_to_mask`, confidence = 1);
  - predicted mask-area severity (`src/vision/infer.predict` + `src/severity.assess_masks`);
  - predicted box-area severity (same formula, area = box area / image area).
  
  Compare per image (Health) and per IoU-matched instance: MAE, Pearson/Spearman, scatter plot, and action-category agreement (accuracy plus Cohen's κ). Also run an oracle **GT-box vs GT-mask** comparison, to separate the area-proxy error from detector error.
- **B2** Save 12 good and 6 failure predictions to `results/qualitative/`. Rank by per-image mask IoU against GT, and take failures from the lowest IoU. Add one line of likely cause per failure.
- **B3** Check that the dashboard runs with the real weights (4.6).
- **B4** Write `docs/REPORT_ASSETS.md`, `results/metrics/summary.json`, `docs/LIMITATIONS.md` and `docs/IMPROVEMENTS.md`, using the spec in the original project prompt. Every number must come from files under `results/`.

## 6. Design decisions already made (don't silently change them)
- **Health** = 100·Π(1 − Sᵢ), bounded to [0, 100]. It replaces 100 − ΣS, which can go negative.
- **Severity** S = clip(w_type · (1 − e^(−k·area_ratio)) · w_location · confidence, 0, 1), with k = 20 and w_type as specified. The critical-zone weight is w = 1 + 0.5 × the share of the defect inside the zone. All weights are assumptions.
- **Actions:** ≥ 80 Routine; 50–79 Schedule; 20–49 Urgent; < 20 Critical/Withdraw.
- **RUL "exponential"** means damage growth, H = 100 − D₀·e^(rt), not exponential decay of H.
- **True failure** is the first day the true Health (confidence 1) drops below 20. A simulated overhaul happens when an observed Health is below 20.
- **SARIMA** is ARMA(1,0,1) plus Fourier annual terms. 84 training weeks is less than 2 seasonal cycles, so a seasonal (P,D,Q,52) model is not identifiable.
- **Forecast protocol:** rolling-origin one-step-ahead forecasts over the last 20 weeks, so severity lags of 2–4 weeks are always known at forecast time. Prophet is refit every 4 weeks. Metrics are MAE, RMSE and MASE (no MAPE, because demand is intermittent).
- **Demand coupling is by construction.** A gain from severity features shows the method *can* use such a signal if it exists, not that real demand behaves this way.

## 7. Known risks (untested here)
- `src/vision/yolo_aug.py` patches `ultralytics.data.build.YOLODataset`. It relies on 8.3.x internals.
- `scripts/evaluate.py` uses `metrics.box/seg.map50|map|mp|mr|ap|ap50`, `ap_class_index`, `confusion_matrix` and `speed`. Check these against the installed ultralytics version.
- The converter's mask-PNG pairing is heuristic (folder/suffix tokens such as `mask`, `gt`, `label`, `_lab`). Check the `mask_without_image` and `img_without_mask` counters for each new source.
- Compressed COCO RLE needs pycocotools. Without it those instances are skipped and counted.

## 8. File map
`configs/` (datasets, train, severity, simulation, forecast) · `scripts/` (one per stage) · `src/common.py`, `src/plotting.py` · `src/vision/` (masks, data yaml helpers, augmentation, inference) · `src/severity/` (severity, rul) · `src/dashboard/` (app.py, views.py) · `.streamlit/config.toml` · `tests/` · `notebooks/train_colab.ipynb` · `results/` (generated outputs only) · `docs/` (this file, PROGRESS, ISSUES_LOG, DATASETS)
