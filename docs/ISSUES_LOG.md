# Issues log (one line each: problem -> fix)

- 2026-10-07 Dev laptop has no GPU, no `data/raw/`, and no ultralytics/statsmodels/prophet/lightgbm/opencv; downloads not allowed -> GPU/heavy steps handed over (docs/HANDOVER.md); GPU scripts are compile-checked only.
- 2026-10-07 Converter fixture test: YOLO-seg not detected when label files mix polygon and box-only rows -> detect on any polygon row.
- 2026-10-07 Converter fixture test: 4-connected labelling split diagonal 1-px cracks into specks (all dropped) -> 8-connected labelling.
- 2026-10-07 Converter fixture test: 1-2 px wide masks trace to ~zero-area polygons -> dilate component by 1 px before contour tracing.
- 2026-10-07 Kaggle `sample_submission.csv` also has an EncodedPixels column -> excluded by name; `train.csv` preferred.
- 2026-10-07 Ultralytics resolves a relative/missing `path` in data.yaml against its global DATASETS_DIR -> train/eval write `data_resolved.yaml` with an absolute path per machine.
- 2026-10-07 Ultralytics has no contrast augmentation -> `BrightnessContrast` inserted via a YOLODataset subclass (src/vision/yolo_aug.py); falls back to hsv_v with a warning if the patch fails.
- 2026-10-07 Simulation v1 (emergence 0.006/day, drifts 0.002-0.006): median true failure day 935 > 728-day window, 0 overhauls, 73% of linear RUL predictions hit the cap (MAE linear 984 d, exp 493 d) -> raised emergence to 0.008/day and drifts by ~1.45x so failures fall inside the window (v2: median 574 d, 11 overhauls). v2 is the reported run.
- 2026-10-07 Forecasting on dev laptop: statsmodels/prophet/lightgbm missing -> only naive/Croston/SBA ran; with-vs-without-severity sensitivity table not yet produced.
- 2026-10-07 Streamlit 1.51 deprecates `use_container_width` -> `width="stretch"`.
- 2026-10-07 Streamlit AppTest cannot `switch_page` to function-based `st.Page`s -> views live in `src/dashboard/views.py` and are tested by calling each view directly.
