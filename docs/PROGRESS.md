# Progress (resume point: see docs/HANDOVER.md)

## Done
- [x] A1 converter `scripts/convert_to_yolo_seg.py` + `scripts/dataset_stats.py`: tested on a synthetic fixture covering all 4 formats (no real data yet)
- [x] A1 `docs/DATASETS.md` (mappings; licences = check manually)
- [x] A2 `scripts/train.py`, `scripts/evaluate.py`, `scripts/compare_models.py`, `notebooks/train_colab.ipynb`: written and compile-checked only (no GPU/ultralytics here)
- [x] A3 `src/severity/` + `configs/severity.yaml` + `tests/test_severity.py`: 15/15 tests pass
- [x] A4 `scripts/simulate_timelines.py` synthetic mode: run; results in results/simulation, results/tables/rul_*, results/figures/rul_*
- [x] A5 `scripts/run_forecasting.py`: run with naive/Croston/SBA only (other libs missing); partial results
- [x] A6 dashboard (`src/dashboard/app.py` + `views.py`, theme in `.streamlit/config.toml`): native navigation, no emoji, status pills, filters, CSV export; all 4 views render in a headless Streamlit test

## Next (needs a GPU machine / full requirements)
- [ ] `pip install -r requirements.txt` (verify the unverified pins)
- [ ] Download datasets into data/raw/<source>/ -> convert -> stats -> update licences in DATASETS.md
- [ ] Train smoke, then yolo11n-seg and yolo11s-seg -> evaluate -> compare
- [ ] Re-run `scripts/run_forecasting.py` with statsmodels/prophet/lightgbm (sensitivity table)
- [ ] Stage B: B1 severity experiment, B2 qualitative, B3 dashboard with real weights, B4 report docs
