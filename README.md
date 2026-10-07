# VisionGuard AI

Equipment image → YOLO11-seg defect segmentation → severity → Health Score → RUL → maintenance action → defect-driven spare-parts forecast → Streamlit dashboard.

Public proxy datasets and simulated fleet timelines are used because real defence data is unavailable.

**Start here: [docs/HANDOVER.md](docs/HANDOVER.md)** (status, remaining work, exact commands).

```bash
pip install -r requirements.txt
python -m pytest -q tests/
python scripts/simulate_timelines.py && python scripts/run_forecasting.py   # CPU, synthetic
streamlit run src/dashboard/app.py
```
