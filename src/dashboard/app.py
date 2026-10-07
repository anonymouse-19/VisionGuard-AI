"""VisionGuard dashboard.  Run from the repo root: streamlit run src/dashboard/app.py"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.dashboard import views  # noqa: E402

st.set_page_config(page_title="VisionGuard", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 2.2rem; max-width: 1280px;}
[data-testid="stMetricValue"] {font-size: 1.55rem; font-variant-numeric: tabular-nums;}
[data-testid="stMetricLabel"] p {color: #52514e;}
.vg-meta {color: #6b6a65; font-size: .86rem; margin: -.55rem 0 1.1rem;}
.vg-pill {display: inline-flex; align-items: center; gap: .4rem; padding: .08rem .55rem; font-size: .8rem;
          border: 1px solid #e1e0d9; border-radius: 999px; background: #fff; color: #1f1f1d; white-space: nowrap;}
.vg-pill::before {content: ""; width: .55rem; height: .55rem; border-radius: 50%; background: var(--c);}
.vg-legend {display: flex; flex-wrap: wrap; gap: .45rem; margin: .1rem 0 .6rem;}
</style>""", unsafe_allow_html=True)

nav = st.navigation({"VisionGuard": [
    st.Page(views.fleet, title="Fleet status", url_path="fleet", default=True),
    st.Page(views.alerts, title="Alerts", url_path="alerts"),
    st.Page(views.trends, title="Degradation & parts", url_path="trends"),
    st.Page(views.inspect, title="Inspect image", url_path="inspect"),
]})
st.sidebar.caption("Fleet data is simulated (results/simulation). The model is trained on public proxy datasets.")
nav.run()
