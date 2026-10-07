"""Dashboard pages. Fleet, trend and alert views read the simulated fleet in results/; Inspect runs the model."""
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.common import ROOT
from src.plotting import INK, MUTED, SERIES, STATUS, plt
from src.severity import assess_masks, load_config
from src.severity.rul import fit_exponential, fit_linear, predict_health, predict_rul

CFG = load_config()
THR, ALERT = CFG["rul_failure_threshold"], CFG["alert_below"]
SIM, FC = ROOT / "results/simulation", ROOT / "results/forecasting"
BANDS = [(80, "Routine Monitoring"), (50, "Schedule Maintenance"), (20, "Urgent Intervention"), (0, "Critical/Withdraw")]


# ----------------------------------------------------------------------------- shared helpers
@st.cache_data
def load_csv(path):
    p = Path(path)
    return pd.read_csv(p) if p.is_file() else None


@st.cache_resource
def get_model(weights):
    from src.vision.infer import load_model
    return load_model(weights)


def meta(text):
    st.markdown(f"<div class='vg-meta'>{text}</div>", unsafe_allow_html=True)


def pill(action):
    return f"<span class='vg-pill' style='--c:{STATUS.get(action, MUTED)}'>{action}</span>"


def legend():
    labels = ["≥ 80 Routine Monitoring", "50–79 Schedule Maintenance", "20–49 Urgent Intervention", "< 20 Critical/Withdraw"]
    html = "".join(f"<span class='vg-pill' style='--c:{STATUS[a]}'>{t}</span>" for (_, a), t in zip(BANDS, labels))
    st.markdown(f"<div class='vg-legend'>{html}</div>", unsafe_allow_html=True)


def fleet_or_stop():
    fh = load_csv(SIM / "fleet_health.csv")
    if fh is None:
        st.info("No fleet data yet. Generate it with `python scripts/simulate_timelines.py`.")
        st.stop()
    return fh


def latest(fh):
    lt = fh.sort_values("day").groupby("equipment_id").tail(1).copy()
    lt["date"] = pd.to_datetime(lt["date"])
    return lt.sort_values("health_obs")


def current_cycle(fh, eq):
    g = fh[fh["equipment_id"] == eq]
    return g[g["cycle"] == g["cycle"].max()].sort_values("day")


def rul(g):
    t, h = g["day"].to_numpy(float), g["health_obs"].to_numpy(float)
    if len(g) < 3:
        return {"linear": np.nan, "exponential": np.nan}
    return {m: predict_rul(t, h, m, THR, 1825)[0] for m in ("linear", "exponential")}


def tint(v):
    return f"background-color: {STATUS[v]}33" if v in STATUS else ""


HEALTH_COL = st.column_config.ProgressColumn("Health", min_value=0, max_value=100, format="%.0f")
FLEET_COLS = {
    "equipment_id": st.column_config.TextColumn("Equipment"),
    "date": st.column_config.DateColumn("Last inspected", format="D MMM YYYY"),
    "cycle": st.column_config.NumberColumn("Life cycle", format="%d"),
    "health_obs": HEALTH_COL,
    "action": st.column_config.TextColumn("Action"),
    "n_defects": st.column_config.NumberColumn("Open defects", format="%d"),
    "max_severity": st.column_config.NumberColumn("Worst severity", format="%.2f"),
}


# ----------------------------------------------------------------------------- pages
def fleet():
    fh = fleet_or_stop()
    lt = latest(fh)
    st.title("Fleet status")
    meta(f"{len(lt)} assets · last inspection {lt['date'].max():%d %b %Y} · simulated fleet")

    c = st.columns(4)
    c[0].metric("Mean health", f"{lt['health_obs'].mean():.1f}", border=True)
    c[1].metric(f"Below {ALERT}", f"{(lt['health_obs'] < ALERT).sum()} of {len(lt)}", border=True)
    c[2].metric("Critical / withdraw", int((lt["health_obs"] < 20).sum()), border=True)
    c[3].metric("Overhauls to date", int(fh["overhaul_after"].sum()), border=True)

    show = st.segmented_control("Show", ["All", "Below 80", f"Below {ALERT}", "Below 20"], default="All")
    cut = {"Below 80": 80, f"Below {ALERT}": ALERT, "Below 20": 20}.get(show or "All")
    view = lt if cut is None else lt[lt["health_obs"] < cut]
    cols = list(FLEET_COLS)
    st.dataframe(view[cols].style.map(tint, subset=["action"]), column_config=FLEET_COLS,
                 hide_index=True, width="stretch")

    st.subheader("Health by inspection")
    order = lt["equipment_id"].tolist()
    mat = fh.pivot_table(index="equipment_id", columns="inspection_idx", values="health_obs").reindex(order)
    color = np.full(mat.shape + (4,), 0.0)
    from matplotlib.colors import to_rgba
    for (i, j), v in np.ndenumerate(mat.to_numpy()):
        if not np.isnan(v):
            color[i, j] = to_rgba(next(STATUS[a] for lo, a in BANDS if v >= lo))
    fig, ax = plt.subplots(figsize=(10, 0.26 * len(mat) + 0.8))
    for (i, j), v in np.ndenumerate(mat.to_numpy()):
        if not np.isnan(v):
            ax.add_patch(plt.Rectangle((j + 0.04, i + 0.06), 0.92, 0.88, color=color[i, j], lw=0))
            ax.text(j + 0.5, i + 0.52, f"{v:.0f}", ha="center", va="center", fontsize=6.5, color=INK)
    ax.set_xlim(0, mat.shape[1])
    ax.set_ylim(len(mat), 0)
    ax.set_yticks(np.arange(len(mat)) + 0.5, mat.index, fontsize=7)
    ax.set_xticks(np.arange(mat.shape[1]) + 0.5, mat.columns, fontsize=7)
    ax.set_xlabel("inspection #")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    st.pyplot(fig)
    legend()
    st.caption("Sorted by latest health, worst first. Blank cells: no inspection yet.")


def inspect():
    st.title("Inspect image")
    meta("Segment defects in one image and score the asset")
    left, right = st.columns([1, 2.2], gap="large")
    with left:
        up = st.file_uploader("Image", type=["jpg", "jpeg", "png", "bmp"])
        zone_up = st.file_uploader("Critical-zone mask (optional)", type=["png", "jpg"],
                                   help="White pixels mark load-bearing or safety-critical areas.")
        conf = st.slider("Minimum confidence", 0.05, 0.95, float(CFG["inference"]["conf"]), 0.05)
        with st.expander("Model"):
            weights = st.text_input("Weights", CFG["inference"]["weights"])
    model, reason = get_model(weights)
    with right:
        if model is None:
            st.info(f"No model loaded: {reason}.\n\nTrain one with `python scripts/train.py --model yolo11s-seg`, "
                    "or point *Model → Weights* to an existing `best.pt`.")
            return
        if up is None:
            st.caption("Upload an image to start.")
            return
        from src.vision.infer import predict
        img = np.asarray(Image.open(up).convert("RGB"))
        out = predict(model, img, conf=conf, imgsz=CFG["inference"]["imgsz"])
        zone = None
        if zone_up is not None:
            zone = np.asarray(Image.open(zone_up).convert("L").resize(img.shape[1::-1], Image.NEAREST)) > 127
        table, health, action = assess_masks(out["masks"], out["classes"], out["confs"], zone, CFG)
        a, b = st.columns(2)
        a.image(img, caption="Input", width="stretch")
        b.image(out["plot"], caption=f"Detections ({len(table)})", width="stretch")
        st.markdown(f"**Health {health:.1f}** &nbsp; {pill(action)}", unsafe_allow_html=True)
        if len(table):
            t = table.assign(area_pct=100 * table["area_ratio"]).sort_values("severity", ascending=False)
            st.dataframe(t[["class", "area_pct", "confidence", "w_location", "severity"]], hide_index=True,
                         width="stretch", column_config={
                             "class": "Defect", "area_pct": st.column_config.NumberColumn("Area, % of image", format="%.2f"),
                             "confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
                             "w_location": st.column_config.NumberColumn("Location weight", format="%.2f"),
                             "severity": st.column_config.ProgressColumn("Severity", min_value=0, max_value=1, format="%.2f")})
        else:
            st.caption("No defects above the confidence threshold.")
        with st.expander("How the score is computed"):
            st.markdown(f"Severity per defect: S = clip(w_type · (1 − e^(−{CFG['k']:g}·area)) · w_location · confidence, 0, 1).  \n"
                        "Health = 100 · Π(1 − S). Type weights and thresholds: `configs/severity.yaml`.")


def trends():
    st.title("Degradation & parts")
    tab_rul, tab_parts = st.tabs(["Remaining useful life", "Spare-parts demand"])
    fh = fleet_or_stop()
    with tab_rul:
        ids = latest(fh)["equipment_id"].tolist()  # worst first
        eq = st.selectbox("Equipment", ids)
        g = current_cycle(fh, eq)
        r = rul(g)
        c = st.columns(4)
        c[0].metric("Health", f"{g['health_obs'].iloc[-1]:.1f}", border=True)
        c[1].metric("RUL, exponential", "–" if np.isnan(r["exponential"]) else f"{r['exponential']:.0f} d", border=True)
        c[2].metric("RUL, linear", "–" if np.isnan(r["linear"]) else f"{r['linear']:.0f} d", border=True)
        c[3].metric("Inspections this cycle", len(g), border=True)
        t, h = g["day"].to_numpy(float), g["health_obs"].to_numpy(float)
        fig, ax = plt.subplots(figsize=(9, 3.3))
        ax.plot(t, g["health_true"], "o", color=MUTED, ms=4, label="true (simulated)")
        ax.plot(t, h, "o-", color=SERIES[0], label="observed")
        if len(g) >= 3:
            horizon = np.nanmax([r["exponential"], r["linear"], 60.0])
            tt = np.linspace(t.min(), t.max() + min(horizon, 720) + 30, 200)
            for m, col, p in [("linear", SERIES[1], fit_linear(t, h)), ("exponential", SERIES[2], fit_exponential(t, h))]:
                ax.plot(tt, np.clip(predict_health(p, tt, m), 0, 100), "--", color=col, lw=1.5, label=f"{m} fit")
        ax.axhline(THR, color=STATUS["Critical/Withdraw"], ls=":", lw=1)
        ax.text(ax.get_xlim()[0], THR + 2, f" failure threshold {THR}", fontsize=7, color=INK)
        ax.set_ylim(0, 105)
        ax.set_xlabel("day")
        ax.set_ylabel("health")
        ax.legend(fontsize=7, loc="lower left")
        st.pyplot(fig)
        st.caption(f"Life cycle {int(g['cycle'].max())}; earlier cycles ended in an overhaul.")

    with tab_parts:
        fc, dem = load_csv(FC / "forecasts.csv"), load_csv(FC / "demand_synthetic.csv")
        if fc is None or dem is None:
            st.info("No forecasts yet. Run `python scripts/run_forecasting.py`.")
            return
        c = st.columns([1, 1, 2])
        level = c[0].selectbox("Severity coupling", list(dict.fromkeys(fc["level"])), index=0)
        sku = c[1].selectbox("SKU", sorted(fc["sku"].unique()))
        all_m = list(dict.fromkeys(fc["model"]))
        models = c[2].multiselect("Models", all_m, default=[m for m in ("sba", "lgbm", "lgbm_sev") if m in all_m] or all_m[:2])
        d = dem[(dem["level"] == level) & (dem["rep"] == 0) & (dem["sku"] == sku)]
        f = fc[(fc["level"] == level) & (fc["rep"] == 0) & (fc["sku"] == sku)]
        fig, ax = plt.subplots(figsize=(9, 3.1))
        ax.axvspan(f["week"].min() - 0.5, f["week"].max() + 0.5, color="#f0efec", lw=0)
        ax.step(d["week"], d["demand"], where="mid", color=INK, lw=1.2, label="actual")
        for i, m in enumerate(models):
            fm = f[f["model"] == m]
            ax.plot(fm["week"], fm["pred"], "--" if m.endswith("_sev") else "-", color=SERIES[i % len(SERIES)], label=m)
        ax.set_xlabel("week (shaded: test period)")
        ax.set_ylabel("units")
        ax.legend(fontsize=7, ncol=4, loc="upper left")
        st.pyplot(fig)
        met = load_csv(ROOT / "results/tables/forecast_metrics.csv")
        if met is not None:
            mt = met[(met["level"] == level) & (met["sku"] == sku)].groupby("model")[["MAE", "RMSE", "MASE"]].mean()
            st.dataframe(mt.sort_values("MASE").reset_index(), hide_index=True, column_config={
                "model": "Model", **{k: st.column_config.NumberColumn(k, format="%.3f") for k in ("MAE", "RMSE", "MASE")}})
            st.caption("Errors averaged over replicates (test weeks only). Demand is synthetic, and its link to "
                       "severity is set in the simulation, so this tests the method, not real demand.")


def alerts():
    fh = fleet_or_stop()
    lt = latest(fh)
    al = lt[lt["health_obs"] < ALERT].copy()
    st.title("Alerts")
    meta(f"{len(al)} assets below health {ALERT} · sorted worst first")
    if al.empty:
        st.caption(f"Nothing below {ALERT}.")
        return
    r = {e: rul(current_cycle(fh, e)) for e in al["equipment_id"]}
    al["rul_exp"] = [r[e]["exponential"] for e in al["equipment_id"]]
    al["rul_lin"] = [r[e]["linear"] for e in al["equipment_id"]]
    cols = ["equipment_id", "date", "health_obs", "action", "n_defects", "max_severity", "rul_exp", "rul_lin"]
    st.dataframe(al[cols].style.map(tint, subset=["action"]), hide_index=True, width="stretch", column_config={
        **FLEET_COLS, "rul_exp": st.column_config.NumberColumn("RUL exp., days", format="%.0f"),
        "rul_lin": st.column_config.NumberColumn("RUL lin., days", format="%.0f")})
    st.download_button("Export CSV", al[cols].to_csv(index=False).encode(), "visionguard_alerts.csv", "text/csv")
