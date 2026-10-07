"""A5: defect-driven spare-parts demand forecasting on SYNTHETIC demand (coupling is by construction).

Demand for 6 SKUs (one per defect class + one general SKU) is generated from a baseline, annual seasonality,
intermittency and negative-binomial noise, plus an uplift driven by A4 fleet severity lagged 2-4 weeks.
Models (rolling-origin one-step-ahead over the last 20 weeks): naive, Croston, SBA, SARIMA (ARMA + Fourier),
SARIMAX + severity, Prophet, LightGBM, LightGBM + severity. Metrics: MAE, RMSE, MASE (in-sample naive scale).
Sensitivity: coupling gamma in {zero, low, medium, high}; at zero the severity features should not help.
Models whose library is missing are skipped with a warning and listed in results/metrics/forecast_summary.json.

Usage: python scripts/run_forecasting.py   (run scripts/simulate_timelines.py first)
"""
import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import df_to_md, ensure_dir, load_yaml, rpath, save_json, save_table  # noqa: E402
from src.plotting import INK, MUTED, SERIES, plt, savefig  # noqa: E402

MODEL_COLORS = {"naive": MUTED, "croston": SERIES[6], "sba": SERIES[3], "sarima": SERIES[1], "sarimax_sev": SERIES[1],
                "prophet": SERIES[4], "lgbm": SERIES[0], "lgbm_sev": SERIES[0]}
PAIRS = [("sarima", "sarimax_sev"), ("lgbm", "lgbm_sev")]


# ----------------------------------------------------------------------------- data
def lagged(x, lags):
    out = np.zeros_like(x, dtype=float)
    for lag, w in lags.items():
        out[int(lag):] += w * x[: len(x) - int(lag)]
    return out


def norm_severity(fsev, cls):
    s = fsev[cls].to_numpy(float)
    return s / s.mean() if s.mean() > 0 else np.zeros_like(s)


def make_demand(fsev, cfg, gamma, rep):
    n, t = cfg["n_weeks"], np.arange(cfg["n_weeks"])
    rows = []
    for i, s in enumerate(cfg["skus"]):
        rng = np.random.default_rng([cfg["seed"], rep, i])  # identical draws at every coupling level
        occ_u, gam_u, poi_u = rng.random(n), rng.random(n), rng.random(n)
        z = norm_severity(fsev, s["defect_class"])[:n]
        uplift = gamma * s["base"] * lagged(z, cfg["coupling_lags"])
        mu = s["base"] * (1 + s["season_amp"] * np.sin(2 * np.pi * (t + s["season_phase"]) / 52)) + uplift
        r, p = s["nb_dispersion"], s["p_occurrence"]
        lam = (mu / p) * stats.gamma.ppf(np.clip(gam_u, 1e-9, 1 - 1e-9), a=r, scale=1 / r)  # NB = Poisson-Gamma
        y = (occ_u < p) * stats.poisson.ppf(np.clip(poi_u, 1e-12, 1 - 1e-12), np.maximum(lam, 1e-9))
        rows.append(pd.DataFrame({"sku": s["sku"], "week": t, "week_start": fsev["week_start"].to_numpy()[:n],
                                  "demand": y.astype(int), "mu": mu, "uplift": uplift}))
    return pd.concat(rows, ignore_index=True)


# ----------------------------------------------------------------------------- models
def f_naive(y, n_tr, **_):
    return y[n_tr - 1:-1].astype(float)


def f_croston(y, n_tr, alpha=0.1, sba=False, **_):
    z = p = None
    q, preds = 1, np.zeros(len(y))
    for t in range(len(y)):
        if z is not None:
            preds[t] = z / p * ((1 - alpha / 2) if sba else 1.0)
        if y[t] > 0:
            if z is None:
                z, p = float(y[t]), float(q)
            else:
                z, p = z + alpha * (y[t] - z), p + alpha * (q - p)
            q = 1
        else:
            q += 1
    return preds[n_tr:]


def fourier(n, period, k):
    t = np.arange(n)
    return np.column_stack([f(2 * np.pi * j * t / period) for j in range(1, k + 1) for f in (np.sin, np.cos)])


def f_sarima(y, n_tr, exog, order, **_):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(y[:n_tr].astype(float), exog=exog[:n_tr], order=tuple(order), trend="c").fit(disp=False, maxiter=200)
        full = res.apply(y.astype(float), exog=exog)  # same parameters, filtered on all data -> one-step-ahead
        return np.asarray(full.get_prediction(start=n_tr, end=len(y) - 1).predicted_mean)


def f_prophet(y, n_tr, dates, refit_every, yearly_seasonality, **_):
    from prophet import Prophet
    preds = []
    for o in range(n_tr, len(y), refit_every):
        m = Prophet(yearly_seasonality=yearly_seasonality, weekly_seasonality=False, daily_seasonality=False,
                    uncertainty_samples=0)
        m.fit(pd.DataFrame({"ds": dates[:o], "y": y[:o].astype(float)}))
        preds.extend(m.predict(pd.DataFrame({"ds": dates[o:min(o + refit_every, len(y))]}))["yhat"])
    return np.asarray(preds)


def lgbm_frame(dem, fsev, cfg):
    f, out = cfg["features"], []
    for s in cfg["skus"]:
        d = dem[dem["sku"] == s["sku"]].sort_values("week").copy()
        for lag in f["demand_lags"]:
            d[f"lag_{lag}"] = d["demand"].shift(lag)
        for w in f["rolling_means"]:
            d[f"roll_{w}"] = d["demand"].shift(1).rolling(w, min_periods=1).mean()
        d["sin52"], d["cos52"] = np.sin(2 * np.pi * d["week"] / 52), np.cos(2 * np.pi * d["week"] / 52)
        z = norm_severity(fsev, s["defect_class"])[: len(d)]
        for lag in f["severity_lags"]:
            d[f"sev_lag_{lag}"] = pd.Series(z).shift(lag).fillna(0.0).to_numpy()
        out.append(d)
    df = pd.concat(out, ignore_index=True)
    df["sku_cat"] = df["sku"].astype("category")
    return df


def f_lgbm(frame, n_tr, cfg, with_sev):
    import lightgbm as lgb
    f = cfg["features"]
    cols = ([f"lag_{l}" for l in f["demand_lags"]] + [f"roll_{w}" for w in f["rolling_means"]]
            + ["sin52", "cos52", "sku_cat"] + ([f"sev_lag_{l}" for l in f["severity_lags"]] if with_sev else []))
    tr = frame[(frame["week"] < n_tr) & (frame["week"] >= 1)]
    te = frame[frame["week"] >= n_tr]
    m = lgb.LGBMRegressor(**cfg["models"]["lightgbm"], random_state=cfg["seed"], n_jobs=1, deterministic=True,
                          force_row_wise=True, verbose=-1)
    m.fit(tr[cols], tr["demand"])
    return dict(zip(zip(te["sku"], te["week"]), m.predict(te[cols])))


def available(enabled):
    libs = {"sarima": "statsmodels", "sarimax_sev": "statsmodels", "prophet": "prophet",
            "lgbm": "lightgbm", "lgbm_sev": "lightgbm"}
    ok, skipped = [], {}
    for m in enabled:
        lib = libs.get(m)
        if lib:
            try:
                __import__(lib)
            except ImportError:
                skipped[m] = f"{lib} not installed"
                continue
        ok.append(m)
    return ok, skipped


def scores(y, yhat, y_train):
    e = y - yhat
    scale = np.mean(np.abs(np.diff(y_train)))
    return {"MAE": np.mean(np.abs(e)), "RMSE": np.sqrt(np.mean(e ** 2)),
            "MASE": np.mean(np.abs(e)) / scale if scale > 0 else np.nan}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/forecast.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)
    fs_path = rpath(cfg["fleet_severity_csv"])
    if not fs_path.is_file():
        sys.exit(f"{fs_path} missing - run scripts/simulate_timelines.py first")
    fsev = pd.read_csv(fs_path)
    if len(fsev) < cfg["n_weeks"]:
        sys.exit(f"fleet severity has {len(fsev)} weeks < n_weeks={cfg['n_weeks']}")
    models, skipped = available(cfg["models"]["enabled"])
    for m, why in skipped.items():
        print(f"[WARN] model {m} skipped: {why}")
    n, n_tr, mc = cfg["n_weeks"], cfg["n_weeks"] - cfg["test_weeks"], cfg["models"]
    four = fourier(n, mc["sarima"]["period"], mc["sarima"]["fourier_k"])

    demand_all, fc_rows, met_rows = [], [], []
    for level, gamma in cfg["coupling_levels"].items():
        for rep in range(cfg["n_replicates"]):
            dem = make_demand(fsev, cfg, gamma, rep)
            demand_all.append(dem.assign(level=level, gamma=gamma, rep=rep))
            frame = lgbm_frame(dem, fsev, cfg) if {"lgbm", "lgbm_sev"} & set(models) else None
            lg = {m: f_lgbm(frame, n_tr, cfg, m == "lgbm_sev") for m in ("lgbm", "lgbm_sev") if m in models}
            for s in cfg["skus"]:
                d = dem[dem["sku"] == s["sku"]].sort_values("week")
                y, dates = d["demand"].to_numpy(), pd.to_datetime(d["week_start"]).to_numpy()
                sev = np.column_stack([pd.Series(norm_severity(fsev, s["defect_class"])[:n]).shift(l).fillna(0.0)
                                       for l in cfg["features"]["severity_lags"]])
                for m in models:
                    if m == "naive":
                        p = f_naive(y, n_tr)
                    elif m in ("croston", "sba"):
                        p = f_croston(y, n_tr, mc["croston_alpha"], sba=m == "sba")
                    elif m == "sarima":
                        p = f_sarima(y, n_tr, four, mc["sarima"]["order"])
                    elif m == "sarimax_sev":
                        p = f_sarima(y, n_tr, np.column_stack([four, sev]), mc["sarima"]["order"])
                    elif m == "prophet":
                        p = f_prophet(y, n_tr, dates, **mc["prophet"])
                    else:
                        p = np.array([lg[m][(s["sku"], w)] for w in range(n_tr, n)])
                    p = np.clip(np.asarray(p, float), 0, None)
                    for w, a, ph in zip(range(n_tr, n), y[n_tr:], p):
                        fc_rows.append((level, rep, s["sku"], w, a, m, ph))
                    met_rows.append({"level": level, "gamma": gamma, "rep": rep, "sku": s["sku"], "model": m,
                                     **scores(y[n_tr:], p, y[:n_tr])})
            print(f"[INFO] level={level} rep={rep} done")

    out = ensure_dir("results/forecasting")
    pd.concat(demand_all).to_csv(out / "demand_synthetic.csv", index=False)
    fc = pd.DataFrame(fc_rows, columns=["level", "rep", "sku", "week", "actual", "model", "pred"])
    fc.to_csv(out / "forecasts.csv", index=False)
    met = pd.DataFrame(met_rows)
    save_table(met, "forecast_metrics")
    order = list(cfg["coupling_levels"])
    summ = met.groupby(["level", "model"], sort=False)[["MAE", "RMSE", "MASE"]].mean().reset_index()
    summ["level"] = pd.Categorical(summ["level"], order, ordered=True)
    summ = summ.sort_values(["level", "MASE"])
    save_table(summ, "forecast_summary")

    sens = []
    for base, ext in PAIRS:
        if base in models and ext in models:
            for level in order:
                a = met[(met["level"] == level) & (met["model"] == base)].set_index(["rep", "sku"])["MASE"]
                b = met[(met["level"] == level) & (met["model"] == ext)].set_index(["rep", "sku"])["MASE"]
                delta = (b - a).dropna()
                per_rep = delta.groupby(level="rep").mean()
                sens.append({"pair": f"{ext} vs {base}", "level": level, "gamma": cfg["coupling_levels"][level],
                             "MASE_without": a.mean(), "MASE_with": b.mean(), "delta_MASE": delta.mean(),
                             "delta_MASE_sd_over_reps": per_rep.std(ddof=1) if len(per_rep) > 1 else np.nan,
                             "share_series_improved": float((delta < 0).mean()), "n_series": len(delta)})
    sens_df = pd.DataFrame(sens)
    if len(sens_df):
        save_table(sens_df, "forecast_sensitivity")

    # plots: forecast vs actual per SKU (replicate 0), one figure per coupling level
    plot_models = [m for m in cfg["plot_models"] if m in models]
    for level in order:
        dl = pd.concat(demand_all)
        dl = dl[(dl["level"] == level) & (dl["rep"] == 0)]
        fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
        for ax, s in zip(axes.ravel(), cfg["skus"]):
            d = dl[(dl["sku"] == s["sku"]) & (dl["week"] >= n_tr - 30)]
            ax.axvspan(n_tr - 0.5, n - 0.5, color="#f0efec", lw=0)
            ax.step(d["week"], d["demand"], where="mid", color=INK, lw=1.2, label="actual")
            for m in plot_models:
                f_ = fc[(fc["level"] == level) & (fc["rep"] == 0) & (fc["sku"] == s["sku"]) & (fc["model"] == m)]
                ax.plot(f_["week"], f_["pred"], "--" if m.endswith("_sev") else "-", color=MODEL_COLORS[m], lw=1.5, label=m)
            ax.set_title(f"{s['sku']} ({s['part']} <- {s['defect_class']})", loc="left", fontsize=8)
        axes[0, 0].legend(fontsize=7, ncol=2)
        for ax in axes[1]:
            ax.set_xlabel("week (shaded = test)")
        fig.suptitle(f"Forecast vs actual, coupling = {level} (gamma {cfg['coupling_levels'][level]}), replicate 0",
                     x=0.01, ha="left", fontsize=9)
        savefig(fig, f"results/figures/forecast_vs_actual_{level}.png")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m in models:
        y_ = [summ[(summ["level"] == lv) & (summ["model"] == m)]["MASE"].mean() for lv in order]
        ax.plot(order, y_, ("--" if m.endswith("_sev") else "-"), marker="o", color=MODEL_COLORS[m], label=m)
    ax.set_ylabel("mean MASE over SKUs and replicates (lower = better)")
    ax.set_xlabel("severity -> demand coupling strength")
    ax.set_title("Sensitivity to coupling (dashed = with severity features)", loc="left")
    ax.legend(fontsize=7, ncol=2)
    savefig(fig, "results/figures/forecast_sensitivity.png")

    save_json({"synthetic": True, "coupling_by_construction": True, "models_run": models, "models_skipped": skipped,
               "n_replicates": cfg["n_replicates"], "test_weeks": cfg["test_weeks"],
               "summary": summ.astype({"level": str}).to_dict(orient="records"),
               "sensitivity": sens_df.to_dict(orient="records")}, "results/metrics/forecast_summary.json")
    print(df_to_md(summ.astype({"level": str})))
    if len(sens_df):
        print(df_to_md(sens_df))
    else:
        print("[WARN] no with/without-severity model pair available -> sensitivity table not produced")


if __name__ == "__main__":
    main()
