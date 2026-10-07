"""A4: synthetic inspection timelines -> Health trajectories -> RUL (linear vs exponential) vs ground truth.

SYNTHETIC DATA: real defence-equipment inspection histories are unavailable, so defects emerge (Poisson) and
grow (geometric Brownian motion on mask-area ratio) with parameters in configs/simulation.yaml. Inspections
observe the latent state with detection limits, area noise and detector confidence; severity/health use the
same rules as the real pipeline (src/severity). Ground-truth failure = first day TRUE health < threshold.

Usage:
  python scripts/simulate_timelines.py                      # synthetic mode (no weights needed)
  python scripts/simulate_timelines.py --from-images inspections.csv --weights runs/segment/yolo11s-seg/weights/best.pt
      (CSV columns: equipment_id, timestamp, image_path; no ground truth, RUL predictions only)
Outputs: results/simulation/{fleet_health.csv, inspection_defects.csv, fleet_severity_weekly.csv},
         results/tables/rul_errors.csv, rul_mae_by_k.csv, rul_summary.csv|md, results/metrics/rul_summary.json,
         results/figures/rul_sample_curves.png, rul_mae_by_k.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import CLASSES, ensure_dir, load_yaml, rpath, save_json, save_table  # noqa: E402
from src.plotting import MUTED, SERIES, plt, savefig  # noqa: E402
from src.severity import health_score, load_config, maintenance_action, severity  # noqa: E402
from src.severity.rul import fit_exponential, fit_linear, predict_health, predict_rul  # noqa: E402

METHODS = ["linear", "exponential"]


def simulate_cycle(rng, n_days, frailty, p, scfg):
    """Latent defects for one life cycle: list of (class, area[n_days]) and true daily health."""
    probs = np.array([p["class_probs"][c] for c in CLASSES])
    n_new = rng.poisson(p["emergence_rate_per_day"] * frailty * n_days)
    defects = []
    for t0, c in zip(np.sort(rng.integers(0, n_days, n_new)), rng.choice(CLASSES, n_new, p=probs / probs.sum())):
        a0 = p["initial_area"]["median"] * np.exp(p["initial_area"]["log_sigma"] * rng.standard_normal())
        steps = p["growth_drift_per_day"][c] * frailty + p["growth_volatility"] * rng.standard_normal(n_days - t0 - 1)
        area = np.zeros(n_days)
        area[t0:] = np.minimum(a0 * np.exp(np.r_[0.0, np.cumsum(steps)]), p["max_area_ratio"])
        defects.append((c, area))
    if defects:
        s = np.stack([severity([c] * n_days, a, 1.0, 1.0, scfg) for c, a in defects])
        h_true = 100.0 * np.prod(1.0 - s, axis=0)
    else:
        h_true = np.full(n_days, 100.0)
    return defects, h_true


def simulate(cfg, scfg):
    rng = np.random.default_rng(cfg["seed"])
    p, ob, thr = cfg["defect_process"], cfg["observation"], cfg["rul"]["failure_threshold"]
    total = cfg["horizon_days"] + cfg["gt_extra_days"]
    insp_rows, defect_rows, gt = [], [], {}
    for e in range(cfg["n_equipment"]):
        eq = f"EQ-{e + 1:02d}"
        frailty = float(np.exp(p["frailty_log_sigma"] * rng.standard_normal()))
        n_insp = int(rng.integers(cfg["n_inspections"][0], cfg["n_inspections"][1] + 1))
        iv = cfg["inspection_interval_days"]
        days = int(rng.integers(*cfg["first_inspection_day"])) + np.r_[0, np.cumsum(
            rng.integers(iv["mean"] - iv["jitter"], iv["mean"] + iv["jitter"] + 1, n_insp - 1))]
        days = days[days < cfg["horizon_days"]]
        defects, h_true = simulate_cycle(rng, total, frailty, p, scfg)
        fail = np.flatnonzero(h_true < thr)
        gt[eq] = {"t_fail": int(fail[0]) if fail.size else None, "h_true_cycle1": h_true, "frailty": frailty}
        cycle, start = 1, 0
        for k, d in enumerate(days, start=1):
            obs_s = []
            for j, (c, area) in enumerate(defects):
                a = area[d - start]
                if a < ob["detection_min_area"]:
                    continue
                a_obs = min(1.0, a * np.exp(ob["area_noise_log_sigma"] * rng.standard_normal()))
                conf = float(rng.beta(*ob["confidence_beta"]))
                s = severity(c, a_obs, conf, 1.0, scfg)
                obs_s.append(s)
                defect_rows.append(dict(equipment_id=eq, inspection_idx=k, day=int(d), cycle=cycle,
                                        defect_id=f"{eq}-c{cycle}-d{j}", cls=c, area_ratio_true=a,
                                        area_ratio_obs=a_obs, confidence=conf, severity=s))
            h_obs = health_score(obs_s)
            overhaul = cfg["overhaul"]["enabled"] and h_obs < cfg["overhaul"]["observed_health_below"]
            insp_rows.append(dict(equipment_id=eq, inspection_idx=k, day=int(d), cycle=cycle,
                                  n_defects=len(obs_s), max_severity=max(obs_s, default=0.0),
                                  health_obs=h_obs, health_true=float(h_true[d - start]),
                                  action=maintenance_action(h_obs, scfg), overhaul_after=bool(overhaul)))
            if overhaul:  # new life cycle starting at this inspection
                cycle, start = cycle + 1, int(d)
                defects, h_true = simulate_cycle(rng, total - start, frailty, p, scfg)
    fh, dr = pd.DataFrame(insp_rows), pd.DataFrame(defect_rows)
    t0 = pd.Timestamp(cfg["start_date"])
    for df in (fh, dr):
        df.insert(3, "date", (t0 + pd.to_timedelta(df["day"], unit="D")).dt.date)
    return fh, dr, gt


def weekly_fleet_severity(fh, dr, cfg):
    """Fleet severity load per class per week: sum over equipment of the latest inspection's severities."""
    per_insp = dr.groupby(["equipment_id", "inspection_idx", "cls"])["severity"].sum().unstack(fill_value=0.0)
    per_insp = per_insp.reindex(columns=CLASSES, fill_value=0.0)
    base = fh.set_index(["equipment_id", "inspection_idx"])[["day"]].join(per_insp).fillna(0.0).reset_index()
    weeks = np.arange(cfg["horizon_days"] // 7)
    rows = []
    for w in weeks:
        latest = base[base["day"] <= 7 * w + 6].sort_values("day").groupby("equipment_id").tail(1)
        rows.append({"week": int(w), **{c: float(latest[c].sum()) for c in CLASSES}})
    out = pd.DataFrame(rows)
    out["all"] = out[CLASSES].sum(axis=1)
    out.insert(1, "week_start", (pd.Timestamp(cfg["start_date"]) + pd.to_timedelta(7 * out["week"], unit="D")).dt.date)
    return out


def evaluate_rul(fh, gt, cfg):
    r = cfg["rul"]
    rows = []
    for eq, g in fh[fh["cycle"] == 1].groupby("equipment_id"):
        t, h = g["day"].to_numpy(float), g["health_obs"].to_numpy(float)
        t_fail = gt[eq]["t_fail"]
        for k in range(r["min_inspections_for_fit"], len(g) + 1):
            tk = t[k - 1]
            if t_fail is not None and tk >= t_fail:
                break  # already failed: RUL is no longer defined for this life cycle
            censored = t_fail is None or (t_fail - tk) > r["cap_days"]
            true_rul = r["cap_days"] if censored else t_fail - tk
            row = dict(equipment_id=eq, k=k, day=int(tk), health_obs=h[k - 1], true_rul=true_rul, censored=censored)
            for m in METHODS:
                pred, _ = predict_rul(t[:k], h[:k], m, r["failure_threshold"], r["cap_days"])
                row[f"pred_{m}"], row[f"abs_err_{m}"] = pred, abs(pred - true_rul)
            rows.append(row)
    return pd.DataFrame(rows)


def summarise(err, cfg):
    unc = err[~err["censored"]]
    summ = []
    for m in METHODS:
        e = unc[f"abs_err_{m}"]
        summ.append({"method": m, "n_eval_points": len(unc), "n_equipment": unc["equipment_id"].nunique(),
                     "MAE_days": e.mean(), "median_AE_days": e.median(),
                     "RMSE_days": float(np.sqrt((e ** 2).mean())),
                     "share_pred_capped": float((unc[f"pred_{m}"] >= cfg["rul"]["cap_days"]).mean()),
                     "MAE_days_incl_censored_capped": err[f"abs_err_{m}"].mean()})
    by_k = unc.groupby("k")[[f"abs_err_{m}" for m in METHODS]].agg(["mean", "count"])
    by_k.columns = [f"{a.replace('abs_err_', 'MAE_')}_{b}".replace("_mean", "") for a, b in by_k.columns]
    return pd.DataFrame(summ), by_k.reset_index()


def plot_samples(fh, gt, err, cfg):
    r, thr = cfg["rul"], cfg["rul"]["failure_threshold"]
    last = err[~err["censored"]].sort_values("k").groupby("equipment_id").tail(1)
    picks = last.sort_values("equipment_id").head(r["n_sample_curves"])
    n = len(picks)
    if n == 0:
        return
    cols = 3
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(11, 3.1 * np.ceil(n / cols)), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, row in zip(axes.ravel(), picks.itertuples()):
        g = fh[(fh["equipment_id"] == row.equipment_id) & (fh["cycle"] == 1)]
        t, h = g["day"].to_numpy(float)[:row.k], g["health_obs"].to_numpy(float)[:row.k]
        t_fail = gt[row.equipment_id]["t_fail"]
        tt = np.arange(0, t_fail + 120)
        ax.plot(tt, gt[row.equipment_id]["h_true_cycle1"][:len(tt)], color=MUTED, lw=1.2, label="true health (simulated)")
        ax.plot(g["day"], g["health_obs"], "o", color=SERIES[0], ms=5, alpha=0.35)
        ax.plot(t, h, "o", color=SERIES[0], ms=6, label="observed (used for fit)")
        for m, color, params in [("linear", SERIES[1], fit_linear(t, h)), ("exponential", SERIES[2], fit_exponential(t, h))]:
            ax.plot(tt, np.clip(predict_health(params, tt, m), 0, 100), "--", color=color, lw=1.5, label=f"{m} fit")
        ax.axhline(thr, color=SERIES[7], lw=1, ls=":")
        ax.axvline(t_fail, color=SERIES[7], lw=1)
        ax.set_ylim(0, 105)
        ax.set_title(f"{row.equipment_id}: true RUL {row.true_rul:.0f} d | lin {row.pred_linear:.0f} | "
                     f"exp {row.pred_exponential:.0f}", loc="left", fontsize=8)
        ax.set_xlabel("day")
        ax.set_ylabel("health")
    axes.ravel()[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Synthetic degradation: fits at the last inspection before true failure (red line = H 20 / failure day)",
                 x=0.01, ha="left", fontsize=9)
    savefig(fig, "results/figures/rul_sample_curves.png")


def plot_by_k(by_k):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for m, c in zip(METHODS, SERIES[1:3]):
        ax.plot(by_k["k"], by_k[f"MAE_{m}"], "o-", color=c, label=m)
    ax.set_xlabel("inspections used for the fit (k)")
    ax.set_ylabel("RUL MAE, days")
    ax.set_title("RUL error vs number of inspections (non-censored points)", loc="left")
    ax.legend()
    savefig(fig, "results/figures/rul_mae_by_k.png")


def from_images(args, cfg, scfg):
    from src.severity import assess_masks
    from src.vision.infer import load_model, predict
    model, err = load_model(args.weights or scfg["inference"]["weights"])
    if model is None:
        sys.exit(err)
    man = pd.read_csv(args.from_images, parse_dates=["timestamp"]).sort_values(["equipment_id", "timestamp"])
    rows = []
    for r in man.itertuples():
        out = predict(model, rpath(r.image_path), scfg["inference"]["conf"], scfg["inference"]["imgsz"])
        table, h, action = assess_masks(out["masks"], out["classes"], out["confs"], cfg=scfg)
        rows.append(dict(equipment_id=r.equipment_id, timestamp=r.timestamp, image_path=r.image_path,
                         n_defects=len(table), max_severity=table["severity"].max() if len(table) else 0.0,
                         health_obs=h, action=action))
    fh = pd.DataFrame(rows)
    fh["day"] = (fh["timestamp"] - fh["timestamp"].min()).dt.days
    fh.to_csv(ensure_dir("results/simulation") / "from_images_health.csv", index=False)
    res = []
    for eq, g in fh.groupby("equipment_id"):
        row = {"equipment_id": eq, "n_inspections": len(g), "last_health": g["health_obs"].iloc[-1]}
        for m in METHODS:
            row[f"rul_{m}_days"] = (predict_rul(g["day"].to_numpy(float), g["health_obs"].to_numpy(float), m,
                                                cfg["rul"]["failure_threshold"], cfg["rul"]["cap_days"])[0]
                                    if len(g) >= 2 else np.nan)
        res.append(row)
    save_table(pd.DataFrame(res), "rul_from_images")
    print("[DONE] from-images RUL -> results/tables/rul_from_images.csv (no ground truth in this mode)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/simulation.yaml")
    ap.add_argument("--from-images", default=None, help="CSV: equipment_id,timestamp,image_path")
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()
    cfg, scfg = load_yaml(args.config), load_config()
    if args.from_images:
        return from_images(args, cfg, scfg)

    fh, dr, gt = simulate(cfg, scfg)
    out = ensure_dir("results/simulation")
    fh.to_csv(out / "fleet_health.csv", index=False)
    dr.to_csv(out / "inspection_defects.csv", index=False)
    weekly_fleet_severity(fh, dr, cfg).to_csv(out / "fleet_severity_weekly.csv", index=False)
    pd.DataFrame([{"equipment_id": k, "true_failure_day_cycle1": v["t_fail"], "frailty": v["frailty"]}
                  for k, v in gt.items()]).to_csv(out / "ground_truth_failures.csv", index=False)

    err = evaluate_rul(fh, gt, cfg)
    save_table(err, "rul_errors")
    summ, by_k = summarise(err, cfg)
    save_table(summ, "rul_summary")
    save_table(by_k, "rul_mae_by_k")
    plot_samples(fh, gt, err, cfg)
    plot_by_k(by_k)
    fails = [v["t_fail"] for v in gt.values() if v["t_fail"] is not None]
    save_json({"synthetic": True, "n_equipment": cfg["n_equipment"], "n_inspections": len(fh),
               "n_overhauls": int(fh["overhaul_after"].sum()), "n_defects_observed": len(dr),
               "equipment_failing_within_horizon_plus_extra": len(fails),
               "median_true_failure_day": float(np.median(fails)) if fails else None,
               "rul_eval_points_total": len(err), "rul_eval_points_censored": int(err["censored"].sum()),
               "methods": summ.set_index("method").to_dict(orient="index"),
               "config": cfg}, "results/metrics/rul_summary.json")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
