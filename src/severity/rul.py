"""A4: degradation fits on Health(t) and remaining useful life (RUL) to a failure threshold.

linear:       H(t) = a + b t                      (b < 0 means degrading)
exponential:  H(t) = 100 - D0 * exp(r t)          (damage D = 100 - H grows exponentially; standard PHM form)
RUL = t* - t_now where H(t*) = threshold; inf when the fit shows no degradation (callers cap it).
"""
import numpy as np


def fit_linear(t, h):
    b, a = np.polyfit(np.asarray(t, float), np.asarray(h, float), 1)
    return {"a": float(a), "b": float(b)}


def fit_exponential(t, h):
    """Least squares on log(D) for points with visible damage (D > 0.5); needs >= 2 such points."""
    t, d = np.asarray(t, float), 100.0 - np.asarray(h, float)
    ok = d > 0.5
    if ok.sum() < 2:
        return None
    r, log_d0 = np.polyfit(t[ok], np.log(d[ok]), 1)
    return {"d0": float(np.exp(log_d0)), "r": float(r)}


def predict_health(params, t, method):
    t = np.asarray(t, float)
    if params is None:
        return np.full_like(t, np.nan)
    if method == "linear":
        return params["a"] + params["b"] * t
    return np.clip(100.0 - params["d0"] * np.exp(params["r"] * t), -1e9, 100.0)


def time_to_threshold(params, method, threshold=20.0):
    if params is None:
        return np.inf
    if method == "linear":
        return (threshold - params["a"]) / params["b"] if params["b"] < 0 else np.inf
    if params["r"] <= 0:
        return np.inf
    return np.log((100.0 - threshold) / params["d0"]) / params["r"]


def predict_rul(t, h, method, threshold=20.0, cap=None):
    """Fit on observations (t, h) and return (rul_from_last_observation, params). Clipped to [0, cap]."""
    params = fit_linear(t, h) if method == "linear" else fit_exponential(t, h)
    rul = time_to_threshold(params, method, threshold) - float(np.max(t))
    rul = max(rul, 0.0)
    if h[-1] < threshold:
        rul = 0.0
    if cap is not None:
        rul = min(rul, cap)
    return float(rul), params
