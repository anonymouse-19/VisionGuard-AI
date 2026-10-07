"""A3 tests: severity bounds, health bounds, action thresholds."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.severity import health_score, load_config, maintenance_action, severity  # noqa: E402

CFG = load_config()


@pytest.mark.parametrize("cls", list(CFG["w_type"]))
def test_severity_bounds(cls):
    rng = np.random.default_rng(42)
    area, conf, wl = rng.uniform(-0.5, 2, 500), rng.uniform(-0.5, 1.5, 500), rng.uniform(0, 3, 500)
    s = severity([cls] * 500, area, conf, wl, CFG)
    assert np.all((s >= 0) & (s <= 1))
    assert severity(cls, 0.0, 1.0, 1.0, CFG) == 0.0
    assert severity(cls, 0.02, 0.9, 1.0, CFG) <= severity(cls, 0.05, 0.9, 1.0, CFG)  # monotone in area


def test_unknown_class_rejected():
    with pytest.raises(ValueError):
        severity("rust_spot", 0.1)


def test_health_bounds():
    assert health_score([]) == 100.0
    assert health_score([1.0, 0.2]) == 0.0
    rng = np.random.default_rng(42)
    for n in [1, 5, 50, 500]:
        h = health_score(rng.uniform(-1, 2, n))  # out-of-range severities are clipped
        assert 0.0 <= h <= 100.0
    assert health_score([0.5] * 10) > 0  # many moderate defects never go negative (old 100 - sum(S) would)


@pytest.mark.parametrize("health,action", [
    (100, "Routine Monitoring"), (80, "Routine Monitoring"), (79.99, "Schedule Maintenance"),
    (50, "Schedule Maintenance"), (49.99, "Urgent Intervention"), (20, "Urgent Intervention"),
    (19.99, "Critical/Withdraw"), (0, "Critical/Withdraw")])
def test_action_thresholds(health, action):
    assert maintenance_action(health, CFG) == action
