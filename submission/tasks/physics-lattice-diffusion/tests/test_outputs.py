import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.environ.get("APP_DIR", "/app")
sys.path.insert(0, HERE)

import oracle  # noqa: E402

T0 = 300.0
BASE = dict(alpha=0.003, gamma=8.0, q0=5.0e8)
SEEDS = [
    dict(alpha=0.0025, gamma=7.5, q0=5.0e8),
    dict(alpha=0.0035, gamma=8.5, q0=4.5e8),
    dict(alpha=0.002, gamma=7.0, q0=5.5e8),
]


def compare_field(T, Tref):
    T, Tref = np.asarray(T, float), np.asarray(Tref, float)
    assert T.shape == Tref.shape == (100, 100)
    assert np.all(np.isfinite(T))
    rise = np.max(np.abs(T - T0)) - np.max(np.abs(Tref - T0))
    assert abs(rise) / np.max(Tref - T0) <= 1e-3
    assert np.linalg.norm(T - Tref) / np.linalg.norm(Tref) <= 1e-3


@pytest.fixture(scope="module")
def ref():
    P, T, Qc, Tc = oracle.solve(**BASE)
    return dict(T=T.reshape(100, 100), Qc=Qc, Tc=Tc, S=oracle.stability_index(P, T, BASE["q0"]))


@pytest.fixture(scope="module")
def result():
    path = os.path.join(APP, "solution_metrics.json")
    assert os.path.exists(path), "solution_metrics.json not found"
    with open(path) as f:
        return json.load(f)


def test_schema(result):
    for key in ("T_field", "Q_crit", "T_crit", "S", "mms_slope"):
        assert key in result, f"missing key {key}"
    for key in ("Q_crit", "T_crit", "S", "mms_slope"):
        assert np.isfinite(result[key])


def test_temperature_field(result, ref):
    compare_field(result["T_field"], ref["T"])


def test_critical_source(result, ref):
    assert abs(result["Q_crit"] - ref["Qc"]) / ref["Qc"] <= 1.5e-2


def test_critical_temperature(result, ref):
    assert abs(result["T_crit"] - ref["Tc"]) / (ref["Tc"] - T0) <= 2e-2


def test_stability_index(result, ref):
    assert result["S"] < 0
    assert abs(result["S"] - ref["S"]) / abs(ref["S"]) <= 2e-2


def test_mms_slope(result):
    assert abs(result["mms_slope"] - 2.0) <= 0.15


@pytest.mark.parametrize("seed", SEEDS)
def test_hidden_parameters(seed):
    script = os.path.join(APP, "solve_lattice.py")
    assert os.path.exists(script), "solve_lattice.py not found in /app"
    P, T, Qc, Tc = oracle.solve(**seed)
    S_ref = oracle.stability_index(P, T, seed["q0"])
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "out.json")
        cmd = [sys.executable, script, "--alpha", str(seed["alpha"]),
               "--gamma", str(seed["gamma"]), "--q0", str(seed["q0"]), "--out", out]
        subprocess.run(cmd, cwd=APP, check=True, timeout=1500)
        with open(out) as f:
            res = json.load(f)
    compare_field(res["T_field"], T.reshape(100, 100))
    assert abs(res["S"] - S_ref) / abs(S_ref) <= 2e-2