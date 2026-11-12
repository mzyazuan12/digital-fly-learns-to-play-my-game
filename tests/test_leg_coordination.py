"""Read-outs must recover known phases before they are trusted on simulations."""

import numpy as np
import pytest

from experiment import leg_coordination as lc


def _wave(f, phase_cycles, n=1751, dt=0.001, noise=0.0, seed=0):
    t = np.arange(n) * dt
    x = 1 + np.sin(2 * np.pi * (f * t - phase_cycles))
    if noise:
        x = x + np.random.default_rng(seed).normal(0, noise, n)
    return np.clip(x, 0, None)


@pytest.mark.parametrize("phase", [0.1, 0.25, 0.5, 0.8])
def test_lag_recovers_known_phase(phase):
    a = _wave(11.0, 0.0)
    b = _wave(11.0, phase)
    p = lc.period_autocorr(a)
    assert p == pytest.approx(1000 / 11.0, rel=0.03)
    lag = lc.lag_cycles(a, b, p)
    d = (lag - phase + 0.5) % 1.0 - 0.5
    assert abs(d) < 0.03


def test_plv_high_for_locked_low_for_detuned():
    a = _wave(11.0, 0.0, noise=0.05)
    locked = _wave(11.0, 0.3, noise=0.05, seed=1)
    detuned = _wave(14.0, 0.3, noise=0.05, seed=2)
    assert lc.plv(a, locked, 11.0) > 0.95
    assert lc.plv(a, detuned, 11.0) < 0.5


def test_circ_stats():
    s = lc.circ_stats([0.49, 0.5, 0.51, None])
    assert s["n"] == 3 and s["mean_cycles"] == pytest.approx(0.5, abs=1e-6) and s["R"] > 0.99
    s = lc.circ_stats(list(np.linspace(0, 1, 50, endpoint=False)))
    assert s["R"] < 0.05


def test_null_breaks_locking():
    rng = np.random.default_rng(0)
    reps = []
    for k in range(40):
        ph = rng.random()
        reps.append({"T1L": _wave(11.0, ph, seed=k), "T1R": _wave(11.0, ph + 0.5, seed=k + 100)})
    real = lc.circ_stats(
        [lc.lag_cycles(r["T1L"], r["T1R"], lc.period_autocorr(r["T1L"])) for r in reps]
    )
    assert real["R"] > 0.95 and real["mean_cycles"] == pytest.approx(0.5, abs=0.03)
    null = lc.null_pairs(reps, np.random.default_rng(1))
    assert lc.circ_stats(null["T1L-T1R"]["lag"])["R"] < 0.5
