"""Guards on the I=400 DNg100 runner. No integration here; the battery does that."""

from pathlib import Path

import numpy as np
import pytest

from experiment import pugliese_rate_mcns_dng100 as runner
from flybrain.authors_mcns import (
    AUTHORS_MCNS_DNG100_STIM_I,
    AUTHORS_MCNS_STIM_CITATION,
    WTABLE_PATH,
)

MANC = Path(__file__).resolve().parents[1] / "outputs" / "pugliese_rate_manc_portcheck_20260919T133800Z"
needs_export = pytest.mark.skipif(
    not WTABLE_PATH.exists() or not MANC.exists(), reason="authors export or MANC port check missing"
)


def test_published_cns_amplitude_and_citation():
    assert AUTHORS_MCNS_DNG100_STIM_I == 400.0
    assert "400 in the" in AUTHORS_MCNS_STIM_CITATION
    assert "250 " in AUTHORS_MCNS_STIM_CITATION  # MANC value cross-checks the shipped yaml


@needs_export
def test_refuses_non_dng100_stimulus(tmp_path):
    with pytest.raises(ValueError):
        runner.run(tmp_path / "x", manc_portcheck=MANC, stim_bodies=(12044,), replicate_indices=[0])


@needs_export
def test_refuses_unknown_role(tmp_path):
    with pytest.raises(ValueError):
        runner.run(tmp_path / "x", manc_portcheck=MANC, silence_roles=("E9",), replicate_indices=[0])


@needs_export
def test_refuses_existing_output(tmp_path):
    with pytest.raises(FileExistsError):
        runner.run(tmp_path, manc_portcheck=MANC, replicate_indices=[0])


def test_paper_metrics_on_synthetic_oscillation():
    pytest.importorskip("jax")
    score_fn = runner._authors_score()
    dt = 0.001
    t = np.arange(2001) * dt
    rates = np.zeros((4, t.size))
    rates[1] = 5.0 * (1 + np.sin(2 * np.pi * 11.0 * t))  # MN, 11 Hz
    rates[2] = 3.0  # MN, tonic
    rates[3] = 8.0 * (1 + np.sin(2 * np.pi * 11.0 * t))  # not a MN
    m = runner.paper_metrics(rates, np.array([0, 1, 2]), dt, score_fn)
    assert m["n_mn_active_past_transient"] == 2
    assert m["n_active_total"] == 3
    assert m["mn_frequency_hz"] == pytest.approx(11.0, rel=0.05)
    assert 0.3 < m["mn_rhythmicity_score"] < 0.7  # one rhythmic MN of two active


def test_sparse_matches_dense_integration():
    from scipy.sparse import csr_matrix

    from flybrain.pugliese_rate import PuglieseRateParams, integrate_rate, reweight_connectivity

    rng = np.random.default_rng(0)
    n = 60
    W = rng.normal(0, 30, (n, n)) * (rng.random((n, n)) < 0.08)
    p = PuglieseRateParams(stim_amplitude=400.0, T=0.3)
    w = reweight_connectivity(W, p.exc_multiplier, p.inh_multiplier)
    kw = dict(
        tau=np.full(n, 0.02),
        gain=np.ones(n),
        threshold=np.full(n, 7.5),
        fr_cap=np.full(n, 200.0),
        inputs=np.where(np.arange(n) == 0, 400.0, 0.0),
        params=p,
    )
    dense = integrate_rate(w, **kw)
    sparse = integrate_rate(csr_matrix(w), **kw)
    assert dense.max() > 1.0
    # Summation order differs, so adaptive RK45 picks slightly different
    # steps; agreement is at integrator-error level, not bitwise.
    np.testing.assert_allclose(sparse, dense, rtol=1e-3, atol=1e-2)
    assert np.corrcoef(sparse.ravel(), dense.ravel())[0, 1] > 0.99999
