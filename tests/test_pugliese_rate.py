"""PUGLIESE_RATE_MALECNS_V1 is a third model. Not Shiu LIF. Not the MANC Hydra run."""

import numpy as np
import pytest

from flybrain.loader import Connectome
from flybrain.neurons import (
    MALECNS_DNG100_L,
    MALECNS_DNG100_R,
    PUGLIESE_CPG_MODEL,
    PUGLIESE_CPG_STIM_AMPLITUDE,
    PUGLIESE_RATE_MALECNS_V1,
    SHIU_LIF_SANITY_MODEL,
    WSYN_MV,
)
from flybrain.pugliese_rate import (
    PuglieseRateParams,
    integrate_rate,
    make_input,
    rate_equation_half_tanh,
    rate_rhythmicity_score,
    rates_are_valid,
    reweight_connectivity,
    set_sizes,
    signed_weight_matrix,
    swc_frustum_volume,
)
from flybrain.neurons import nt_sign


def _chain(transmitters, anatomical=10.0):
    n = len(transmitters)
    ptr = np.zeros(n + 1, dtype=np.int64)
    ptr[1:] = np.minimum(np.arange(1, n + 1), n - 1)
    post = np.arange(1, n, dtype=np.uint32)
    weight = np.full(n - 1, anatomical, dtype=np.uint32)
    neuron_sign = np.array([nt_sign(name) for name in transmitters], dtype=np.int8)
    sign = np.empty(n - 1, dtype=np.int8)
    for i in range(n - 1):
        sign[ptr[i] : ptr[i + 1]] = neuron_sign[i]
    return Connectome(
        neuron_ids=np.arange(1, n + 1, dtype=np.uint64),
        pre_ptr=ptr,
        post=post,
        anatomical=weight,
        sign=sign,
        superclass=np.array(["descending_neuron"] + ["vnc_intrinsic"] * (n - 1), dtype=object),
        cell_type=np.array(["DNg100"] + [f"IN{i}" for i in range(n - 1)], dtype=object),
        cell_class=np.array([""] * n, dtype=object),
        side=np.array([""] * n, dtype=object),
        neurotransmitter=np.array(transmitters, dtype=object),
        report={"dataset_id": "unit_rate"},
    )


def test_three_model_ids_are_distinct():
    params = PuglieseRateParams()
    assert SHIU_LIF_SANITY_MODEL == "shiu_lif_sanity_v1"
    assert PUGLIESE_CPG_MODEL == "pugliese_cpg_v1"
    assert PUGLIESE_RATE_MALECNS_V1 == "pugliese_rate_malecns_v1"
    assert len({SHIU_LIF_SANITY_MODEL, PUGLIESE_CPG_MODEL, PUGLIESE_RATE_MALECNS_V1}) == 3
    assert params.model_id == PUGLIESE_RATE_MALECNS_V1
    assert params.stim_amplitude == PUGLIESE_CPG_STIM_AMPLITUDE == 250.0
    assert params.stim_amplitude != 40.0
    assert params.stim_amplitude != WSYN_MV
    assert MALECNS_DNG100_R == 10056
    assert MALECNS_DNG100_L == 10045
    assert MALECNS_DNG100_R != 10093


def test_size_normalization_divides_gain_and_multiplies_threshold():
    gain = np.ones(3)
    theta = np.full(3, 7.5)
    a, th = set_sizes(np.array([1.0, 2.0, 0.5]), gain, theta)
    # median is 1, so scales are 1, 2, 0.5
    assert a[0] == pytest.approx(1.0)
    assert a[1] == pytest.approx(0.5)
    assert a[2] == pytest.approx(2.0)
    assert th[0] == pytest.approx(7.5)
    assert th[1] == pytest.approx(15.0)
    assert th[2] == pytest.approx(3.75)


def test_reweight_transposes_pre_post_for_incoming_current():
    W = np.array([[0.0, 10.0], [0.0, 0.0]])  # pre 0 → post 1
    weighted = reweight_connectivity(W, 0.03, 0.03)
    assert weighted[1, 0] == pytest.approx(0.3)
    assert weighted[0, 1] == pytest.approx(0.0)
    R = np.array([5.0, 0.0])
    incoming = weighted @ R
    assert incoming[1] == pytest.approx(1.5)
    assert incoming[0] == pytest.approx(0.0)


def test_gaba_weights_are_negative_and_hyperpolarizing_in_the_rate_rhs():
    graph = _chain(["acetylcholine", "gaba", "acetylcholine"], anatomical=20)
    W = signed_weight_matrix(graph)
    assert W[0, 1] > 0
    assert W[1, 2] < 0
    weighted = reweight_connectivity(W, 0.03, 0.03)
    n = graph.n
    tau = np.full(n, 0.02)
    gain = np.ones(n)
    theta = np.full(n, 0.0)
    fr = np.full(n, 200.0)
    # Inhibitory input onto an already-active cell must drive its rate down
    # (half-tanh cannot go negative, so rest stays at 0).
    R = np.array([0.0, 50.0, 40.0])
    dR = rate_equation_half_tanh(
        0.1,
        R,
        inputs=np.zeros(n),
        pulse_start=0.02,
        pulse_end=1.999,
        tau=tau,
        weighted_W=weighted,
        threshold=theta,
        gain=gain,
        fr_cap=fr,
    )
    assert dR[2] < 0


def test_positive_pulse_raises_driven_cell_rate():
    n = 1
    params = PuglieseRateParams()
    rates = integrate_rate(
        np.zeros((n, n)),
        tau=np.array([0.02]),
        gain=np.array([1.0]),
        threshold=np.array([7.5]),
        fr_cap=np.array([200.0]),
        inputs=make_input(n, [0], 250.0),
        params=params,
    )
    assert rates_are_valid(rates)
    assert rates[0, 0] == pytest.approx(0.0)
    # After the 20 ms pulse onset the isolated cell must leave rest.
    after = int(params.pulse_start / params.dt) + 50
    assert rates[0, after] > 1.0


def test_rate_rhythmicity_does_not_apply_millivolt_gate():
    t = np.arange(1000)
    trace = 80 + 20 * np.sin(2 * np.pi * 11 * t / 1000.0)
    scored = rate_rhythmicity_score(trace, 0.001)
    assert scored["voltage_gate_applied"] is False
    assert scored["fft_executed"] is True
    assert scored["fft_hz"] == pytest.approx(11.0, abs=0.5)
    assert scored["model_id"] == PUGLIESE_RATE_MALECNS_V1


def test_synthetic_rate_circuit_is_not_biological_validation():
    from experiment.pugliese_rate_malecns import _e1_e2_rhythm

    fake = {
        "E1": {"rhythmicity_score": 0.9, "authors_oscillation": None},
        "E2": {"rhythmicity_score": 0.9, "authors_oscillation": None},
    }
    assert _e1_e2_rhythm(fake)
    fake["E2"]["rhythmicity_score"] = 0.1
    assert _e1_e2_rhythm(fake) is False


def test_primary_stim_is_malecns_10056_not_10045_or_manc_10093(tmp_path):
    from experiment.pugliese_rate_malecns import run

    with pytest.raises(ValueError, match="10056"):
        run(tmp_path / "nope", stim_body=10045)
    with pytest.raises(ValueError, match="10056"):
        run(tmp_path / "nope", stim_body=10093)


def test_rate_transfer_refuses_existing_output(tmp_path):
    from experiment.pugliese_rate_malecns import run

    out = tmp_path / "exists"
    out.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        run(out, stim_body=10056)


def test_full_malecns_remains_locked(monkeypatch):
    from experiment import dng100_cpg_rhythm as experiment

    monkeypatch.setattr(experiment, "load_graph", lambda *a, **k: pytest.fail("full graph loaded"))
    with pytest.raises(RuntimeError, match="locked"):
        experiment.run(connectome="malecns")


def test_restricted_graph_primary_transfer_excludes_left_dng100():
    from experiment.restricted_cpg import restricted_cpg_graph

    graph = restricted_cpg_graph(dng100_bodies=(10056,))
    dng = graph.neuron_ids[graph.cell_type == "DNg100"]
    assert list(dng.astype(int)) == [10056]
    assert 10045 not in set(graph.neuron_ids.astype(int))
    assert graph.n < 2000
    assert "motor feedback omitted" in graph.report["subset"]


def test_swc_volume_is_positive_for_dng100_r():
    from pathlib import Path

    path = Path("data/malecns_v1/skeletons-swc/10056.swc")
    if not path.exists():
        pytest.skip("MaleCNS SWC skeletons are not in this checkout")
    vol = swc_frustum_volume(path)
    assert np.isfinite(vol) and vol > 0
