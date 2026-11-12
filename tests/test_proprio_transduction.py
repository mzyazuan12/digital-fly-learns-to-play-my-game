"""Stage 2B plan step 5: body state -> afferent input current (design §4.3, §5.2, §8; amendments A1.5, A3).

From the design's step-5 row: u = 0 gives I = 0; the response is
size-invariant; u = 1 gives a median rate of 100 +- 10 Hz over one replicate's
parameter draws; the stored normalisation is reproduced from the clip (and,
for the campaniform proxy only, from a replay), with left and right frames
pooled (amendment A4). Plus: the hair-plate direction
is a required parameter covering the 2 x 2 factorial; afferent rows, legs and
channels follow data/paper2/stage2B_sensory_table.json.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from organism.proprio_transduction import (
    CHANNELS,
    DOF_INDEX,
    FACTORIAL,
    NORM_PATH,
    SENSORY_TABLE_PATH,
    ProprioTransducer,
    clip_feature_stats,
    gain_for_rate,
    steady_rate,
)

pytestmark = pytest.mark.skipif(not NORM_PATH.exists(), reason="run: python -m organism.proprio_transduction")


@pytest.fixture(scope="module")
def norm():
    return json.loads(NORM_PATH.read_text())


@pytest.fixture(scope="module")
def table():
    return json.loads(SENSORY_TABLE_PATH.read_text())


def _rest_state(norm):
    """Every stimulus zero: hair-plate angles at the median (between p20 and p80),
    tibia at its median with zero velocity, zero force."""
    state = {}
    for leg, bl in (("T1L", "lf"), ("T1R", "rf")):
        st = norm["stats"]  # pooled over both legs (A4)
        ang = np.zeros(7)
        ang[DOF_INDEX["coxa_pitch"]] = st["coxa_pitch_angle"]["p50"]
        ang[DOF_INDEX["trfe_pitch"]] = st["trfe_pitch_angle"]["p50"]
        ang[DOF_INDEX["tibia_pitch"]] = st["tibia_pitch_angle_p50"]
        state[bl] = {"angle": ang, "velocity": np.zeros(7), "actuator_force": np.zeros(7)}
    return state


def _unit_state(norm, tr):
    """Every stimulus exactly 1 for transducer ``tr``."""
    state = _rest_state(norm)
    for leg, bl in (("T1L", "lf"), ("T1R", "rf")):
        st = norm["stats"]  # pooled over both legs (A4)
        for ch, dof, key in (("hair_plate_coxa", "coxa_pitch", "coxa_pitch_angle"),
                             ("hair_plate_trochanter", "trfe_pitch", "trfe_pitch_angle")):
            d = tr.angle_direction[ch][leg]
            state[bl]["angle"][DOF_INDEX[dof]] = st[key]["p95"] if d > 0 else st[key]["p5"]
        state[bl]["angle"][DOF_INDEX["tibia_pitch"]] = st["tibia_pitch_angle_p50"] + st["tibia_pitch_abs_dev_p95"]
        state[bl]["velocity"][DOF_INDEX["tibia_pitch"]] = -st["tibia_pitch_abs_vel_p95"]
        state[bl]["actuator_force"][DOF_INDEX["trfe_pitch"]] = st["trfe_pitch_abs_force_p95"]
    return state


# --------------------------------------------------------------------------- direction parameter
def test_hair_plate_direction_is_required_and_factorial_has_four_unranked_combinations():
    with pytest.raises(TypeError):
        ProprioTransducer()  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        ProprioTransducer(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ProprioTransducer({"hair_plate_coxa": "protraction"})
    with pytest.raises(ValueError):
        ProprioTransducer({"hair_plate_coxa": "protraction", "hair_plate_trochanter": "up"})
    assert len(FACTORIAL) == 4
    assert len({tuple(sorted(c.items())) for c in FACTORIAL}) == 4
    assert all(set(c) == {"hair_plate_coxa", "hair_plate_trochanter"} for c in FACTORIAL)


@pytest.mark.parametrize("combo", FACTORIAL, ids=lambda c: f"{c['hair_plate_coxa']}-{c['hair_plate_trochanter']}")
def test_zero_stimulus_gives_zero_current(norm, combo):
    tr = ProprioTransducer(combo)
    I = tr.currents(_rest_state(norm))
    assert I.shape == (4310,)
    assert np.all(I == 0.0)


def test_direction_selects_the_limit(norm):
    """A protraction-limit coxa hair plate fires at the protraction extreme and not at the
    retraction extreme, and the reverse; u = 0 at onset (80th pct) and 1 at the 95th."""
    pro = ProprioTransducer({"hair_plate_coxa": "protraction", "hair_plate_trochanter": "flexion"})
    ret = ProprioTransducer({"hair_plate_coxa": "retraction", "hair_plate_trochanter": "extension"})
    for leg, bl in (("T1L", "lf"), ("T1R", "rf")):
        pct = norm["stats"]["coxa_pitch_angle"]
        d = pro.angle_direction["hair_plate_coxa"][leg]
        assert ret.angle_direction["hair_plate_coxa"][leg] == -d
        pro_end = pct["p95"] if d > 0 else pct["p5"]
        ret_end = pct["p5"] if d > 0 else pct["p95"]
        onset = pct["p80"] if d > 0 else pct["p20"]
        for theta, expect_pro, expect_ret in ((pro_end, 1.0, 0.0), (ret_end, 0.0, 1.0), (onset, 0.0, 0.0)):
            s = _rest_state(norm)
            s[bl]["angle"][DOF_INDEX["coxa_pitch"]] = theta
            assert pro.stimuli(s)[(leg, "hair_plate_coxa")] == pytest.approx(expect_pro, abs=1e-12)
            assert ret.stimuli(s)[(leg, "hair_plate_coxa")] == pytest.approx(expect_ret, abs=1e-12)


def test_anatomical_directions_use_the_geometric_signs():
    mj = json.loads((NORM_PATH.parent / "stage2B_mn_joint_table.json").read_text())
    tr = ProprioTransducer({"hair_plate_coxa": "protraction", "hair_plate_trochanter": "flexion"})
    for leg in ("T1L", "T1R"):
        assert tr.angle_direction["hair_plate_coxa"][leg] == mj["channels"]["coxa_pitch"]["agonist_angle_sign"][leg]
        assert tr.angle_direction["hair_plate_trochanter"][leg] == mj["channels"]["trfe_pitch"]["agonist_angle_sign"][leg]


# --------------------------------------------------------------------------- rows, legs, channels
def test_rows_and_sizes_follow_the_sensory_table(table):
    from flybrain.authors_mcns import load_authors_wtable

    tr = ProprioTransducer(FACTORIAL[0])
    counts = {f"{leg}:{ch}": int(idx.size) for (leg, ch), (idx, _) in tr.groups.items()}
    assert counts == table["included_counts"]
    w = load_authors_wtable()
    med = float(np.nanmedian(w["size"].to_numpy(dtype=float)))
    for a in table["afferents"]:
        if a["included_2B"]:
            assert w.iloc[a["row"]]["bodyId"] == a["bodyId"]
            assert a["normalized_size"] == pytest.approx(float(w.iloc[a["row"]]["size"]) / med, rel=1e-12)


def test_each_leg_reads_only_its_own_state(norm):
    tr = ProprioTransducer(FACTORIAL[1])
    base = _unit_state(norm, tr)
    I0 = tr.currents(base)
    left_rows = np.concatenate([idx for (leg, _), (idx, _) in tr.groups.items() if leg == "T1L"])
    right_rows = np.concatenate([idx for (leg, _), (idx, _) in tr.groups.items() if leg == "T1R"])
    included = np.concatenate([left_rows, right_rows])
    assert np.all(I0[np.setdiff1d(np.arange(4310), included)] == 0.0)
    assert np.all(I0[included] > 0.0)
    moved = {k: {kk: vv.copy() for kk, vv in v.items()} for k, v in base.items()}
    moved["rf"]["angle"] += 0.3
    moved["rf"]["velocity"] += 5.0
    moved["rf"]["actuator_force"] += 3.0
    I1 = tr.currents(moved)
    assert np.array_equal(I0[left_rows], I1[left_rows])
    assert not np.array_equal(I0[right_rows], I1[right_rows])


# --------------------------------------------------------------------------- scale and units
def test_gain_formula():
    assert gain_for_rate(100.0) == pytest.approx(1 + (200 / 7.5) * np.arctanh(0.5))
    assert gain_for_rate(100.0) == pytest.approx(15.648, abs=1e-3)
    for r in (50.0, 100.0, 199.0):
        assert float(steady_rate(7.5 * gain_for_rate(r), 1.0, 7.5, 200.0)) == pytest.approx(r, rel=1e-9)
    with pytest.raises(ValueError):
        gain_for_rate(200.0)


def test_response_is_size_invariant():
    """Same u -> same steady rate for any afferent size, at nominal parameters
    (gain a / s, threshold 7.5 s, as set_sizes applies)."""
    g = gain_for_rate(100.0)
    u = np.linspace(0, 2, 41)
    rates = [steady_rate(g * 7.5 * s * u, 1.0 / s, 7.5 * s, 200.0) for s in (0.02, 0.1, 1.0, 3.0)]
    for r in rates[1:]:
        assert np.allclose(r, rates[0], rtol=1e-12, atol=1e-9)
    assert rates[0][0] == 0.0 and rates[0][20] == pytest.approx(100.0, rel=1e-9)


def test_unit_stimulus_gives_100_hz_median_over_a_replicates_draws(norm):
    """u = 1 on every channel -> median steady rate of the 31 included afferents,
    with replicate 0 of a seed-1 parameter draw after set_sizes, within 100 +- 10 Hz.
    Presynaptic input from the network is not included (steady state, network silent)."""
    from flybrain.authors_mcns import load_authors_wtable
    from flybrain.pugliese_rate import PuglieseRateParams, sample_cell_parameters_block, set_sizes

    w = load_authors_wtable()
    drawn = sample_cell_parameters_block(4310, 1, 1, PuglieseRateParams(stim_amplitude=400.0))
    gain, theta = set_sizes(w["size"].to_numpy(dtype=float), drawn["gain"], drawn["threshold"])
    for r_star in (100.0, 50.0):  # primary r* and the lower R2 value
        tr = ProprioTransducer(FACTORIAL[2], r_star_hz=r_star)
        state = _unit_state(norm, tr)
        u = tr.stimuli(state)
        assert all(v == pytest.approx(1.0, abs=1e-12) for v in u.values()), u
        I = tr.currents(state)
        rows = np.concatenate([idx for idx, _ in tr.groups.values()])
        rates = steady_rate(I[rows], gain[0, rows], theta[0, rows], drawn["fr_cap"][0, rows])
        tol = 0.1 * r_star
        assert abs(float(np.median(rates)) - r_star) <= tol, (r_star, np.median(rates))


# --------------------------------------------------------------------------- normalisation
def test_clip_percentiles_are_reproduced_from_pooled_frames(norm):
    """Thresholds come from LF and RF clip frames pooled (amendment A4), so both legs share them."""
    pytest.importorskip("flygym")
    from organism.flybody_tethered import load_walking_clip_front_legs

    clip = load_walking_clip_front_legs()
    fresh = clip_feature_stats(clip, float(clip["timestep"]))
    for k, v in fresh.items():
        assert norm["stats"][k] == v, k
    pooled = np.concatenate([clip["lf"][:, DOF_INDEX["coxa_pitch"]], clip["rf"][:, DOF_INDEX["coxa_pitch"]]])
    assert norm["stats"]["coxa_pitch_angle"]["p50"] == float(np.percentile(pooled, 50))
    # The per-leg values are kept for reference only and do differ (the reason for A4).
    ref = norm["per_leg_reference_only"]
    assert abs(ref["T1L"]["coxa_pitch_angle"]["p50"] - ref["T1R"]["coxa_pitch_angle"]["p50"]) > 0.2


def test_both_legs_use_the_same_thresholds(norm):
    tr = ProprioTransducer(FACTORIAL[0])
    s = _rest_state(norm)
    theta = norm["stats"]["coxa_pitch_angle"]["p95"] if tr.angle_direction["hair_plate_coxa"]["T1L"] > 0 else norm["stats"]["coxa_pitch_angle"]["p5"]
    for bl in ("lf", "rf"):
        s[bl]["angle"][DOF_INDEX["coxa_pitch"]] = theta
    u = tr.stimuli(s)
    assert u[("T1L", "hair_plate_coxa")] == u[("T1R", "hair_plate_coxa")] == pytest.approx(1.0)


def test_campaniform_replay_percentile_is_reproduced(norm):
    pytest.importorskip("flygym")
    from organism.flybody_tethered import TetheredFlyBody, load_walking_clip_front_legs
    from organism.proprio_transduction import replay_force_stats

    fresh = replay_force_stats(TetheredFlyBody("position"), load_walking_clip_front_legs())
    assert norm["stats"]["trfe_pitch_abs_force_p95"] == fresh["trfe_pitch_abs_force_p95"]


def test_every_channel_has_a_positive_normaliser(norm):
    st = norm["stats"]
    for key in ("coxa_pitch_angle", "trfe_pitch_angle"):
        p = st[key]
        assert p["p5"] < p["p20"] < p["p50"] < p["p80"] < p["p95"]
    for key in ("tibia_pitch_abs_dev_p95", "tibia_pitch_abs_vel_p95", "trfe_pitch_abs_force_p95"):
        assert st[key] > 0
    assert set(CHANNELS) == {"hair_plate_coxa", "hair_plate_trochanter", "chordotonal_femur_tibia", "campaniform_trochanter"}
