"""Stage 2B plan step 7: calibration and freeze (design §5, §8; amendments A1-A4).

- Calibration code imports and calls no phase/lag/coordination function (AST ban).
- Only replicates 960-1023 are accepted; the test set 0-127 is refused.
- The output is never overwritten.
- Acceptance logic: +-0.1 rad mean, +-20% range, undriven DOF = fail.
- Every channel centred on its median drive; R2 r* = 50 / 150 Hz (amendment A5).
- At most 8 correction rounds (amendment A6).
- Drive statistics, pooled targets (A4), and the replay path, which runs the
  bridge's own command arithmetic on recorded drive.
- A one-replicate end-to-end run reproduces the co-simulation's deafferented
  body bit for bit.
- The frozen file, if present, is internally consistent and matches current data.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from experiment import paper2_calibrate_2B as cal

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "data" / "paper2" / "stage2B_bridge_frozen.json"


def test_calibration_code_has_no_coordination_readout():
    tree = ast.parse((ROOT / "experiment" / "paper2_calibrate_2B.py").read_text())
    banned_modules = {"experiment.leg_coordination", "scipy.signal", "experiment.pugliese_rate_mcns_dng100"}
    banned_names = {"lag_cycles", "plv", "period_autocorr", "circ_stats", "replicate_readouts", "null_pairs",
                    "hilbert", "correlate", "csd", "coherence", "angle", "unwrap", "compute_oscillation_score",
                    "phase_scramble"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module not in banned_modules, node.module
            assert not {a.name for a in node.names} & banned_names
        elif isinstance(node, ast.Import):
            assert not {a.name for a in node.names} & banned_modules
        elif isinstance(node, ast.Attribute):
            assert node.attr not in banned_names, node.attr
        elif isinstance(node, ast.Name):
            assert node.id not in banned_names, node.id


def test_only_calibration_replicates_are_accepted():
    assert cal.CALIBRATION_REPLICATES == tuple(range(960, 1024))
    assert not set(cal.CALIBRATION_REPLICATES) & set(cal.TEST_REPLICATES)
    assert cal.check_replicates([960, 1023]) == (960, 1023)
    for bad in ([0], [127], [959], [1024], [960, 5]):
        with pytest.raises(ValueError):
            cal.check_replicates(bad)


def test_refuses_to_overwrite(tmp_path):
    out = tmp_path / "frozen.json"
    out.write_text("{}")
    with pytest.raises(FileExistsError):
        cal.main(["--out", str(out), "--replicates", "960:961"])
    assert out.read_text() == "{}"


def _targets():
    return {"mean": np.full(7, 0.5), "range": np.full(7, 1.0)}


def test_acceptance_logic():
    driven = np.ones((2, 7), dtype=bool)
    ok = {"mean": np.full((2, 7), 0.55), "range": np.full((2, 7), 1.15)}
    assert cal.acceptance(ok, _targets(), driven)["accepted"]
    off = {"mean": np.full((2, 7), 0.5), "range": np.full((2, 7), 1.0)}
    off["mean"][0, 2] = 0.611
    a = cal.acceptance(off, _targets(), driven)
    assert not a["accepted"] and not a["criteria"]["T1L:coxa_pitch"]["mean_pass"]
    off = {"mean": np.full((2, 7), 0.5), "range": np.full((2, 7), 1.0)}
    off["range"][1, 4] = 0.79
    a = cal.acceptance(off, _targets(), driven)
    assert not a["accepted"] and not a["criteria"]["T1R:trfe_pitch"]["range_pass"]
    und = driven.copy()
    und[1, 4] = False
    a = cal.acceptance({"mean": np.full((2, 7), 0.5), "range": np.full((2, 7), 1.0)}, _targets(), und)
    assert not a["accepted"] and not a["criteria"]["T1R:trfe_pitch"]["pass"]
    assert set(a["criteria"]) == {"T1L:coxa_pitch", "T1L:trfe_pitch", "T1R:coxa_pitch", "T1R:trfe_pitch"}


def test_drive_stats_centre_every_channel_on_the_median():
    """Amendment A5: c is the median drive of every channel (driven or not); s is the
    5-95% half-range, with the driven-DOF median s as fallback for undriven DOFs."""
    rng = np.random.default_rng(0)
    n = 5000
    u = np.zeros((n, 2, 7))
    u[:, 0, 2] = 8 + 10 * np.sin(np.arange(n) * 0.01)  # driven
    sparse = np.zeros(n)
    sparse[::7] = 20.0  # sparse bursts: mean 2.9 Hz, median 0
    u[:, 1, 4] = sparse
    u[:, 1, 3] = 0.2 * rng.standard_normal(n) + 0.05  # below 0.5 Hz half-range -> undriven
    u[:, 0, 0] = 50 * rng.standard_normal(n)  # held DOF: never driven
    ds = cal.drive_stats({960: u, 961: u})
    post = u[cal.TRANSIENT_TICKS:]
    assert ds["c"][0, 2] == pytest.approx(np.median(post[:, 0, 2]))
    assert ds["c_mean"][0, 2] == pytest.approx(post[:, 0, 2].mean())
    assert ds["c"][1, 4] == 0.0 and ds["c_mean"][1, 4] > 2.0  # median, not mean, for sparse drive
    assert ds["c"][1, 3] == pytest.approx(np.median(post[:, 1, 3]))  # undriven channels too
    assert np.array_equal(ds["c"], ds["c_median"])
    assert ds["driven"][0, 2] and not ds["driven"][1, 3] and not ds["driven"][0, 0]
    assert ds["s"][1, 3] == ds["s_fallback_hz"]


def test_correction_limit_follows_a6():
    assert cal.MAX_CORRECTIONS == 8
    import inspect

    assert inspect.signature(cal.calibrate).parameters["max_corrections"].default == 8


def test_r2_rates_follow_a5():
    from organism.proprio_transduction import gain_for_rate

    assert cal.R_STAR_HZ == 100.0
    assert cal.R_STAR_R2_HZ == (50.0, 150.0)
    assert all(np.isfinite(gain_for_rate(r)) for r in cal.R_STAR_R2_HZ)


def test_pooled_targets_follow_a4():
    pytest.importorskip("flygym")
    from organism.flybody_tethered import load_walking_clip_front_legs

    clip = load_walking_clip_front_legs()
    t = cal.pooled_targets(clip)
    pooled = np.concatenate([clip["lf"], clip["rf"]])
    assert np.array_equal(t["mean"], pooled.mean(axis=0))
    assert np.array_equal(t["range"], np.percentile(pooled, 95, axis=0) - np.percentile(pooled, 5, axis=0))


def test_replay_path_uses_the_bridge_arithmetic():
    from organism.mn_joint_bridge import BridgeParams, MNJointBridge

    rng = np.random.default_rng(4)
    p = BridgeParams.unfitted("position")
    for leg in ("T1L", "T1R"):
        p.offset[leg] = rng.normal(0.3, 0.2, 7)
        p.scale[leg] = np.abs(rng.normal(0.4, 0.1, 7))
        p.c[leg] = rng.normal(5, 2, 7)
        p.s[leg] = np.abs(rng.normal(8, 2, 7)) + 0.5
        p.lower[leg] = np.full(7, -0.5)
        p.upper[leg] = np.full(7, 1.5)
    bridge = MNJointBridge()
    Rs = rng.uniform(0, 30, (20, 4310))
    u = np.stack([np.stack([bridge.drive(R)["T1L"], bridge.drive(R)["T1R"]]) for R in Rs])
    replay = cal.replay_bridge(u)
    for k, R in enumerate(Rs):
        a, b = bridge.commands(R, p), replay.commands(k, p)
        assert np.array_equal(a["T1L"], b["T1L"]) and np.array_equal(a["T1R"], b["T1R"])


def test_one_replicate_end_to_end_matches_cosim():
    """Network pass + body passes + equivalence check on replicate 960, 0.3 s.
    The real co-simulation's deafferented run must reproduce the body-pass angles exactly."""
    pytest.importorskip("flygym")
    res = cal.calibrate([960], T=0.3, workers=1, verify=(960,), log=lambda *a: None, max_corrections=1)
    assert res["equivalence_with_cosim"]["pass"] is True
    assert res["replicates"] == [960]
    assert 1 <= len(res["iterations"]) <= 2 and res["max_corrections"] == 1
    assert res["corrections_used"] == len(res["iterations"]) - 1
    assert set(res["acceptance"]["criteria"]) == {"T1L:coxa_pitch", "T1L:trfe_pitch", "T1R:coxa_pitch", "T1R:trfe_pitch"}
    from organism.mn_joint_bridge import BridgeParams

    BridgeParams.from_dict(res["final_params"]).validate()


@pytest.mark.skipif(not FROZEN.exists(), reason="calibration not run yet")
def test_frozen_file_is_consistent():
    doc = json.loads(FROZEN.read_text())
    assert doc["content_sha256"] == cal.canonical_hash(doc)
    assert doc["status"] in ("accepted", "failed")
    c = doc["calibration"]
    assert set(c["replicates"]) <= set(range(960, 1024))
    assert c["equivalence_with_cosim"]["pass"] is True
    crit = c["acceptance"]["criteria"]
    assert all(isinstance(v["pass"], bool) for v in crit.values())
    assert doc["status"] == ("accepted" if all(v["pass"] for v in crit.values()) else "failed")
    assert doc["sensory"]["hair_plate_primary"] is None and len(doc["sensory"]["hair_plate_factorial"]) == 4
    assert doc["drive"]["bodyIds"] == [10045, 10056]
    assert doc["sensory"]["R2_r_star_hz"]["upper"] == 150.0 and doc["sensory"]["R2_r_star_hz"]["lower"] == 50.0
    ds = c["drive_stats"]
    assert ds["centring"].startswith("median")
    assert c["max_corrections"] == 8 and c["corrections_used"] == len(c["iterations"]) - 1 <= 8
    assert [doc["bridge_params"]["c"][leg] for leg in ("T1L", "T1R")] == ds["c_median_hz"]
    from organism.mn_joint_bridge import BridgeParams

    BridgeParams.from_dict(doc["bridge_params"]).validate()
    data = ROOT / "data" / "paper2"
    for key, name in (("sensory_table", "stage2B_sensory_table.json"), ("sensory_norm", "stage2B_sensory_norm.json"),
                      ("mn_joint_table", "stage2B_mn_joint_table.json"), ("body_transfer", "stage2B_body_transfer.json")):
        assert doc["data_sha256"][key] == cal._sha256(data / name), key
