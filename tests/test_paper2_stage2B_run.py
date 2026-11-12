"""Stage 2B plan step 9: the runner (design §6.3, §8 row 9; amendments A1-A7).

Refusals: wrong frozen-bridge hash, dirty bridge files, an existing output
directory, calibration replicates, and an arm whose actuation the frozen bridge
was not calibrated for (R1 torque). Arm settings and job plan. No read-out code
(AST check). A 1-replicate x 1-condition smoke run writes the expected arrays.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from experiment import paper2_stage2B_run as sr

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not sr.FROZEN_PATH.exists(), reason="frozen bridge missing")


def test_expected_hash_is_amendment_a7():
    assert sr.EXPECTED_FROZEN_SHA256 == "91bc54c33dd1c1e60199ece5f6a0c17f947393dc9022963e4d221b6d456fac25"


def test_refuses_a_different_frozen_file(tmp_path):
    bad = tmp_path / "frozen.json"
    doc = json.loads(sr.FROZEN_PATH.read_text())
    doc["status"] = "accepted"  # any edit changes the hash
    bad.write_text(json.dumps(doc, indent=1) + "\n")
    with pytest.raises(RuntimeError, match="SHA-256"):
        sr.preflight(bad)
    with pytest.raises(RuntimeError, match="SHA-256"):
        sr.run_arm("primary", replicates=[0], out=tmp_path / "out", frozen_path=bad, workers=0)
    assert not (tmp_path / "out").exists()


def test_refuses_dirty_bridge_files():
    with pytest.raises(RuntimeError, match="not clean"):
        sr.preflight(dirty_lines=[" M organism/mn_joint_bridge.py"])


def test_refuses_to_overwrite(tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    with pytest.raises(FileExistsError):
        sr.run_arm("primary", replicates=[0], out=out, workers=0)
    assert list(out.iterdir()) == []


def test_refuses_calibration_and_out_of_block_replicates():
    assert sr.check_replicates(range(0, 64)) == tuple(range(64))
    assert sr.PRIMARY_REPLICATES == tuple(range(64))
    for bad in ([960], [1023], [5, 1000], [1024], [-1]):
        with pytest.raises(ValueError):
            sr.check_replicates(bad)


def test_arm_settings_and_r1_refusal():
    frozen = json.loads(sr.FROZEN_PATH.read_text())
    assert sr.arm_settings("primary", frozen) == {"arm": "primary", "actuation": "position", "r_star_hz": 100.0,
                                                  "base": ("open_loop", "deafferented")}
    assert sr.arm_settings("R2_r50", frozen)["r_star_hz"] == 50.0
    assert sr.arm_settings("R2_r150", frozen)["r_star_hz"] == 150.0
    assert sr.arm_settings("R2_r150", frozen)["base"] == ()
    with pytest.raises(RuntimeError, match="torque"):
        sr.arm_settings("R1_torque", frozen)


def test_job_plan_puts_closed_loop_before_scrambled():
    from experiment.paper2_cosim import CONDITIONS
    from organism.proprio_transduction import FACTORIAL

    frozen = json.loads(sr.FROZEN_PATH.read_text())
    st = sr.arm_settings("primary", frozen)
    jobs = sr.plan_jobs(Path("/nonexistent"), st, frozen, range(64), CONDITIONS, FACTORIAL)
    assert len(jobs) == 64 * (1 + 4)
    n_runs = sum(len(j[5]) for j in jobs)
    assert n_runs == 64 * 14
    for j in jobs:
        if j[4] is not None:
            assert j[5] == ("closed_loop", "scrambled", "mismatched")
    with pytest.raises(ValueError):
        sr.plan_jobs(Path("/nonexistent"), st, frozen, [0], ("scrambled",), FACTORIAL)
    r2 = sr.arm_settings("R2_r50", frozen)
    assert sum(len(j[5]) for j in sr.plan_jobs(Path("/x"), r2, frozen, range(64), CONDITIONS, FACTORIAL)) == 64 * 12


def test_recorded_rows():
    rec = sr.recorded_rows()
    kinds = [lab.split("|")[0] for lab in rec["labels"]]
    assert kinds.count("MN") == 129 and kinds.count("CPG") == 10 and kinds.count("DN") == 2 and kinds.count("AFF") == 31
    assert sorted(int(b) for b, k in zip(rec["bodyIds"], kinds) if k == "DN") == [10045, 10056]


def test_runner_computes_no_coordination_readout():
    tree = ast.parse((ROOT / "experiment" / "paper2_stage2B_run.py").read_text())
    banned_modules = {"experiment.leg_coordination", "scipy.signal", "experiment.pugliese_rate_mcns_dng100"}
    banned_names = {"lag_cycles", "plv", "period_autocorr", "circ_stats", "replicate_readouts", "null_pairs",
                    "hilbert", "correlate", "csd", "coherence", "angle", "unwrap", "compute_oscillation_score"}
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


def test_smoke_one_replicate_one_condition(tmp_path):
    """Replicate 0, closed loop, one hair-plate cell, full 2 s, in process. This needs a
    clean tree for the bridge files (the real preflight runs)."""
    pytest.importorskip("flygym")
    from organism.proprio_transduction import FACTORIAL

    out = tmp_path / "smoke"
    rep = sr.run_arm("primary", replicates=[0], out=out, workers=0, conditions=["closed_loop"], cells=[FACTORIAL[0]])
    assert rep["n_runs"] == 1 and rep["n_ok"] == 1 and rep["n_failed"] == 0 and rep["n_rates_invalid"] == 0
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "finished" and report["frozen_sha256"] == sr.EXPECTED_FROZEN_SHA256
    f = out / "runs" / "rep000_closed_loop_protraction-flexion.npz"
    d = np.load(f)
    meta = json.loads(str(d["meta"]))
    assert meta["status"] == "ok" and meta["rates_valid"] and meta["r_star_hz"] == 100.0
    assert d["rates_hz"].shape == (172, 2001) and d["rates_hz"].dtype == np.float32
    assert d["angle"].shape == d["velocity"].shape == d["actuator_force"].shape == d["command"].shape == (2001, 2, 7)
    assert d["afferent_current"].shape == (2001, 31)
    assert float(d["dt_s"]) == pytest.approx(1e-3)
    assert np.max(d["afferent_current"]) > 0
    readout_like = {"lag", "phase", "plv", "coupling", "readout", "stance_vs_swing_cycles"}
    assert not any(any(w in k for w in readout_like) for k in d.files)
