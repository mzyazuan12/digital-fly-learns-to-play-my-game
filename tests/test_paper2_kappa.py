"""Stage 2B-kappa (amendment A8): kappa calibration and runner.

- kappa scales only the 31 included afferents' outgoing columns of the post x pre W.
- kappa = 1 reproduces the stage 2B runner's traces bit for bit (one replicate, closed loop, one cell).
- Refusals: kappa-frozen hash mismatch, uncommitted kappa file, stage 2B frozen-bridge
  hash mismatch, existing output, calibration replicates, off-grid kappa, confirm without kappa*.
- Calibration: replicates 960-1023 only, grid only, refuses to overwrite; criteria logic;
  stimulus construction.
- No coordination read-out code in either module (AST).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from experiment import paper2_kappa_calibrate as kc
from experiment import paper2_kappa_run as kr
from experiment import paper2_stage2B_run as sr

ROOT = Path(__file__).resolve().parents[1]


def _kappa_file(tmp_path, kappa_star=1.0):
    p = tmp_path / "kappa.json"
    p.write_text(json.dumps({"grid": list(kc.GRID), "kappa_star": kappa_star, "status": "test"}))
    return p, sr.sha256(p)


# --------------------------------------------------------------------------- kappa scaling
@pytest.fixture(scope="module")
def net():
    from experiment.paper2_cosim import Network

    return Network.load()


def test_kappa_touches_only_afferent_columns(net):
    rows = kc.afferent_rows()
    base = net.W.tocsc()
    for kappa in (0.3, 0.001):
        Wk = kc.kappa_network(net, kappa, rows).W.tocsc()
        assert Wk.shape == base.shape and np.array_equal(Wk.indptr, base.indptr) and np.array_equal(Wk.indices, base.indices)
        col = np.repeat(np.arange(base.shape[1]), np.diff(base.indptr))
        aff = np.isin(col, rows)
        assert aff.sum() > 0
        assert np.array_equal(Wk.data[~aff], base.data[~aff])
        assert np.array_equal(Wk.data[aff], base.data[aff] * kappa)
    same = kc.kappa_network(net, 1.0, rows).W
    assert np.array_equal(same.data, net.W.data) and same.data is not net.W.data  # a copy
    off = kc.kappa_network(net, 0.5, rows, dn_off=True)
    assert np.all(off.dn_input == 0) and np.any(net.dn_input != 0)  # loaded network unchanged


def test_afferent_rows_are_the_31_included():
    rows = kc.afferent_rows()
    st = json.loads((ROOT / "data" / "paper2" / "stage2B_sensory_table.json").read_text())
    assert rows.tolist() == sorted(a["row"] for a in st["afferents"] if a["included_2B"]) and rows.size == 31


# --------------------------------------------------------------------------- kappa = 1 bit identity
def test_kappa_one_reproduces_stage2b_runner(tmp_path):
    pytest.importorskip("flygym")
    from organism.proprio_transduction import FACTORIAL

    kfile, ksha = _kappa_file(tmp_path)
    a = sr.run_arm("primary", replicates=[0], out=tmp_path / "s2b", workers=0, conditions=["closed_loop"],
                   cells=[FACTORIAL[0]])
    b = kr.run_kappa_arm("scan", kappa_sha256=ksha, kappa_path=kfile, replicates=[0], kappas=[1.0],
                         out=tmp_path / "k", workers=0, conditions=["closed_loop"], cells=[FACTORIAL[0]],
                         kappa_dirty_lines=[])
    assert a["n_ok"] == b["n_ok"] == 1
    fa = np.load(tmp_path / "s2b" / "runs" / "rep000_closed_loop_protraction-flexion.npz")
    fb = np.load(tmp_path / "k" / "runs" / "kappa_1" / "rep000_closed_loop_protraction-flexion.npz")
    for key in ("rates_hz", "rows", "angle", "velocity", "actuator_force", "command", "afferent_current"):
        assert np.array_equal(fa[key], fb[key]), key
    meta = json.loads(str(fb["meta"]))
    assert meta["kappa"] == 1.0 and meta["kappa_frozen_sha256"] == ksha


# --------------------------------------------------------------------------- runner refusals
def test_runner_refusals(tmp_path):
    kfile, ksha = _kappa_file(tmp_path)
    with pytest.raises(RuntimeError, match="kappa-frozen SHA-256"):
        kr.run_kappa_arm("scan", kappa_sha256="0" * 64, kappa_path=kfile, replicates=[0], out=tmp_path / "a",
                         workers=0, kappa_dirty_lines=[])
    with pytest.raises(RuntimeError, match="not committed"):
        kr.kappa_preflight(kfile, ksha, dirty_lines=["?? data/paper2/stage2Bk_kappa_frozen.json"])
    bad = tmp_path / "frozen.json"
    doc = json.loads(sr.FROZEN_PATH.read_text())
    doc["status"] = "x"
    bad.write_text(json.dumps(doc))
    with pytest.raises(RuntimeError, match="SHA-256"):
        kr.run_kappa_arm("scan", kappa_sha256=ksha, kappa_path=kfile, replicates=[0], out=tmp_path / "b",
                         workers=0, frozen_path=bad, kappa_dirty_lines=[])
    (tmp_path / "c").mkdir()
    with pytest.raises(FileExistsError):
        kr.run_kappa_arm("scan", kappa_sha256=ksha, kappa_path=kfile, replicates=[0], out=tmp_path / "c",
                         workers=0, kappa_dirty_lines=[])
    with pytest.raises(ValueError):
        kr.run_kappa_arm("scan", kappa_sha256=ksha, kappa_path=kfile, replicates=[960], out=tmp_path / "d",
                         workers=0, kappa_dirty_lines=[])
    with pytest.raises(ValueError, match="grid"):
        kr.run_kappa_arm("scan", kappa_sha256=ksha, kappa_path=kfile, replicates=[0], kappas=[0.7],
                         out=tmp_path / "e", workers=0, kappa_dirty_lines=[])
    (tmp_path / "none").mkdir()
    nofile, nosha = _kappa_file(tmp_path / "none", kappa_star=None)
    with pytest.raises(RuntimeError, match="no kappa"):
        kr.run_kappa_arm("confirm", kappa_sha256=nosha, kappa_path=nofile, out=tmp_path / "f", workers=0,
                         kappa_dirty_lines=[])
    with pytest.raises(ValueError, match="only"):
        kr.run_kappa_arm("scan", kappa_sha256=ksha, kappa_path=kfile, replicates=[0], out=tmp_path / "g",
                         workers=0, conditions=["open_loop"], kappa_dirty_lines=[])
    for p in "abdefg":
        assert not (tmp_path / p).exists()


def test_arm_definitions():
    assert kr.ARMS["confirm"]["replicates"] == tuple(range(64))
    assert set(kr.ARMS["confirm"]["conditions"]) == {"open_loop", "deafferented", "closed_loop", "scrambled", "mismatched"}
    assert kr.ARMS["scan"]["replicates"] == tuple(range(16))
    assert kr.ARMS["scan"]["conditions"] == ("closed_loop", "scrambled", "mismatched")
    doc = {"grid": list(kc.GRID), "kappa_star": 0.05}
    assert kr.kappas_for("confirm", doc) == (0.05,)
    assert kr.kappas_for("scan", doc) == kc.GRID
    with pytest.raises(ValueError):
        kr.kappas_for("confirm", doc, [0.1])


# --------------------------------------------------------------------------- calibration
def test_calibration_refusals(tmp_path):
    out = tmp_path / "k.json"
    out.write_text("{}")
    with pytest.raises(FileExistsError):
        kc.main(["--out", str(out)])
    assert out.read_text() == "{}"
    for bad in ([0], [959], [1024]):
        with pytest.raises(ValueError):
            kc.check_replicates(bad)
    with pytest.raises(ValueError):
        kc.check_kappas([0.7])
    assert kc.GRID == (1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001)


def test_terciles_and_criteria():
    t = kc.motor_neuron_terciles()
    assert [len(t["T1L"][k]) for k in ("smallest", "middle", "largest")] == [23, 22, 22]
    assert [len(t["T1R"][k]) for k in ("smallest", "middle", "largest")] == [21, 21, 20]
    from flybrain.authors_mcns import load_authors_wtable

    w = load_authors_wtable()
    for leg in ("T1L", "T1R"):
        s = w["size"].to_numpy(float)
        assert s[t[leg]["smallest"]].max() <= s[t[leg]["middle"]].min()
        assert s[t[leg]["middle"]].max() <= s[t[leg]["largest"]].min()
    allrows = t["T1L"]["rows"] + t["T1R"]["rows"]
    quiet = {r: 0.0 for r in allrows}
    r0 = kc.run_criteria(quiet, t)
    assert r0["C1"] and r0["C2"] and not r0["smallest_active"]
    one_large = {**quiet, **{t["T1R"]["largest"][0]: 5.0}}
    assert not kc.run_criteria(one_large, t)["C1"]
    many = {**quiet, **{r: 2.0 for r in t["T1L"]["smallest"]}}  # 23 of 67 > 67/3
    rc = kc.run_criteria(many, t)
    assert rc["C1"] and not rc["C2"] and rc["smallest_active"]
    at_limit = {**quiet, **{r: 2.0 for r in t["T1L"]["smallest"][:22]}}  # 22 <= 22.33
    assert kc.run_criteria(at_limit, t)["C2"]
    threshold = {**quiet, **{t["T1L"]["smallest"][0]: 1.0}}  # must exceed 1 Hz
    assert not kc.run_criteria(threshold, t)["smallest_active"]


def test_kappa_star_selection():
    def rows_for(k, cell, n_pass, n=20, small=0):
        out = []
        for i in range(n):
            ok = i < n_pass
            out.append({"kappa": k, "cell": cell, "status": "ok", "C1": ok, "C2": ok, "both": ok,
                        "smallest_active": i < small, "max_mn_rate_hz": 1.0,
                        "T1L": {"n_active": 0, "n_active_largest": 0}, "T1R": {"n_active": 0, "n_active_largest": 0}})
        return out
    cells = ["a", "b"]
    rows = rows_for(0.1, "a", 20, small=12) + rows_for(0.1, "b", 18) + rows_for(0.05, "a", 20, small=12) + rows_for(0.05, "b", 19, small=3)
    s = kc.summarise(rows, (0.1, 0.05), cells)
    assert s["passing_kappas"] == [0.05] and s["kappa_star"] == 0.05  # 18/20 = 0.90 fails, 19/20 = 0.95 passes
    assert s["effective_by_cell"] == {"a": True, "b": False} and s["label"] == "mixed"
    none = kc.summarise(rows_for(0.1, "a", 10) + rows_for(0.1, "b", 10), (0.1,), cells)
    assert none["kappa_star"] is None and none["label"] == "no grid kappa meets C1 and C2"
    err = rows_for(0.1, "a", 20) + rows_for(0.1, "b", 20)
    err[0]["status"] = "error"
    assert not kc.summarise(err, (0.1,), cells)["table"]["0.1"]["a"]["pass"]  # a failed run blocks a pass


def test_stimulus_is_the_clip_tiled_with_velocities_from_the_1s_clip():
    pytest.importorskip("flygym")
    from organism.flybody_tethered import load_walking_clip_front_legs
    from organism.proprio_transduction import FACTORIAL, ProprioTransducer

    clip = load_walking_clip_front_legs()
    st = kc.stimulus_states(T=2.0)
    n1 = clip["lf"].shape[0]
    assert st["lf"]["angle"].shape == (20000, 7)
    assert np.array_equal(st["lf"]["angle"][:n1], clip["lf"]) and np.array_equal(st["lf"]["angle"][n1:], clip["lf"])
    vel = np.gradient(clip["rf"], float(clip["timestep"]), axis=0)
    assert np.array_equal(st["rf"]["velocity"][n1:], vel)
    assert np.all(st["rf"]["actuator_force"][:, [0, 1, 2, 3, 5, 6]] == 0)
    rows, I = kc.afferent_current_sequence({k: {kk: v[:50] for kk, v in d.items()} for k, d in st.items()}, FACTORIAL[0])
    tr = ProprioTransducer(FACTORIAL[0], r_star_hz=100.0)
    k = 37
    ref = tr.currents({bl: {key: st[bl][key][k] for key in ("angle", "velocity", "actuator_force")} for bl in ("lf", "rf")})
    assert np.array_equal(I[k], ref[rows])


def test_small_calibration_end_to_end():
    """Two replicates, two grid kappas, one cell, 0.5 s: structure and criteria fields."""
    pytest.importorskip("flygym")
    from organism.proprio_transduction import FACTORIAL

    res = kc.calibrate([960, 961], kappas=[1.0, 0.001], cells=[FACTORIAL[0]], T=0.5, workers=0, log=lambda *a: None)
    assert res["n_runs"] == 4 and res["n_failed"] == 0
    assert set(res["table"]) == {"1", "0.001"}
    assert all(r["C1"] in (True, False) and "T1L" in r for r in res["runs"])


# --------------------------------------------------------------------------- no read-outs
@pytest.mark.parametrize("module", ["experiment/paper2_kappa_calibrate.py", "experiment/paper2_kappa_run.py"])
def test_no_coordination_readout(module):
    tree = ast.parse((ROOT / module).read_text())
    banned_modules = {"experiment.leg_coordination", "scipy.signal", "experiment.pugliese_rate_mcns_dng100",
                      "experiment.paper2_stage2B_readout"}
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
