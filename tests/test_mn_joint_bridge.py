"""Stage 2B plan step 4: motor neuron -> FlyBody joint bridge (design §3.3-3.4, §8).

- all 129 motor neurons are assigned to a joint channel of their own leg, or
  logged as not actuated (substrate grip only); the two legs share one table;
- the agonist-antagonist scheme computes what §3.4 says, refuses unfitted
  parameters, and has no fitted values in it (they belong to plan step 7);
- geometric sign tests on FlyBody, both kinematic (the stored signs) and
  through the bridge and the servos (activating a module moves the leg the
  anatomical way);
- FlyMimic (mesh-free) moment-arm sign agreement on the pitch axis, with the
  one documented exception (sternal adductor).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from organism.mn_joint_bridge import (
    CHANNELS,
    DOF_TEMPLATES,
    FLYMIMIC_PITCH_EXCEPTIONS,
    LEGS,
    TABLE_PATH,
    BridgeParams,
    MNJointBridge,
    assign_motor_neurons,
    flymimic_agreement,
)

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not TABLE_PATH.exists(), reason="run: python -m organism.mn_joint_bridge")


@pytest.fixture(scope="module")
def table():
    return json.loads(TABLE_PATH.read_text())


@pytest.fixture(scope="module")
def wtable():
    from flybrain.authors_mcns import load_authors_wtable

    return load_authors_wtable()


@pytest.fixture(scope="module")
def body():
    pytest.importorskip("flygym")
    from organism.flybody_tethered import TetheredFlyBody

    return TetheredFlyBody("position")


@pytest.fixture(scope="module")
def clip():
    pytest.importorskip("flygym")
    from organism.flybody_tethered import load_walking_clip_front_legs

    return load_walking_clip_front_legs()


# --------------------------------------------------------------------------- coverage
def test_all_129_motor_neurons_assigned_or_logged(table, wtable):
    mns = table["motor_neurons"]
    assert table["n_motor_neurons"] == len(mns) == 129
    mn_rows = np.flatnonzero(wtable["motor module"].notna().to_numpy())
    assert sorted(r["row"] for r in mns) == mn_rows.tolist()
    for r in mns:
        w = wtable.iloc[r["row"]]
        assert r["bodyId"] == int(w["bodyId"]) and r["motor_module"] == w["motor module"]
        assert r["leg"] == {"L": "T1L", "R": "T1R"}[w["somaSide"]]
        if r["role"] == "logged_not_actuated":
            assert r["motor_module"] == "substrate grip" and r["channel"] is None
        else:
            assert r["role"] in ("agonist", "antagonist")
            body_leg = {"T1L": "lf", "T1R": "rf"}[r["leg"]]
            assert r["dof"] == DOF_TEMPLATES[CHANNELS[r["channel"]]["dof"]].format(leg=body_leg)
    # Every MN of every module except substrate grip is actuated.
    actuated = {r["motor_module"] for r in mns if r["role"] != "logged_not_actuated"}
    assert actuated == set(wtable["motor module"].dropna()) - {"substrate grip"}


def test_table_matches_fresh_assignment_and_counts(table, wtable):
    assert assign_motor_neurons(wtable) == table["motor_neurons"]
    left = pd.Series([r["motor_module"] for r in table["motor_neurons"] if r["leg"] == "T1L"]).value_counts()
    assert left["coxa swing"] == table["counts"]["T1L:coxa_pitch:agonist"] == 7
    assert left["coxa stance"] == table["counts"]["T1L:coxa_pitch:antagonist"] == 6
    assert table["counts"]["T1L:trfe_pitch:agonist"] == 10 and table["counts"]["T1L:trfe_pitch:antagonist"] == 8
    assert table["counts"]["T1L:logged_not_actuated:logged_not_actuated"] == 8


def test_two_legs_share_one_mirror_table(table):
    by = {leg: {(r["type"], r["motor_module"], r["channel"], r["role"]) for r in table["motor_neurons"] if r["leg"] == leg}
          for leg in LEGS}
    assert by["T1L"] == by["T1R"]  # same type -> channel/role rule on both sides
    for ch, spec in table["channels"].items():
        assert spec["agonist_angle_sign"]["T1L"] == spec["agonist_angle_sign"]["T1R"] != 0


# --------------------------------------------------------------------------- scheme
def _params(mode="position"):
    rng = np.random.default_rng(0)
    p = BridgeParams.unfitted(mode)
    for leg in LEGS:
        p.offset[leg] = rng.normal(0, 0.3, 7)
        p.scale[leg] = np.abs(rng.normal(0.3, 0.1, 7))
        p.scale[leg][[0, 1]] = 0.0
        p.c[leg] = rng.normal(2, 1, 7)
        p.s[leg] = np.abs(rng.normal(5, 1, 7)) + 0.5
        p.lower[leg] = np.full(7, -1.0)
        p.upper[leg] = np.full(7, 1.0)
    return p


def test_drive_is_agonist_minus_antagonist(table):
    bridge = MNJointBridge(table)
    R = np.random.default_rng(1).uniform(0, 20, 4310)
    u = bridge.drive(R)
    for leg in LEGS:
        for ch, spec in CHANNELS.items():
            ag = sum(R[r["row"]] for r in table["motor_neurons"] if r["leg"] == leg and r["channel"] == ch and r["role"] == "agonist")
            an = sum(R[r["row"]] for r in table["motor_neurons"] if r["leg"] == leg and r["channel"] == ch and r["role"] == "antagonist")
            assert u[leg][spec["dof"]] == pytest.approx(ag - an, rel=1e-12, abs=1e-12)
        assert u[leg][0] == 0.0 and u[leg][1] == 0.0  # coxa yaw / roll held
    # Logged-not-actuated and non-MN rates change nothing.
    R2 = R.copy()
    for r in table["motor_neurons"]:
        if r["role"] == "logged_not_actuated":
            R2[r["row"]] += 50.0
    mn = {r["row"] for r in table["motor_neurons"]}
    R2[[i for i in range(4310) if i not in mn][:500]] += 50.0
    u2 = bridge.drive(R2)
    for leg in LEGS:
        assert np.array_equal(u[leg], u2[leg])


def test_legs_are_independent(table):
    bridge = MNJointBridge(table)
    R = np.zeros(4310)
    for r in table["motor_neurons"]:
        if r["leg"] == "T1R" and r["role"] == "agonist":
            R[r["row"]] = 10.0
    u = bridge.drive(R)
    assert np.all(u["T1L"] == 0.0) and np.any(u["T1R"] != 0.0)


@pytest.mark.parametrize("mode", ["position", "torque"])
def test_commands_follow_section_3_4(table, mode):
    bridge = MNJointBridge(table)
    p = _params(mode)
    R = np.random.default_rng(2).uniform(0, 30, 4310)
    u = bridge.drive(R)
    cmd = bridge.commands(R, p)
    for leg in LEGS:
        expect = p.offset[leg] + bridge.sign[leg] * p.scale[leg] * (u[leg] - p.c[leg]) / p.s[leg]
        if mode == "position":
            expect = np.clip(expect, -1.0, 1.0)
        assert np.allclose(cmd[leg], expect, rtol=0, atol=1e-12)
        held = p.offset[leg][[0, 1]] if mode == "torque" else np.clip(p.offset[leg][[0, 1]], -1.0, 1.0)
        assert np.array_equal(cmd[leg][[0, 1]], held)  # coxa yaw / roll stay at their offset


def test_unfitted_or_invalid_params_are_refused(table):
    bridge = MNJointBridge(table)
    R = np.zeros(4310)
    with pytest.raises(ValueError, match="not fitted"):
        bridge.commands(R, BridgeParams.unfitted())
    p = _params()
    p.s["T1L"][2] = 0.0
    with pytest.raises(ValueError):
        bridge.commands(R, p)
    p = _params()
    p.scale["T1R"][4] = -0.1
    with pytest.raises(ValueError):
        bridge.commands(R, p)
    p = BridgeParams.from_dict(_params().to_dict())
    assert np.array_equal(p.offset["T1L"], _params().offset["T1L"])


def test_module_neither_runs_nor_imports_the_network():
    """Offsets and scales are fitted in plan step 7; this module must not touch the rate model."""
    src = (ROOT / "organism" / "mn_joint_bridge.py").read_text()
    names = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
    banned = {"flybrain.pugliese_rate", "flybrain.rate_stepper", "experiment.leg_coordination"}
    assert not names & banned


# --------------------------------------------------------------------------- geometry
def test_stored_signs_match_flybody_geometry(table, body, clip):
    from organism.mn_joint_bridge import geometric_signs, operating_posture

    signs = geometric_signs(body, operating_posture(clip))
    for ch, spec in table["channels"].items():
        for leg in LEGS:
            assert signs[leg][ch] == spec["agonist_angle_sign"][leg], (ch, leg)
            # Regression guard, not a pre-registered bar: measured at build time, coxa
            # pitch's sign holds in 87% (T1L) / 84% (T1R) of clip frames, the others in >= 98%.
            assert spec["sign_holds_fraction_of_clip_frames"][leg] >= 0.8, (ch, leg)


def _segment(body):
    from organism.mn_joint_bridge import _segment_positions

    return _segment_positions(body)


@pytest.mark.parametrize("leg", LEGS)
def test_activating_a_module_moves_the_leg_the_anatomical_way(table, body, clip, leg):
    """End to end through the bridge and the position servos. Parameters here are
    test values (offset = walking mean, scale 0.2 rad, c = 0, s = 10 Hz), not fits."""
    from organism.mn_joint_bridge import _criterion

    bridge = MNJointBridge(table)
    p = BridgeParams.unfitted("position")
    desc = body.describe()["dofs"]
    for lg in LEGS:
        bl = {"T1L": "lf", "T1R": "rf"}[lg]
        p.offset[lg] = clip[bl].mean(axis=0)
        p.scale[lg] = np.full(7, 0.2)
        p.c[lg] = np.zeros(7)
        p.s[lg] = np.full(7, 10.0)
        p.lower[lg] = np.array([desc[n]["range_rad"][0] for n in body.dof_names[bl]])
        p.upper[lg] = np.array([desc[n]["range_rad"][1] for n in body.dof_names[bl]])
    bl = {"T1L": "lf", "T1R": "rf"}[leg]

    def settle(R):
        cmd = bridge.commands(R, p)
        body.reset(dict(zip(body.dof_names["lf"], cmd["T1L"])), dict(zip(body.dof_names["rf"], cmd["T1R"])),
                   warmup_s=0.08,
                   left_angles=dict(zip(body.dof_names["lf"], p.offset["T1L"])),
                   right_angles=dict(zip(body.dof_names["rf"], p.offset["T1R"])))
        return _segment(body)

    rest = settle(np.zeros(4310))
    for ch, spec in CHANNELS.items():
        for role, sign in (("agonist", 1), ("antagonist", -1)):
            rows = [r["row"] for r in table["motor_neurons"] if r["leg"] == leg and r["channel"] == ch and r["role"] == role]
            if not rows:
                continue  # femur reductor has no antagonist
            R = np.zeros(4310)
            R[rows] = 10.0 / len(rows)  # module sum = 10 Hz -> 0.2 rad command step
            moved = settle(R)
            change = _criterion(moved, bl, spec["criterion"]) - _criterion(rest, bl, spec["criterion"])
            assert np.sign(change) == sign, (leg, ch, role, change)


# --------------------------------------------------------------------------- FlyMimic
def test_flymimic_pitch_signs_agree_except_the_sternal_adductor(table):
    pytest.importorskip("flygym")
    fresh = flymimic_agreement()
    assert fresh == table["flymimic_check"]
    assert fresh["pitch_only_disagreements"] == list(FLYMIMIC_PITCH_EXCEPTIONS)
    must_agree = {
        "LFC_tergopleural_promotor_a", "LFC_tergopleural_promotor_b", "LFC_pleural_promotor",
        "LFC_pleural_remotor_and_abductor", "LFC_sternal_posterior_rotator",
        "LFF_trochanter_flexor_a", "LFF_trochanter_flexor_b", "LFF_accesory_trochanter_flexor",
        "LFF_sterno-tergo-trochanter_extensor_a", "LFF_sterno-tergo-trochanter_extensor_b",
        "LFF_trochanter_extensor", "LFTibia_flex_93434", "LFTibia_extensor_93932",
    }
    for m in must_agree:
        assert fresh["muscles"][m]["pitch_only_agrees"], m
