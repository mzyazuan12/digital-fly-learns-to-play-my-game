"""Stage 2B plan step 3: the tethered FlyBody (design §3.2, §6.1, §8; amendment A1.2).

Validation tests from the design's step-3 row: 7 actuated DOFs per front leg
with the §3.2 names; held legs stay within 0.01 rad over 2 s; two runs are
bit-identical; replay of the bundled NeuroMechFly walking clip on the front
legs tracks its targets with RMS < 0.05 rad. Plus: the clip conversion
reproduces FlyGym's own FlyBody step asset, and the saved transfer function
(data/paper2/stage2B_body_transfer.json) is reproduced by a live run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("flygym")

from organism.flybody_tethered import (  # noqa: E402
    FRONT_LEGS,
    TetheredFlyBody,
    expected_front_dof_names,
    load_walking_clip_front_legs,
)

ROOT = Path(__file__).resolve().parents[1]
TRANSFER_JSON = ROOT / "data" / "paper2" / "stage2B_body_transfer.json"


@pytest.fixture(scope="module")
def pos_body():
    return TetheredFlyBody("position")


@pytest.fixture(scope="module")
def tq_body():
    return TetheredFlyBody("torque")


@pytest.fixture(scope="module")
def clip():
    return load_walking_clip_front_legs()


def _bodies(request, mode):
    return request.getfixturevalue("pos_body" if mode == "position" else "tq_body")


# --------------------------------------------------------------------------- structure
@pytest.mark.parametrize("mode", ["position", "torque"])
def test_front_leg_dofs_match_design(request, mode):
    body = _bodies(request, mode)
    assert body.timestep == 1e-4
    for leg in FRONT_LEGS:
        assert body.dof_names[leg] == expected_front_dof_names(leg)
        assert len(body.dof_names[leg]) == 7
    assert len(body.held_dof_names) == 28
    assert not any("lf_" in n or "rf_" in n for n in body.held_dof_names)
    desc = body.describe()
    assert desc["gravity_mm_s2"] == [0.0, 0.0, -9810.0]
    for leg in FRONT_LEGS:
        gains = [desc["dofs"][n]["actuator_gain"] for n in body.dof_names[leg]]
        if mode == "position":  # FlyBody actuators.yaml: coxa-femur 80, tarsus-tibia 40
            assert gains == [80.0] * 5 + [40.0] * 2
        else:
            assert gains == [1.0] * 7
            assert all(desc["dofs"][n]["actuator_bias"] == [0.0, 0.0, 0.0] for n in body.dof_names[leg])
        assert all(desc["dofs"][n]["limited"] for n in body.dof_names[leg])


def test_state_api(pos_body):
    body = pos_body
    body.reset()
    lf = body.dof_names["lf"]
    body.set_front_leg_targets({lf[2]: 0.5}, None)
    body.step(50)
    state = body.get_state()
    assert set(state) == {"t", "dof_names", "lf", "rf"}
    assert state["t"] == pytest.approx(50 * 1e-4)
    arrays = body.front_leg_arrays()
    for leg in FRONT_LEGS:
        assert set(state[leg]) == {"angle", "velocity", "actuator_force"}
        for key in ("angle", "velocity", "actuator_force"):
            assert list(state[leg][key]) == body.dof_names[leg]
            assert np.array_equal(np.array(list(state[leg][key].values())), arrays[leg][key])
    # Coxa pitch moved toward 0.5; other commands kept their neutral value (0).
    assert 0.0 < state["lf"]["angle"][lf[2]] <= 0.5
    with pytest.raises(KeyError):
        body.set_front_leg_targets({"c_thorax-rf_coxa-pitch": 0.1}, None)
    with pytest.raises(KeyError):
        body.set_front_leg_targets(None, {"not-a-dof": 0.0})
    with pytest.raises(ValueError):
        body.set_front_leg_targets_array(np.zeros(6), np.zeros(7))


# --------------------------------------------------------------------------- held legs
@pytest.mark.parametrize("mode", ["position", "torque"])
def test_held_legs_within_0p01_rad_over_2s(request, mode, clip):
    """The four non-front legs stay within 0.01 rad of FlyBody's neutral pose while
    the front legs move for 2 s (position: the walking clip twice; torque: 11.5 Hz
    torques of amplitude 10 on all 14 front DOFs, about +-0.14 rad)."""
    body = _bodies(request, mode)
    body.reset()
    n = 20000
    worst = 0.0
    for k in range(n):
        if mode == "position":
            body.set_front_leg_targets_array(clip["lf"][k % 10000], clip["rf"][k % 10000])
        else:
            tau = 10.0 * np.sin(2 * np.pi * 11.5 * k * 1e-4) * np.ones(7)
            body.set_front_leg_targets_array(tau, tau)
        body.step()
        worst = max(worst, float(np.max(np.abs(body.held_leg_angles() - body.held_leg_targets()))))
    assert worst < 0.01, f"held legs deviated by {worst:.4f} rad"


# --------------------------------------------------------------------------- determinism
def _record(body, mode, clip, n=3000):
    out = []
    for k in range(n):
        if mode == "position":
            body.set_front_leg_targets_array(clip["lf"][k], clip["rf"][k])
        else:
            body.set_front_leg_targets_array(
                5.0 * np.sin(2 * np.pi * 11.5 * k * 1e-4 + np.arange(7)),
                5.0 * np.cos(2 * np.pi * 7.0 * k * 1e-4 + np.arange(7)),
            )
        body.step()
        s = body.front_leg_arrays()
        out.append(np.concatenate([s[leg][key] for leg in FRONT_LEGS for key in ("angle", "velocity", "actuator_force")]
                                  + [body.held_leg_angles()]))
    return np.array(out)


@pytest.mark.parametrize("mode", ["position", "torque"])
def test_two_runs_bit_identical(request, mode, clip):
    """Same instance after reset, and a freshly built instance, give bit-identical runs."""
    body = _bodies(request, mode)
    start = dict(zip(body.dof_names["lf"], clip["lf"][0])) if mode == "position" else None
    body.reset(start, None)
    a = _record(body, mode, clip)
    body.reset(start, None)
    b = _record(body, mode, clip)
    fresh = TetheredFlyBody(mode)
    fresh.reset(start, None)
    c = _record(fresh, mode, clip)
    assert np.array_equal(a, b)
    assert np.array_equal(a, c)
    assert np.ptp(a[:, :7], axis=0).max() > 0.01  # the front leg actually moved


# --------------------------------------------------------------------------- recorded walking
def test_clip_conversion_reproduces_flygym_flybody_asset(clip):
    """The clip conversion is FlyGym's own FlyBody pipeline: resampling our RF trace with
    FlyGym's step-extraction resampler (copied below because importing that module
    needs tqdm) reproduces its shipped ``single_steps_flybody.npz`` front-leg cycle,
    picked from the RF leg at samples 7736-9363, to the asset's 1e-6 rounding."""
    from importlib.resources import files

    meta = json.loads((files("flygym_demo.complex_terrain") / "assets/single_steps_flybody.meta.json").read_text())
    asset = np.load(files("flygym_demo.complex_terrain") / "assets/single_steps_flybody.npz")["joint_angles"][0]
    pick = meta["picks"]["F"]
    assert pick["side"] == "r"
    seg = clip["rf"][pick["start"] : pick["end"]]
    # Asset DOF order (FlyGym _DOFS_PER_LEG): coxa pitch, roll, yaw; trochanterfemur pitch, roll; tibia; tarsus1.
    seg = seg[:, [2, 1, 0, 4, 3, 5, 6]]
    L, nd = seg.shape
    cut = max(2, int(np.floor(L * 0.95)))
    sp = np.empty(cut + 1)
    sp[:cut] = np.arange(cut) / L * 2 * np.pi
    sp[cut] = 2 * np.pi
    sv = np.vstack([seg[:cut], seg[:1]])
    grid = np.linspace(0, 2 * np.pi, asset.shape[1], endpoint=False)
    ours = np.stack([np.interp(grid, sp, sv[:, d]) for d in range(nd)])
    assert np.max(np.abs(ours - asset)) < 1e-5


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Protocol amendment A3: FlyBody's own position servos lag ~10 ms, so the "
        "walking clip is tracked at 0.07-0.25 rad RMS, not < 0.05. Kept visible; "
        "the measured transfer function (data/paper2/stage2B_body_transfer.json) "
        "replaces this gate."
    ),
)
def test_replay_tracks_clip_rms_below_0p05_rad(pos_body, clip):
    """Replay of the bundled walking clip on both front legs (position servos) must track
    the commanded angles with RMS < 0.05 rad on every DOF over the 1 s clip.

    Conversion: FlyGym's own FlyBody replay conversion (``MotionSnippet`` with
    ``angles_global2anatomical=False``, Savitzky-Golay 0.05 s, cubic interpolation to
    0.1 ms, DOFs matched by name, no sign flips); the previous test shows it reproduces
    FlyGym's shipped FlyBody step asset. Remaining uncertainty: the clip does not record
    the axis order its inverse kinematics used. FlyBody's native chain (yaw, roll, pitch)
    is assumed. That affects the meaning of the 3-DOF coxa angles, not how well a servo
    tracks them. The 0.05 rad bar is the design's and is not loosened here; the message
    reports RMS against the range-clipped target and after removing the best pure delay,
    to show where any shortfall comes from.
    """
    body = pos_body
    n = clip["lf"].shape[0]
    body.reset(
        dict(zip(body.dof_names["lf"], clip["lf"][0])),
        dict(zip(body.dof_names["rf"], clip["rf"][0])),
        left_angles=dict(zip(body.dof_names["lf"], clip["lf"][0])),
        right_angles=dict(zip(body.dof_names["rf"], clip["rf"][0])),
    )
    q = {leg: np.empty((n, 7)) for leg in FRONT_LEGS}
    for k in range(n):
        body.set_front_leg_targets_array(clip["lf"][k], clip["rf"][k])
        body.step()
        s = body.front_leg_arrays()
        for leg in FRONT_LEGS:
            q[leg][k] = s[leg]["angle"]
    desc = body.describe()["dofs"]
    lines, worst = [], 0.0
    for leg in FRONT_LEGS:
        target = clip[leg]
        lo = np.array([desc[nm]["range_rad"][0] for nm in body.dof_names[leg]])
        hi = np.array([desc[nm]["range_rad"][1] for nm in body.dof_names[leg]])
        rms = np.sqrt(((q[leg] - target) ** 2).mean(axis=0))
        rms_clipped = np.sqrt(((q[leg] - np.clip(target, lo, hi)) ** 2).mean(axis=0))
        best = []
        for d in range(7):
            errs = [np.sqrt(((q[leg][s:, d] - target[: n - s, d]) ** 2).mean()) for s in range(0, 300, 5)]
            best.append((min(errs), 0.5 * int(np.argmin(errs))))  # (rms, delay ms)
        worst = max(worst, float(rms.max()))
        for d, nm in enumerate(body.dof_names[leg]):
            lines.append(
                f"{nm}: RMS {rms[d]:.3f}, vs range-clipped target {rms_clipped[d]:.3f}, "
                f"after best delay ({best[d][1]:.1f} ms) {best[d][0]:.3f}"
            )
    assert worst < 0.05, "replay RMS exceeds 0.05 rad:\n" + "\n".join(lines)


# --------------------------------------------------------------------------- saved transfer function
def test_saved_transfer_function_is_reproduced(pos_body):
    """data/paper2/stage2B_body_transfer.json covers all 14 front DOFs in both actuation
    models, and a live 11.5 Hz sine on LF coxa pitch reproduces its saved value exactly
    (the simulation is deterministic)."""
    if not TRANSFER_JSON.exists():
        pytest.skip("run: .venv/bin/python -m experiment.paper2_body_transfer")
    from experiment import paper2_body_transfer as tf

    saved = json.loads(TRANSFER_JSON.read_text())
    names = expected_front_dof_names("lf") + expected_front_dof_names("rf")
    for mode in ("position", "torque"):
        assert sorted(saved["results"][mode]) == sorted(names)
        for nm in names:
            r = saved["results"][mode][nm]
            assert len(r["gain"]) == len(r["freq_hz"]) == len(r["phase_lag_deg"])
            assert all(np.isfinite(r["gain"])) and all(np.isfinite(r["phase_lag_deg"]))
    body = pos_body
    clip = load_walking_clip_front_legs()
    op = tf.operating_point(body, clip)
    h = body.timestep
    n = int(round(tf.SINE["duration_s"] / h))
    ts = np.arange(n) * h
    base = tf.drive(body, op, None, 0, np.zeros(n))
    x = tf.AMPLITUDE["position"] * np.sin(2 * np.pi * tf.SINE["freq_hz"] * ts)
    y = tf.drive(body, op, "lf", 2, x)[:, 0, 2] - base[:, 0, 2]
    m = int(round(tf.SINE["fit_last_s"] / h))
    H = tf.sine_fit(y[-m:], ts[-m:], tf.SINE["freq_hz"]) / tf.AMPLITUDE["position"]
    ref = saved["results"]["position"]["c_thorax-lf_coxa-pitch"]["sine_11p5hz"]
    assert abs(H) == pytest.approx(ref["gain"], abs=1e-9)
    assert -np.degrees(np.angle(H)) == pytest.approx(ref["phase_lag_deg"], abs=1e-6)
