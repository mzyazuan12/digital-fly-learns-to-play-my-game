"""Stage 2B plan step 6: the co-simulation tick loop (design §2.2, §6.2, §7 F7, §8).

- F7 bit-identities: deafferented = open loop; closed loop with feedback gain 0 =
  open loop; mismatched with a left-to-left map = closed loop; scrambled with an
  identity "scramble" = closed loop. Plus sanity checks that feedback is live
  (closed loop differs from open loop, and swapped legs differ from closed loop).
- Open loop equals RateStepper.run (the G1-validated integrator) bit for bit.
- Surrogate: per-channel amplitude spectrum and all cross-spectra preserved,
  real output, mean kept, seed-reproducible.
- 2-neuron toy: motor output reaches the afferent with exactly one tick of delay.
- The module computes no phase, lag or coordination read-out (AST check).

Real-network runs are 0.3 s long (3,000 ticks) to keep the file under ~1 min.
Bridge parameters here are test values, not the frozen ones.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from experiment import paper2_cosim as cs
from experiment.paper2_cosim import RunConfig, phase_scramble, run, state_to_vec, surrogate_rng, tick_loop, vec_to_state

ROOT = Path(__file__).resolve().parents[1]
T_TEST = 0.3
HP = {"hair_plate_coxa": "protraction", "hair_plate_trochanter": "flexion"}


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def net():
    return cs.Network.load()


@pytest.fixture(scope="module")
def parts():
    pytest.importorskip("flygym")
    from organism.flybody_tethered import TetheredFlyBody, load_walking_clip_front_legs
    from organism.mn_joint_bridge import BridgeParams, MNJointBridge
    from organism.proprio_transduction import ProprioTransducer

    body = TetheredFlyBody("position")
    clip = load_walking_clip_front_legs()
    pooled = np.concatenate([clip["lf"], clip["rf"]])
    desc = body.describe()["dofs"]
    p = BridgeParams.unfitted("position")
    for leg, bl in (("T1L", "lf"), ("T1R", "rf")):
        p.offset[leg] = pooled.mean(axis=0)
        p.scale[leg] = (np.percentile(pooled, 95, axis=0) - np.percentile(pooled, 5, axis=0)) / 2
        p.scale[leg][[0, 1]] = 0.0
        p.c[leg] = np.zeros(7)
        p.s[leg] = np.full(7, 10.0)
        p.lower[leg] = np.array([desc[n]["range_rad"][0] for n in body.dof_names[bl]])
        p.upper[leg] = np.array([desc[n]["range_rad"][1] for n in body.dof_names[bl]])
    return {"body": body, "bridge": MNJointBridge(), "params": p, "tr": ProprioTransducer(HP)}


def _run(net, parts, condition, **kw):
    surrogate = kw.pop("surrogate_source", None)
    cfg = RunConfig(condition, 0, T=T_TEST, hair_plate_directions=HP if condition in cs.FEEDBACK_CONDITIONS else None, **kw)
    return run(cfg, net=net, params=parts["params"], body=parts["body"], bridge=parts["bridge"],
               transducer=parts["tr"], surrogate_source=surrogate)


@pytest.fixture(scope="module")
def open_loop(net, parts):
    return _run(net, parts, "open_loop")


@pytest.fixture(scope="module")
def closed(net, parts):
    return _run(net, parts, "closed_loop", record_state_every_tick=True)


def _same(a, b, keys=("rates_hz",)):
    return all(np.array_equal(a[k], b[k]) for k in keys)


BODY_KEYS = ("rates_hz", "angle", "velocity", "actuator_force", "command")


# --------------------------------------------------------------------------- drive and integrator
def test_bilateral_drive(net):
    assert tuple(int(net.body_ids[r]) for r in net.drive_rows) == (10045, 10056)
    assert np.flatnonzero(net.dn_input).tolist() == sorted(net.drive_rows)
    assert np.all(net.dn_input[list(net.drive_rows)] == 400.0)


def test_open_loop_is_the_g1_stepper(net, open_loop):
    ref = net.stepper(0).run(T=T_TEST)
    assert np.array_equal(open_loop["rates_hz"], ref.astype(np.float32))


# --------------------------------------------------------------------------- F7 bit-identities
def test_deafferented_equals_open_loop(net, parts, open_loop):
    d = _run(net, parts, "deafferented")
    assert np.array_equal(d["rates_hz"], open_loop["rates_hz"])
    assert np.ptp(d["angle"][:, 0, 2]) > 0.05  # the body did move


def test_feedback_is_live(open_loop, closed):
    assert np.max(closed["afferent_current"]) > 0
    assert not np.array_equal(closed["rates_hz"], open_loop["rates_hz"])


def test_closed_loop_with_zero_gain_equals_open_loop(net, parts, open_loop):
    z = _run(net, parts, "closed_loop", feedback_scale=0.0)
    assert np.array_equal(z["rates_hz"], open_loop["rates_hz"])


def test_mismatched_with_identity_map_equals_closed_loop(net, parts, closed):
    m = _run(net, parts, "mismatched", feedback_map={"lf": "lf", "rf": "rf"})
    assert _same(m, closed, BODY_KEYS + ("afferent_current",))


def test_mismatched_swaps_legs(net, parts, closed):
    m = _run(net, parts, "mismatched")
    assert not np.array_equal(m["rates_hz"], closed["rates_hz"])


def test_scrambled_with_identity_scramble_equals_closed_loop(net, parts, closed, monkeypatch):
    """Feeding back this replicate's own recorded closed-loop state, unscrambled,
    reproduces the closed loop exactly, so the scrambled path changes timing only.
    Angle clipping to the joint ranges is part of that path; the identity holds
    because recorded angles never leave the ranges by more than rounding."""
    monkeypatch.setattr(cs, "phase_scramble", lambda X, rng: np.array(X, copy=True))
    src = closed["state_every_tick"]
    ai = cs.angle_channel_index()
    p = parts["params"]
    lo = np.concatenate([p.lower["T1L"], p.lower["T1R"]])
    hi = np.concatenate([p.upper["T1L"], p.upper["T1R"]])
    inside = np.all((src[:, ai] >= lo) & (src[:, ai] <= hi))
    s = _run(net, parts, "scrambled", surrogate_source=src)
    if inside:
        assert _same(s, closed, BODY_KEYS + ("afferent_current",))
    else:
        pytest.fail("recorded angles left the joint ranges; clipping makes the identity inexact")


def test_scrambled_runs_differ_and_reproduce(net, parts, closed):
    src = closed["state_every_tick"]
    a = _run(net, parts, "scrambled", surrogate_source=src)
    b = _run(net, parts, "scrambled", surrogate_source=src)
    assert _same(a, b, BODY_KEYS + ("afferent_current",))
    assert not np.array_equal(a["afferent_current"], closed["afferent_current"])
    assert a["surrogate_rng"] == [1, 0, cs.SCRAMBLE_SALT]
    with pytest.raises(ValueError):
        _run(net, parts, "scrambled")


def test_transducer_must_match_the_run_config(net, parts):
    other = {"hair_plate_coxa": "retraction", "hair_plate_trochanter": "flexion"}
    with pytest.raises(ValueError, match="hair-plate"):
        run(RunConfig("closed_loop", 0, T=0.001, hair_plate_directions=other), net=net, params=parts["params"],
            body=parts["body"], bridge=parts["bridge"], transducer=parts["tr"])
    with pytest.raises(ValueError, match="r\\*"):
        run(RunConfig("closed_loop", 0, T=0.001, hair_plate_directions=HP, r_star_hz=50.0), net=net,
            params=parts["params"], body=parts["body"], bridge=parts["bridge"], transducer=parts["tr"])


def test_feedback_conditions_need_a_hair_plate_direction():
    for c in cs.FEEDBACK_CONDITIONS:
        with pytest.raises(ValueError):
            RunConfig(c, 0)
    assert RunConfig("mismatched", 0, hair_plate_directions=HP).feedback_map == {"lf": "rf", "rf": "lf"}
    assert RunConfig("closed_loop", 0, hair_plate_directions=HP).feedback_map == {"lf": "lf", "rf": "rf"}


# --------------------------------------------------------------------------- surrogate
def _synthetic(n=4000, ch=6, seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 1e-4
    base = np.sin(2 * np.pi * 11.5 * t)[:, None] * rng.uniform(0.5, 2, ch)
    X = base + 0.3 * rng.standard_normal((n, ch)) + rng.uniform(-1, 1, ch)
    X[:, 1] = np.gradient(X[:, 0], 1e-4)  # a derived channel, like velocity
    return X


@pytest.mark.parametrize("n", [4000, 3999])
def test_surrogate_preserves_spectra_and_cross_spectra(n):
    X = _synthetic(n)
    Y = phase_scramble(X, surrogate_rng(1, 7))
    assert Y.shape == X.shape and np.isrealobj(Y)
    assert np.allclose(Y.mean(axis=0), X.mean(axis=0), atol=1e-10)
    FX = np.fft.rfft(X - X.mean(0), axis=0)
    FY = np.fft.rfft(Y - Y.mean(0), axis=0)
    scale = np.abs(FX).max()
    assert np.max(np.abs(np.abs(FY) - np.abs(FX))) / scale < 1e-10
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            SX = FX[:, i] * np.conj(FX[:, j])
            SY = FY[:, i] * np.conj(FY[:, j])
            assert np.max(np.abs(SY - SX)) / (scale**2) < 1e-10
    assert np.max(np.abs(Y - X)) > 0.1  # timing really changed


def test_surrogate_seed_reproducibility():
    X = _synthetic()
    a = phase_scramble(X, surrogate_rng(1, 5))
    b = phase_scramble(X, surrogate_rng(1, 5))
    c = phase_scramble(X, surrogate_rng(1, 6))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_state_vector_round_trip():
    rng = np.random.default_rng(0)
    st = {leg: {k: rng.standard_normal(7) for k in cs.STATE_KEYS} for leg in cs.FRONT}
    back = vec_to_state(state_to_vec(st))
    for leg in cs.FRONT:
        for k in cs.STATE_KEYS:
            assert np.array_equal(back[leg][k], st[leg][k])
    assert cs.angle_channel_index().tolist() == list(range(0, 7)) + list(range(21, 28))


# --------------------------------------------------------------------------- one-tick delay
class _ToyBody:
    """Perfect servo: after step(), the left-leg angle equals the last command."""

    def __init__(self):
        self.cmd = np.zeros(7)
        self.angle = np.zeros(7)

    def set_front_leg_targets_array(self, left, right):
        self.cmd = np.array(left, dtype=float)

    def step(self):
        self.angle = self.cmd.copy()

    def front_leg_arrays(self):
        z = np.zeros(7)
        return {"lf": {"angle": self.angle.copy(), "velocity": z, "actuator_force": z},
                "rf": {"angle": z, "velocity": z, "actuator_force": z}}


def test_two_neuron_toy_has_exactly_one_tick_of_delay():
    """Neuron 0 is a 'motor neuron' with DN drive; the leg angle copies its rate;
    neuron 1 is an 'afferent' whose input current equals the angle. The input on
    tick k must equal the motor rate after tick k-1 (R_k), so the delay is exactly one tick."""
    from flybrain.rate_stepper import RateStepper

    st = RateStepper(W=np.zeros((2, 2)), tau=np.array([0.02, 0.02]), gain=np.ones(2), threshold=np.zeros(2),
                     fr_cap=np.full(2, 200.0), dn_input=np.array([50.0, 0.0]), h=1e-4, pulse_start=0.0, pulse_end=1.0)
    body = _ToyBody()
    seen_I, seen_R = [], []

    def sensory_fn(k, state):
        return np.array([0.0, state["lf"]["angle"][0]])

    def command_fn(R):
        return {"T1L": np.full(7, R[0]), "T1R": np.zeros(7)}

    def on_tick(k, state, I, R, cmd):
        seen_I.append(I[1])
        seen_R.append(R.copy())

    tick_loop(st, 200, body=body, command_fn=command_fn, sensory_fn=sensory_fn, on_tick=on_tick)
    R = np.array(seen_R)
    assert seen_I[0] == 0.0
    assert np.array_equal(np.array(seen_I[1:]), R[:-1, 0])  # I_k = R_k (motor rate one tick earlier)
    assert R[-1, 0] > 0 and R[-1, 1] > 0  # the afferent was driven


# --------------------------------------------------------------------------- no read-outs
def test_cosim_computes_no_coordination_readout():
    tree = ast.parse((ROOT / "experiment" / "paper2_cosim.py").read_text())
    banned_modules = {"experiment.leg_coordination", "scipy.signal"}
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
