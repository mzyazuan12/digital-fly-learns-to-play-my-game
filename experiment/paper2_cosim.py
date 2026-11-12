"""Stage 2B co-simulation: rate model <-> tethered FlyBody (design §2.2, §6.2; amendments A1, A3, A4; plan step 6).

One tick = one 0.1 ms RK4 step of the rate model and one 0.1 ms MuJoCo step:

  1. read the front-leg body state s_k (angles, velocities, actuator forces);
  2. sensory current I_sens,k = transduction of s_k (or the condition's substitute);
  3. one RK4 step R_k -> R_{k+1} with DN drive and I_sens,k held over the step;
  4. bridge: R_{k+1} -> joint commands -> set_front_leg_targets_array;
  5. one physics step -> s_{k+1}.

So motor output reaches the afferents with a one-tick delay, plus the neural
and mechanical dynamics, and nothing else. The bridge reads rates clipped to
[0, 1000] Hz, as recorded output is.

Conditions (amendment A1). Drive is both DNg100 (10045 + 10056) at 400:

  open_loop     no body; I_sens = None
  deafferented  body moves under the bridge; I_sens = None. Its neural traces
                are bit-identical to open loop by construction (A1.4).
  closed_loop   each leg's afferents <- that leg's state
  mismatched    left afferents <- right leg's state and vice versa (A1.3).
                FlyBody's legs are mirror-symmetric, so no sign flips.
  scrambled     afferents <- a multivariate Fourier phase-randomised surrogate
                (Prichard & Theiler 1994) of this replicate's own closed-loop
                body-state channels, recorded every tick. Per design §6.2 the
                scrambling is applied to body state before the rectifying
                transduction, and surrogate angles are clipped to the joint
                ranges. The body still moves under the bridge, but its state
                is ignored by the afferents.

``feedback_scale`` multiplies I_sens. It is 1.0 in every pre-registered
condition; 0.0 is used only by the F7 plumbing test (closed loop with zero
gain must equal open loop).

This module records rates, body state, commands and afferent currents. It
computes **no** phase, lag, coherence or coordination read-out (enforced by
tests/test_paper2_cosim.py).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("open_loop", "deafferented", "closed_loop", "mismatched", "scrambled")
FEEDBACK_CONDITIONS = ("closed_loop", "mismatched", "scrambled")
DRIVE_BODY_IDS = (10045, 10056)  # DNg100_L, DNg100_R (amendment A1.1)
DRIVE_AMPLITUDE = 400.0
H = 1e-4
T_S = 2.0
RECORD_EVERY = 10  # 1 ms
RATE_MAX_HZ = 1000.0
SCRAMBLE_SALT = 0x5C  # surrogate RNG = default_rng([seed, replicate, SCRAMBLE_SALT])
FRONT = ("lf", "rf")
STATE_KEYS = ("angle", "velocity", "actuator_force")
LEG_OF = {"T1L": "lf", "T1R": "rf"}


# ----------------------------------------------------------------------------- state vectors
def state_to_vec(state: Mapping[str, Mapping[str, np.ndarray]]) -> np.ndarray:
    """42 channels: for lf then rf, angle(7), velocity(7), actuator_force(7)."""
    return np.concatenate([np.asarray(state[leg][key], dtype=np.float64) for leg in FRONT for key in STATE_KEYS])


def vec_to_state(v: np.ndarray) -> dict:
    v = np.asarray(v, dtype=np.float64)
    out, i = {}, 0
    for leg in FRONT:
        out[leg] = {}
        for key in STATE_KEYS:
            out[leg][key] = v[i:i + 7]
            i += 7
    return out


def angle_channel_index() -> np.ndarray:
    idx = []
    for li, _ in enumerate(FRONT):
        base = li * 21
        idx.extend(range(base, base + 7))
    return np.array(idx)


# ----------------------------------------------------------------------------- surrogate
def phase_scramble(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Multivariate Fourier phase randomisation (Prichard & Theiler 1994).

    X: (n_time, n_channels). Every channel's mean is removed, the same random
    phase is added to every channel at each positive frequency below Nyquist
    (DC and Nyquist untouched), and the means are added back. Each channel's
    amplitude spectrum and every cross-spectrum X_i X_j* are preserved exactly
    (to rounding). Only timing relative to everything else is destroyed.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    mean = X.mean(axis=0)
    F = np.fft.rfft(X - mean, axis=0)
    phi = rng.uniform(0.0, 2.0 * np.pi, F.shape[0])
    phi[0] = 0.0
    if n % 2 == 0:
        phi[-1] = 0.0
    Y = np.fft.irfft(F * np.exp(1j * phi)[:, None], n=n, axis=0)
    return Y + mean


def surrogate_rng(seed: int, replicate: int) -> np.random.Generator:
    return np.random.default_rng([int(seed), int(replicate), SCRAMBLE_SALT])


# ----------------------------------------------------------------------------- core loop
def tick_loop(
    stepper,
    n_steps: int,
    *,
    body=None,
    command_fn: Callable[[np.ndarray], Mapping[str, np.ndarray]] | None = None,
    sensory_fn: Callable[[int, Mapping | None], np.ndarray | None] | None = None,
    on_tick: Callable[[int, Mapping | None, np.ndarray | None, np.ndarray, Mapping | None], None] | None = None,
) -> None:
    """The §2.2 tick, ``n_steps`` times. ``on_tick(k, s_k, I_k, R_{k+1}, cmd_{k+1})`` records."""
    for k in range(n_steps):
        state = body.front_leg_arrays() if body is not None else None
        I = sensory_fn(k, state) if sensory_fn is not None else None
        stepper.step(I)
        cmd = None
        if body is not None:
            cmd = command_fn(np.clip(stepper.R, 0.0, RATE_MAX_HZ))
            body.set_front_leg_targets_array(cmd["T1L"], cmd["T1R"])
            body.step()
        if on_tick is not None:
            on_tick(k, state, I, stepper.R, cmd)


# ----------------------------------------------------------------------------- network
_NET_CACHE: dict = {}


@dataclass
class Network:
    """The authors' 4,310-neuron mCNS network with the seed-1 parameter block (Paper 1)."""

    seed: int = 1
    param_block: int = 1024
    W: object = None
    tau: np.ndarray = None
    gain: np.ndarray = None
    threshold: np.ndarray = None
    fr_cap: np.ndarray = None
    dn_input: np.ndarray = None
    drive_rows: tuple = ()
    body_ids: np.ndarray = None
    sampler: str = ""

    @classmethod
    def load(cls, seed: int = 1, param_block: int = 1024) -> "Network":
        key = (seed, param_block)
        if key in _NET_CACHE:
            return _NET_CACHE[key]
        from scipy.sparse import csr_matrix

        from flybrain.authors_mcns import load_authors_W, load_authors_wtable
        from flybrain.pugliese_rate import PuglieseRateParams, reweight_connectivity, sample_cell_parameters_block, set_sizes

        t = load_authors_wtable()
        p = PuglieseRateParams(stim_amplitude=DRIVE_AMPLITUDE)
        d = sample_cell_parameters_block(len(t), param_block, seed, p)
        g, th = set_sizes(t["size"].to_numpy(float), d["gain"], d["threshold"])
        W, _, _ = load_authors_W()
        body_ids = t["bodyId"].to_numpy(np.int64)
        rows = tuple(int(np.flatnonzero(body_ids == b)[0]) for b in DRIVE_BODY_IDS)
        dn = np.zeros(len(t))
        dn[list(rows)] = DRIVE_AMPLITUDE
        net = cls(seed, param_block, csr_matrix(reweight_connectivity(W, p.exc_multiplier, p.inh_multiplier)),
                  d["tau"], g, th, d["fr_cap"], dn, rows, body_ids, d["sampler"])
        _NET_CACHE[key] = net
        return net

    def stepper(self, replicate: int, h: float = H):
        from flybrain.rate_stepper import RateStepper

        r = int(replicate)
        return RateStepper(W=self.W, tau=self.tau[r], gain=self.gain[r], threshold=self.threshold[r],
                           fr_cap=self.fr_cap[r], dn_input=self.dn_input, h=h)


# ----------------------------------------------------------------------------- run
@dataclass
class RunConfig:
    condition: str
    replicate: int
    seed: int = 1
    T: float = T_S
    hair_plate_directions: Mapping[str, str] | None = None
    r_star_hz: float = 100.0
    feedback_scale: float = 1.0
    # Which leg's state each leg's afferents read. None: swap for mismatched
    # (amendment A1.3), identity otherwise. Overridden only by the F7 plumbing test.
    feedback_map: Mapping[str, str] | None = None
    record_state_every_tick: bool = False

    def __post_init__(self):
        if self.condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {CONDITIONS}")
        if self.feedback_map is None:
            swap = self.condition == "mismatched"
            self.feedback_map = {"lf": "rf", "rf": "lf"} if swap else {"lf": "lf", "rf": "rf"}
        if set(self.feedback_map) != set(FRONT) or not set(self.feedback_map.values()) <= set(FRONT):
            raise ValueError("feedback_map must map lf/rf to lf/rf")
        if self.condition in FEEDBACK_CONDITIONS and self.hair_plate_directions is None:
            raise ValueError("feedback conditions need hair_plate_directions (no default, amendment A1.5)")


def run(
    cfg: RunConfig,
    *,
    net: Network,
    params=None,
    body=None,
    bridge=None,
    transducer=None,
    surrogate_source: np.ndarray | None = None,
    initial_angles: Mapping[str, np.ndarray] | None = None,
    warmup_s: float = 0.1,
) -> dict:
    """One replicate of one condition. Returns recorded arrays (1 ms grid unless noted)."""
    n_steps = int(round(cfg.T / H))
    n_rec = n_steps // RECORD_EVERY + 1
    stepper = net.stepper(cfg.replicate)
    if cfg.condition == "open_loop":
        body = None  # open loop has no body, even if one is passed
    out = {
        "condition": cfg.condition,
        "replicate": int(cfg.replicate),
        "rates_hz": np.zeros((net.W.shape[0], n_rec), dtype=np.float32),
        "dt_s": RECORD_EVERY * H,
    }
    out["rates_hz"][:, 0] = np.clip(stepper.R, 0.0, RATE_MAX_HZ)

    sensory_fn = None
    aff_rows = None
    if cfg.condition in FEEDBACK_CONDITIONS:
        if transducer is None:
            raise ValueError(f"{cfg.condition} needs a transducer")
        if dict(transducer.hair_plate_directions) != dict(cfg.hair_plate_directions):
            raise ValueError("transducer hair-plate directions differ from the run config")
        if any(v != float(cfg.r_star_hz) for v in transducer.r_star.values()):
            raise ValueError("transducer r* differs from the run config")
        aff_rows = np.concatenate([idx for idx, _ in transducer.groups.values()])
        out["afferent_rows"] = aff_rows
        out["afferent_current"] = np.zeros((n_rec, aff_rows.size), dtype=np.float64)
        Ibuf = np.zeros(net.W.shape[0])
        scale = float(cfg.feedback_scale)
        if cfg.condition == "scrambled":
            if surrogate_source is None:
                raise ValueError("scrambled needs this replicate's closed-loop body state (every tick)")
            if surrogate_source.shape != (n_steps, 42):
                raise ValueError(f"surrogate source must be ({n_steps}, 42)")
            sur = phase_scramble(surrogate_source, surrogate_rng(cfg.seed, cfg.replicate))
            ai = angle_channel_index()
            lo = np.concatenate([params.lower["T1L"], params.lower["T1R"]])
            hi = np.concatenate([params.upper["T1L"], params.upper["T1R"]])
            sur[:, ai] = np.clip(sur[:, ai], lo, hi)
            out["surrogate_rng"] = [int(cfg.seed), int(cfg.replicate), SCRAMBLE_SALT]

            def sensory_fn(k, state):
                I = transducer.currents(vec_to_state(sur[k]), out=Ibuf)
                return I * scale if scale != 1.0 else I
        else:
            fmap = dict(cfg.feedback_map)

            def sensory_fn(k, state):
                mapped = {leg: state[fmap[leg]] for leg in FRONT}
                I = transducer.currents(mapped, out=Ibuf)
                return I * scale if scale != 1.0 else I

    command_fn = None
    if cfg.condition != "open_loop":
        if body is None or bridge is None or params is None:
            raise ValueError(f"{cfg.condition} needs body, bridge and params")
        params.validate()

        def command_fn(R):
            return bridge.commands(R, params)

        cmd0 = command_fn(np.zeros(net.W.shape[0]))
        if initial_angles is None and params.mode == "position":
            initial_angles = {"T1L": cmd0["T1L"], "T1R": cmd0["T1R"]}
        body.reset(
            dict(zip(body.dof_names["lf"], cmd0["T1L"])),
            dict(zip(body.dof_names["rf"], cmd0["T1R"])),
            warmup_s=warmup_s,
            left_angles=None if initial_angles is None else dict(zip(body.dof_names["lf"], initial_angles["T1L"])),
            right_angles=None if initial_angles is None else dict(zip(body.dof_names["rf"], initial_angles["T1R"])),
        )
        out["angle"] = np.zeros((n_rec, 2, 7))
        out["velocity"] = np.zeros((n_rec, 2, 7))
        out["actuator_force"] = np.zeros((n_rec, 2, 7))
        out["command"] = np.zeros((n_rec, 2, 7))
        s0 = body.front_leg_arrays()
        for li, leg in enumerate(FRONT):
            out["angle"][0, li] = s0[leg]["angle"]
            out["velocity"][0, li] = s0[leg]["velocity"]
            out["actuator_force"][0, li] = s0[leg]["actuator_force"]
            out["command"][0, li] = cmd0[("T1L", "T1R")[li]]
        if cfg.record_state_every_tick:
            out["state_every_tick"] = np.zeros((n_steps, 42))

    def on_tick(k, state, I, R, cmd):
        if cfg.record_state_every_tick and state is not None:
            out["state_every_tick"][k] = state_to_vec(state)
        if (k + 1) % RECORD_EVERY:
            return
        j = (k + 1) // RECORD_EVERY
        out["rates_hz"][:, j] = np.clip(R, 0.0, RATE_MAX_HZ)
        if aff_rows is not None:
            out["afferent_current"][j] = I[aff_rows]
        if body is not None:
            s = body.front_leg_arrays()
            for li, leg in enumerate(FRONT):
                out["angle"][j, li] = s[leg]["angle"]
                out["velocity"][j, li] = s[leg]["velocity"]
                out["actuator_force"][j, li] = s[leg]["actuator_force"]
                out["command"][j, li] = cmd[("T1L", "T1R")[li]]

    tick_loop(stepper, n_steps, body=body, command_fn=command_fn, sensory_fn=sensory_fn, on_tick=on_tick)
    return out


def build_components(frozen: Mapping, *, actuation: str | None = None, hair_plate_directions=None, r_star_hz=None):
    """Body, bridge, params and (optionally) transducer from a frozen-bridge dict."""
    from organism.flybody_tethered import TetheredFlyBody
    from organism.mn_joint_bridge import BridgeParams, MNJointBridge
    from organism.proprio_transduction import ProprioTransducer

    params = BridgeParams.from_dict(frozen["bridge_params"])
    body = TetheredFlyBody(actuation or params.mode)
    bridge = MNJointBridge()
    transducer = None
    if hair_plate_directions is not None:
        transducer = ProprioTransducer(hair_plate_directions, r_star_hz=r_star_hz or frozen["sensory"]["r_star_hz"])
    return body, bridge, params, transducer


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    """Single run, for inspection. The stage 2B runner (plan step 9) is separate."""
    import argparse

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--condition", required=True, choices=[c for c in CONDITIONS if c != "scrambled"])
    p.add_argument("--replicate", type=int, required=True)
    p.add_argument("--frozen", type=Path, default=ROOT / "data" / "paper2" / "stage2B_bridge_frozen.json")
    p.add_argument("--hair-plates", type=str, default=None, help="coxa,trochanter e.g. protraction,flexion")
    p.add_argument("--T", type=float, default=T_S)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    if a.out.exists():
        raise FileExistsError(a.out)
    frozen = json.loads(a.frozen.read_text())
    hp = None
    if a.hair_plates:
        c, t = a.hair_plates.split(",")
        hp = {"hair_plate_coxa": c, "hair_plate_trochanter": t}
    body, bridge, params, tr = build_components(frozen, hair_plate_directions=hp)
    res = run(RunConfig(a.condition, a.replicate, T=a.T, hair_plate_directions=hp), net=Network.load(),
              params=params, body=body, bridge=bridge, transducer=tr)
    res["frozen_sha256"] = file_sha256(a.frozen)
    np.savez_compressed(a.out, **{k: v for k, v in res.items() if isinstance(v, np.ndarray)},
                        meta=json.dumps({k: v for k, v in res.items() if not isinstance(v, np.ndarray)}))
    print(a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
