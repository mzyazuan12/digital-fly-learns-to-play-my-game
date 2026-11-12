"""Body state -> afferent input current for stage 2B (design §4.3, §5.2; amendment A3; plan step 5).

ENGINEERED throughout. Which afferents receive input, and from which leg and
channel, comes from ``data/paper2/stage2B_sensory_table.json`` (plan step 2).
Every afferent in a (leg, channel) group receives the same dimensionless
stimulus u >= 0, and its input current is

    I_i = g_c * 7.5 * s_i * u,     g_c = 1 + (200 / 7.5) * atanh(r* / 200)

where s_i is the afferent's size normalised by the wTable median. With the
rate model's own parameters (threshold theta_raw * s_i, gain a_raw / s_i), the
steady rate is r_max * tanh((a_raw / r_max) * (7.5 * g_c * u - theta_raw))+.
That does not depend on s_i, and u = 0 gives I = 0 (identical to open loop).
For nominal parameters (a = 1, theta = 7.5, r_max = 200) the rate at u = 1 is
r*. r* = 100 Hz by default (NEEDS SOURCE, design B4); robustness arm R2 uses 50
and 200 Hz.

Channels and stimuli (per leg; the T1L afferents read the lf state, T1R the rf state):

  hair_plate_coxa         coxa pitch angle, limit detector
  hair_plate_trochanter   trochanter-femur pitch angle, limit detector (joint GUESS)
  chordotonal_femur_tibia femur-tibia angle and velocity, direction-free:
                          u = 0.5 |theta - p50| / p95|theta - p50| + 0.5 |dtheta| / p95|dtheta|
  campaniform_trochanter  |trochanter-femur pitch actuator force| / p95 (cuticular strain proxy)

A limit detector toward angle direction d = +1 fires as u = (theta - p80) / (p95 - p80)
(rectified); toward d = -1, as u = (p20 - theta) / (p20 - p5). The onset (80th) and
saturation (95th) percentiles are a GUESS (design §4.3).

**Hair-plate direction has no default** (amendment A1.5). The caller passes the
anatomical direction of each hair-plate channel: ``hair_plate_coxa`` in
{"protraction", "retraction"} and ``hair_plate_trochanter`` in {"flexion",
"extension"}. ``FACTORIAL`` lists the four combinations, none marked primary.
Anatomical directions become angle directions through the agonist angle signs
measured from FlyBody geometry in ``stage2B_mn_joint_table.json``. Coxa swing is
protraction and femur/tr flex is coxa-trochanter flexion.

Normalisation (``stage2B_sensory_norm.json``) uses the walking clip's own angles
and velocities (amendment A3), each leg from its own clip leg. The one
exception is the campaniform force proxy, which needs a replay of the clip on
the tethered FlyBody (position servos).

Build the normalisation file (needs FlyGym; a few seconds):
  .venv/bin/python -m organism.proprio_transduction
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SENSORY_TABLE_PATH = ROOT / "data" / "paper2" / "stage2B_sensory_table.json"
MN_JOINT_TABLE_PATH = ROOT / "data" / "paper2" / "stage2B_mn_joint_table.json"
NORM_PATH = ROOT / "data" / "paper2" / "stage2B_sensory_norm.json"
LEGS = ("T1L", "T1R")
BODY_LEG = {"T1L": "lf", "T1R": "rf"}
NOMINAL_THRESHOLD = 7.5
NOMINAL_RMAX = 200.0
DEFAULT_R_STAR_HZ = 100.0
# Index of each DOF in FlyBody actuation order (organism.flybody_tethered).
DOF_INDEX = {"coxa_pitch": 2, "trfe_pitch": 4, "tibia_pitch": 5}
CHANNELS = ("hair_plate_coxa", "hair_plate_trochanter", "chordotonal_femur_tibia", "campaniform_trochanter")
HAIR_PLATE_CHOICES = {
    "hair_plate_coxa": ("protraction", "retraction"),
    "hair_plate_trochanter": ("flexion", "extension"),
}
FACTORIAL = tuple(
    {"hair_plate_coxa": c, "hair_plate_trochanter": t}
    for c in HAIR_PLATE_CHOICES["hair_plate_coxa"]
    for t in HAIR_PLATE_CHOICES["hair_plate_trochanter"]
)
HP_ONSET_PCT, HP_SAT_PCT = 80.0, 95.0


def gain_for_rate(r_star_hz: float) -> float:
    """g_c such that u = 1 gives r* for nominal parameters (a = 1, theta = 7.5, r_max = 200)."""
    if not 0.0 < r_star_hz < NOMINAL_RMAX:
        raise ValueError("r* must lie in (0, 200) Hz")
    return 1.0 + (NOMINAL_RMAX / NOMINAL_THRESHOLD) * float(np.arctanh(r_star_hz / NOMINAL_RMAX))


def steady_rate(current, gain, threshold, fr_cap):
    """Steady-state rate of the half-tanh model for a constant total input."""
    return np.maximum(fr_cap * np.tanh((gain / fr_cap) * (np.asarray(current) - threshold)), 0.0)


# ----------------------------------------------------------------------------- normalisation
def _pcts(x: np.ndarray) -> dict[str, float]:
    return {f"p{p:g}": float(np.percentile(x, p)) for p in (5, 20, 50, 80, 95)}


def clip_feature_stats(clip: Mapping[str, np.ndarray], timestep: float) -> dict:
    """Percentiles from the clip's own angles and velocities, per leg."""
    out = {}
    for leg in LEGS:
        a = np.asarray(clip[BODY_LEG[leg]])
        vel = np.gradient(a, timestep, axis=0)
        ti = a[:, DOF_INDEX["tibia_pitch"]]
        ti50 = float(np.percentile(ti, 50))
        out[leg] = {
            "coxa_pitch_angle": _pcts(a[:, DOF_INDEX["coxa_pitch"]]),
            "trfe_pitch_angle": _pcts(a[:, DOF_INDEX["trfe_pitch"]]),
            "tibia_pitch_angle_p50": ti50,
            "tibia_pitch_abs_dev_p95": float(np.percentile(np.abs(ti - ti50), 95)),
            "tibia_pitch_abs_vel_p95": float(np.percentile(np.abs(vel[:, DOF_INDEX["tibia_pitch"]]), 95)),
        }
    return out


def replay_force_stats(body, clip: Mapping[str, np.ndarray]) -> dict:
    """p95 of |trochanter-femur pitch actuator force| while the position servos replay the clip."""
    if body.front_actuation != "position":
        raise ValueError("the campaniform normalisation replays the clip with position servos")
    n = clip["lf"].shape[0]
    body.reset(
        dict(zip(body.dof_names["lf"], clip["lf"][0])),
        dict(zip(body.dof_names["rf"], clip["rf"][0])),
        left_angles=dict(zip(body.dof_names["lf"], clip["lf"][0])),
        right_angles=dict(zip(body.dof_names["rf"], clip["rf"][0])),
    )
    f = {leg: np.empty(n) for leg in LEGS}
    for k in range(n):
        body.set_front_leg_targets_array(clip["lf"][k], clip["rf"][k])
        body.step()
        s = body.front_leg_arrays()
        for leg in LEGS:
            f[leg][k] = s[BODY_LEG[leg]]["actuator_force"][DOF_INDEX["trfe_pitch"]]
    return {leg: {"trfe_pitch_abs_force_p95": float(np.percentile(np.abs(f[leg]), 95))} for leg in LEGS}


def build_norm(body=None, clip=None) -> dict:
    from organism.flybody_tethered import TetheredFlyBody, load_walking_clip_front_legs

    clip = clip if clip is not None else load_walking_clip_front_legs()
    body = body if body is not None else TetheredFlyBody("position")
    stats = clip_feature_stats(clip, float(clip["timestep"]))
    force = replay_force_stats(body, clip)
    for leg in LEGS:
        stats[leg].update(force[leg])
    return {
        "what": "Stage 2B afferent stimulus normalisation (design §4.3, §5.2; amendment A3; plan step 5)",
        "source_clip": "flygym_demo/ball_flybody_data/assets/ball_flybody_clip.npz (NeuroMechFly v1 tethered ball walking; FlyGym FlyBody conversion)",
        "clip_timestep_s": float(clip["timestep"]),
        "angles_and_velocities": "the clip's own angles (0.1 ms cubic interpolant of the Savitzky-Golay-filtered 100 fps data) and their numerical derivative; T1L from the LF trace, T1R from the RF trace",
        "campaniform_force": "replay of the clip on the tethered FlyBody with position servos (organism.flybody_tethered), trochanter-femur pitch actuator force",
        "hair_plate_onset_and_saturation_percentiles": [HP_ONSET_PCT, HP_SAT_PCT],
        "units": {"angle": "rad", "velocity": "rad/s", "force": "model torque units"},
        "stats": stats,
    }


# ----------------------------------------------------------------------------- transducer
class ProprioTransducer:
    """Front-leg body state -> input current for every neuron (zeros outside the included afferents)."""

    def __init__(
        self,
        hair_plate_directions: Mapping[str, str],
        *,
        r_star_hz: float | Mapping[str, float] = DEFAULT_R_STAR_HZ,
        sensory_table: dict | None = None,
        norm: dict | None = None,
        mn_joint_table: dict | None = None,
        n_neurons: int = 4310,
    ) -> None:
        if hair_plate_directions is None or set(hair_plate_directions) != set(HAIR_PLATE_CHOICES):
            raise ValueError(f"hair_plate_directions must name exactly {sorted(HAIR_PLATE_CHOICES)}; there is no default")
        for ch, choice in hair_plate_directions.items():
            if choice not in HAIR_PLATE_CHOICES[ch]:
                raise ValueError(f"{ch} direction must be one of {HAIR_PLATE_CHOICES[ch]}, not {choice!r}")
        self.hair_plate_directions = dict(hair_plate_directions)
        st = sensory_table if sensory_table is not None else json.loads(SENSORY_TABLE_PATH.read_text())
        nm = norm if norm is not None else json.loads(NORM_PATH.read_text())
        mj = mn_joint_table if mn_joint_table is not None else json.loads(MN_JOINT_TABLE_PATH.read_text())
        self.stats = nm["stats"]
        self.n_neurons = int(n_neurons)
        if isinstance(r_star_hz, Mapping):
            self.r_star = {ch: float(r_star_hz[ch]) for ch in CHANNELS}
        else:
            self.r_star = {ch: float(r_star_hz) for ch in CHANNELS}
        self.g = {ch: gain_for_rate(self.r_star[ch]) for ch in CHANNELS}

        # Afferent rows and the per-row current scale g_c * 7.5 * s_i, per (leg, channel).
        self.groups: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        for leg in LEGS:
            for ch in CHANNELS:
                rows = [a for a in st["afferents"] if a["included_2B"] and a["leg"] == leg and a["channel"] == ch]
                if not rows:
                    continue
                idx = np.array([a["row"] for a in rows], dtype=np.int64)
                size = np.array([a["normalized_size"] for a in rows], dtype=np.float64)
                self.groups[(leg, ch)] = (idx, self.g[ch] * NOMINAL_THRESHOLD * size)

        # Anatomical hair-plate direction -> direction of increasing FlyBody angle.
        sign = {
            "hair_plate_coxa": {leg: int(mj["channels"]["coxa_pitch"]["agonist_angle_sign"][leg]) for leg in LEGS},
            "hair_plate_trochanter": {leg: int(mj["channels"]["trfe_pitch"]["agonist_angle_sign"][leg]) for leg in LEGS},
        }
        agonist_direction = {"hair_plate_coxa": "protraction", "hair_plate_trochanter": "flexion"}
        self.angle_direction = {
            ch: {leg: sign[ch][leg] * (1 if self.hair_plate_directions[ch] == agonist_direction[ch] else -1) for leg in LEGS}
            for ch in HAIR_PLATE_CHOICES
        }

    @staticmethod
    def _limit(theta: float, pct: Mapping[str, float], d: int) -> float:
        if d > 0:
            on, sat = pct["p80"], pct["p95"]
            return max(0.0, (theta - on) / (sat - on))
        on, sat = pct["p20"], pct["p5"]
        return max(0.0, (on - theta) / (on - sat))

    def stimuli(self, state: Mapping[str, Mapping[str, np.ndarray]]) -> dict[tuple[str, str], float]:
        """u per (leg, channel) from ``TetheredFlyBody.front_leg_arrays()``-style state."""
        out = {}
        for leg in LEGS:
            s = state[BODY_LEG[leg]]
            ang, vel, frc = s["angle"], s["velocity"], s["actuator_force"]
            st = self.stats[leg]
            out[(leg, "hair_plate_coxa")] = self._limit(
                float(ang[DOF_INDEX["coxa_pitch"]]), st["coxa_pitch_angle"], self.angle_direction["hair_plate_coxa"][leg]
            )
            out[(leg, "hair_plate_trochanter")] = self._limit(
                float(ang[DOF_INDEX["trfe_pitch"]]), st["trfe_pitch_angle"], self.angle_direction["hair_plate_trochanter"][leg]
            )
            ti = DOF_INDEX["tibia_pitch"]
            out[(leg, "chordotonal_femur_tibia")] = (
                0.5 * abs(float(ang[ti]) - st["tibia_pitch_angle_p50"]) / st["tibia_pitch_abs_dev_p95"]
                + 0.5 * abs(float(vel[ti])) / st["tibia_pitch_abs_vel_p95"]
            )
            out[(leg, "campaniform_trochanter")] = (
                abs(float(frc[DOF_INDEX["trfe_pitch"]])) / st["trfe_pitch_abs_force_p95"]
            )
        return out

    def currents(self, state: Mapping[str, Mapping[str, np.ndarray]], out: np.ndarray | None = None) -> np.ndarray:
        """Input current vector (n_neurons,): g_c * 7.5 * s_i * u on included afferents, 0 elsewhere."""
        I = np.zeros(self.n_neurons) if out is None else out
        if out is not None:
            I[:] = 0.0
        u = self.stimuli(state)
        for key, (idx, scale) in self.groups.items():
            I[idx] = scale * u[key]
        return I

    def describe(self) -> dict:
        return {
            "hair_plate_directions": self.hair_plate_directions,
            "angle_direction": self.angle_direction,
            "r_star_hz": self.r_star,
            "g_c": self.g,
            "n_afferents": {f"{k[0]}:{k[1]}": int(v[0].size) for k, v in self.groups.items()},
        }


def main(argv: list[str] | None = None) -> int:
    import argparse
    import subprocess
    from datetime import datetime, timezone

    p = argparse.ArgumentParser(description="Build data/paper2/stage2B_sensory_norm.json")
    p.add_argument("--out", type=Path, default=NORM_PATH)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    if a.out.exists() and not a.overwrite:
        raise FileExistsError(f"Refusing to overwrite {a.out}; pass --overwrite")
    norm = build_norm()
    norm["generated_utc"] = datetime.now(timezone.utc).isoformat()
    try:
        norm["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
    except Exception:
        norm["git_commit"] = None
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(norm, indent=1) + "\n")
    print(json.dumps(norm["stats"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
