"""Motor neuron -> FlyBody joint bridge for stage 2B (design §3.3-3.4, plan step 4).

ENGINEERED throughout. Every one of the 129 motor neurons in the authors'
4,310-neuron wTable (rows with a ``motor module``) is either assigned to a
joint channel of its own leg (``somaSide`` L -> T1L, R -> T1R) as agonist or
antagonist, or recorded as "logged, not actuated". Both front legs use the same
table; FlyBody's leg axes are mirror-symmetric, and the signs below are
measured separately for each leg anyway.

Channels (FlyBody DOF of leg ``{leg}`` = lf / rf):

  coxa_pitch     c_thorax-{leg}_coxa-pitch               agonist coxa swing, antagonist coxa stance
  trfe_roll      {leg}_coxa-{leg}_trochanterfemur-roll    agonist femur reductor (unopposed)
  trfe_pitch     {leg}_coxa-{leg}_trochanterfemur-pitch   agonist femur/tr flex, antagonist femur/tr extend
  tibia_pitch    {leg}_trochanterfemur-{leg}_tibia-pitch  agonist tibia flex, antagonist tibia extend
  tarsus1_pitch  {leg}_tibia-{leg}_tarsus1-pitch          agonist Ta depressor, antagonist Ta levator

Coxa yaw and coxa roll receive no neural drive and are held at their offset
(design §3.3). Substrate grip (ltm) motor neurons are logged, not actuated:
FlyBody does not actuate the claws.

Scheme (design §3.4), per leg and DOF j, with R the rate vector (Hz):

  u_j = sum(R over agonist MNs of j) - sum(R over antagonist MNs of j)        [Hz]
  command_j = offset_j + sign_j * scale_j * (u_j - c_j) / s_j

clipped to the joint range in position mode. In position mode the command is
a joint-angle target (offset = mu_j, scale = Delta_j, both rad). In torque mode
(robustness arm R1) it is a joint torque (offset = bias, scale = gain). sign_j
is +1 when the agonist's anatomical action increases the FlyBody angle. It is
measured from FlyBody geometry, not chosen. c_j, s_j, offset_j and scale_j are
**not fitted here**: plan step 7 fits them on calibration replicates. Until
then ``BridgeParams.unfitted()`` holds NaN and ``commands`` refuses to run.

The geometric criteria (design §8, step 4) use positions in the thorax frame:

  protraction    claw (tarsus5 origin) moves anteriorly
  CTr flexion    distance thorax-coxa joint -> femur-tibia joint decreases
  FTi flexion    distance coxa-trochanter joint -> tibia-tarsus joint decreases
  TiTa flexion   distance femur-tibia joint -> claw decreases (GUESS: Ta depressor = TiTa flexion)
  femur reduction  claw moves posteriorly (NEEDS SOURCE: reductor = posterior femur rotation)

Build the table (needs FlyGym; about 10 s):
  .venv/bin/python -m organism.mn_joint_bridge
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TABLE_PATH = ROOT / "data" / "paper2" / "stage2B_mn_joint_table.json"
LEGS = ("T1L", "T1R")
BODY_LEG = {"T1L": "lf", "T1R": "rf"}
SIDE_TO_LEG = {"L": "T1L", "R": "T1R"}
N_DOF = 7
# FlyBody actuation order per leg (organism.flybody_tethered.FRONT_DOF_SUFFIXES).
DOF_TEMPLATES = (
    "c_thorax-{leg}_coxa-yaw",
    "c_thorax-{leg}_coxa-roll",
    "c_thorax-{leg}_coxa-pitch",
    "{leg}_coxa-{leg}_trochanterfemur-roll",
    "{leg}_coxa-{leg}_trochanterfemur-pitch",
    "{leg}_trochanterfemur-{leg}_tibia-pitch",
    "{leg}_tibia-{leg}_tarsus1-pitch",
)
HELD_DOF_INDEX = (0, 1)

# channel -> (DOF index in actuation order, agonist spec, antagonist spec, agonist action, criterion)
# A spec is a list of (motor module, MN types or None for "all types of the module").
CHANNELS: dict[str, dict] = {
    "coxa_pitch": {
        "dof": 2,
        "agonist": [("coxa swing", None)],
        "antagonist": [("coxa stance", None)],
        "agonist_action": "protraction",
        "criterion": "claw_anterior",
    },
    "trfe_roll": {
        "dof": 3,
        "agonist": [("femur reductor", None)],
        "antagonist": [],
        "agonist_action": "femur reduction (posterior rotation); NEEDS SOURCE; DOF assignment GUESS",
        "criterion": "claw_posterior",
    },
    "trfe_pitch": {
        "dof": 4,
        "agonist": [("femur/tr flex", None)],
        "antagonist": [("femur/tr extend", None)],
        "agonist_action": "coxa-trochanter flexion",
        "criterion": "thc_to_fti_shorter",
    },
    "tibia_pitch": {
        "dof": 5,
        "agonist": [("tibia flex", None)],
        "antagonist": [("tibia extend", None)],
        "agonist_action": "femur-tibia flexion",
        "criterion": "ctr_to_tita_shorter",
    },
    "tarsus1_pitch": {
        "dof": 6,
        "agonist": [("tarsus control", ("Ta depressor MN",))],
        "antagonist": [("tarsus control", ("Ta levator MN",))],
        "agonist_action": "tarsus depression = tibia-tarsus flexion (GUESS)",
        "criterion": "fti_to_claw_shorter",
    },
}
LOGGED_NOT_ACTUATED = {"substrate grip": "long tendon muscle (claw); FlyBody claws are not actuated"}

# MaleCNS MN type -> FlyMimic muscle(s) (Miller numbering in FlyMimic site names).
FLYMIMIC_MUSCLES: dict[str, tuple[str, ...]] = {
    "Tergopleural/Pleural promotor MN": (
        "LFC_tergopleural_promotor_a",
        "LFC_tergopleural_promotor_b",
        "LFC_pleural_promotor",
    ),
    "Pleural remotor/abductor MN": ("LFC_pleural_remotor_and_abductor",),
    "Sternal anterior rotator MN": ("LFC_sternal_anterior_rotator",),
    "Sternal posterior rotator MN": ("LFC_sternal_posterior_rotator",),
    "Sternal adductor MN": ("LFC_sternal_adductor",),
    "Tr flexor MN": ("LFF_trochanter_flexor_a", "LFF_trochanter_flexor_b"),
    "Acc. tr flexor MN": ("LFF_accesory_trochanter_flexor",),
    "Sternotrochanter MN": ("LFF_sterno-tergo-trochanter_extensor_a", "LFF_sterno-tergo-trochanter_extensor_b"),
    "Tergotr. MN": ("LFF_sterno-tergo-trochanter_extensor_a", "LFF_sterno-tergo-trochanter_extensor_b"),
    "Tr extensor MN": ("LFF_trochanter_extensor",),
    "Ti flexor MN": ("LFTibia_flex_93434",),
    "Acc. ti flexor MN": ("LFTibia_flex_93434",),
    "Ti extensor MN": ("LFTibia_extensor_93932",),
}
# Known disagreement on the pitch axis (design §3.3): the sternal adductor is
# labelled coxa swing but retracts in FlyMimic's geometry.
FLYMIMIC_PITCH_EXCEPTIONS = ("LFC_sternal_adductor",)


def dof_names(body_leg: str) -> list[str]:
    return [t.format(leg=body_leg) for t in DOF_TEMPLATES]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- MN assignment
def assign_motor_neurons(table) -> list[dict]:
    """One row per wTable motor neuron: leg, channel and role, or logged-not-actuated."""
    module = table["motor module"]
    rows = []
    for i in np.flatnonzero(module.notna().to_numpy()):
        r = table.iloc[i]
        mod, typ, side = str(r["motor module"]), str(r["type"]), str(r["somaSide"])
        leg = SIDE_TO_LEG.get(side)
        if leg is None:
            raise ValueError(f"MN row {i} has somaSide {side!r}")
        channel, role = None, None
        for ch, spec in CHANNELS.items():
            for rl in ("agonist", "antagonist"):
                for m, types in spec[rl]:
                    if mod == m and (types is None or typ in types):
                        if channel is not None:
                            raise ValueError(f"MN row {i} matches two channels")
                        channel, role = ch, rl
        if channel is None:
            if mod not in LOGGED_NOT_ACTUATED:
                raise ValueError(f"MN row {i} ({typ}, {mod}) has no channel and is not listed as logged")
            role = "logged_not_actuated"
        rows.append(
            {
                "row": int(i),
                "bodyId": int(r["bodyId"]),
                "type": typ,
                "motor_module": mod,
                "somaSide": side,
                "leg": leg,
                "channel": channel,
                "dof": dof_names(BODY_LEG[leg])[CHANNELS[channel]["dof"]] if channel else None,
                "role": role,
                "flymimic_muscles": list(FLYMIMIC_MUSCLES.get(typ, ())),
                "note": LOGGED_NOT_ACTUATED.get(mod),
            }
        )
    return rows


# ----------------------------------------------------------------------------- FlyBody geometry
def _segment_positions(body) -> dict[str, np.ndarray]:
    """Body-segment origins in the thorax frame (x anterior, y left, z up)."""
    import mujoco

    names = [s.name for s in body.fly.get_bodysegs_order()]
    pos = body.sim.get_body_positions(body.name)
    quat = body.sim.get_body_rotations(body.name)
    i_th = names.index("c_thorax")
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat[i_th], dtype=np.float64))
    R = R.reshape(3, 3)
    return {n: (pos[i] - pos[i_th]) @ R for i, n in enumerate(names)}


def _criterion(p: dict[str, np.ndarray], leg: str, criterion: str) -> float:
    """Value that increases when the agonist's anatomical action happens."""
    g = lambda s: p[f"{leg}_{s}"]  # noqa: E731
    if criterion == "claw_anterior":
        return float(g("tarsus5")[0])
    if criterion == "claw_posterior":
        return -float(g("tarsus5")[0])
    if criterion == "thc_to_fti_shorter":
        return -float(np.linalg.norm(g("tibia") - g("coxa")))
    if criterion == "ctr_to_tita_shorter":
        return -float(np.linalg.norm(g("tarsus1") - g("trochanterfemur")))
    if criterion == "fti_to_claw_shorter":
        return -float(np.linalg.norm(g("tarsus5") - g("tibia")))
    raise ValueError(criterion)


def geometric_signs(body, posture: Mapping[str, np.ndarray], eps: float = 0.02) -> dict[str, dict[str, int]]:
    """sign_j per leg and channel: +1 if increasing the FlyBody angle performs the
    agonist's anatomical action at ``posture`` ({"lf": 7 angles, "rf": 7 angles}).
    Kinematic only (positions after mj_forward; no dynamics)."""

    def place(lf, rf):
        body.reset(
            warmup_s=0.0,
            left_angles=dict(zip(body.dof_names["lf"], lf)),
            right_angles=dict(zip(body.dof_names["rf"], rf)),
        )
        return _segment_positions(body)

    out = {}
    for leg in LEGS:
        bl = BODY_LEG[leg]
        out[leg] = {}
        for ch, spec in CHANNELS.items():
            base = {k: np.asarray(posture[k], dtype=np.float64).copy() for k in ("lf", "rf")}
            c0 = _criterion(place(base["lf"], base["rf"]), bl, spec["criterion"])
            base[bl][spec["dof"]] += eps
            c1 = _criterion(place(base["lf"], base["rf"]), bl, spec["criterion"])
            out[leg][ch] = int(np.sign(c1 - c0))
    return out


def sign_consistency(body, clip: Mapping[str, np.ndarray], signs, every: int = 100) -> dict[str, dict[str, float]]:
    """Fraction of clip frames (every ``every``-th sample) at which each channel's sign holds."""
    frames = range(0, clip["lf"].shape[0], every)
    hits = {leg: {ch: 0 for ch in CHANNELS} for leg in LEGS}
    n = 0
    for k in frames:
        s = geometric_signs(body, {"lf": clip["lf"][k], "rf": clip["rf"][k]})
        n += 1
        for leg in LEGS:
            for ch in CHANNELS:
                hits[leg][ch] += int(s[leg][ch] == signs[leg][ch])
    return {leg: {ch: hits[leg][ch] / n for ch in CHANNELS} for leg in LEGS}


# ----------------------------------------------------------------------------- FlyMimic check
FLYMIMIC_XML = "flygym/assets/model/musculoskeletal/best_combined_arm_damping_stiff_cvt3.xml"
_FM_PITCH_JOINT = {"coxa": "joint_LFCoxa_pitch", "troch": "joint_LFTrochanter_pitch", "tibia": "joint_LFTibia_pitch"}


def flymimic_model():
    """FlyMimic compiled without its (non-bundled) meshes; its inertials are explicit."""
    import mujoco
    import flygym

    path = Path(flygym.__file__).resolve().parents[1] / FLYMIMIC_XML
    xml = path.read_text()
    xml = re.sub(r"<mesh [^>]*/>", "", xml)
    xml = re.sub(r'<geom [^>]*type="mesh"[^>]*/>', "", xml)
    m = mujoco.MjModel.from_xml_string(xml)
    return m, mujoco.MjData(m), str(path)


def flymimic_actions() -> dict[str, dict]:
    """For each FlyMimic muscle at the default keyframe: its active joint torque
    (fully activated, minus the passive-force baseline), and the anatomical
    effect of a small rotation of its joint's pitch DOF (alone) and of the
    joint's full DOF set in the direction of that torque.

    Effects are +1/-1 for: coxa muscles, claw anterior/posterior; trochanter
    muscles, CTr flexion/extension; tibia muscles, FTi flexion/extension."""
    import mujoco

    m, d, _ = flymimic_model()
    jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)  # noqa: E731
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
    joints = {
        "coxa": ["joint_LFCoxa_yaw", "joint_LFCoxa_pitch", "joint_LFCoxa_roll"],
        "troch": ["joint_LFTrochanter_yaw", "joint_LFTrochanter_pitch", "joint_LFTrochanter_roll"],
        "tibia": ["joint_LFTibia_pitch"],
    }

    def reset():
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.ctrl[:] = 0.0
        d.act[:] = 0.0
        mujoco.mj_forward(m, d)

    def crit(kind):
        th = d.xpos[bid("Thorax")]
        R = d.xmat[bid("Thorax")].reshape(3, 3)
        g = lambda b: (d.xpos[bid(b)] - th) @ R  # noqa: E731
        if kind == "coxa":
            return g("LFTarsus5")[0]
        if kind == "troch":
            return -np.linalg.norm(g("LFTibia") - g("LFCoxa"))
        return -np.linalg.norm(g("LFTarsus1") - g("LFTrochanter"))

    def torque(js):
        return np.array([d.qfrc_actuator[m.jnt_dofadr[jid(j)]] for j in js])

    out = {}
    for a in range(m.nu):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        kind = "coxa" if name.startswith("LFC_") else ("troch" if name.startswith("LFF_") else "tibia")
        js = joints[kind]
        reset()
        base = torque(js)
        d.ctrl[a] = 1.0
        d.act[a] = 1.0
        mujoco.mj_forward(m, d)
        tau = torque(js) - base
        reset()
        c0 = crit(kind)
        k = js.index(_FM_PITCH_JOINT[kind])
        d.qpos[m.jnt_qposadr[jid(js[k])]] += 0.01 * np.sign(tau[k])
        mujoco.mj_forward(m, d)
        pitch_effect = int(np.sign(crit(kind) - c0))
        reset()
        unit = tau / np.linalg.norm(tau)
        for j, w in zip(js, unit):
            d.qpos[m.jnt_qposadr[jid(j)]] += 0.01 * w
        mujoco.mj_forward(m, d)
        full_effect = int(np.sign(crit(kind) - c0))
        out[name] = {
            "joint": kind,
            "active_torque": dict(zip(js, np.round(tau, 6).tolist())),
            "pitch_only_effect": pitch_effect,
            "full_torque_effect": full_effect,
        }
    return out


def flymimic_agreement(actions: dict | None = None) -> dict:
    """Compare FlyMimic effects with the module labels of the MN types mapped to each muscle."""
    actions = actions or flymimic_actions()
    expected = {"coxa swing": 1, "coxa stance": -1, "femur/tr flex": 1, "femur/tr extend": -1, "tibia flex": 1, "tibia extend": -1}
    module_of_type = {
        "Tergopleural/Pleural promotor MN": "coxa swing",
        "Sternal anterior rotator MN": "coxa swing",
        "Sternal adductor MN": "coxa swing",
        "Pleural remotor/abductor MN": "coxa stance",
        "Sternal posterior rotator MN": "coxa stance",
        "Tr flexor MN": "femur/tr flex",
        "Acc. tr flexor MN": "femur/tr flex",
        "Sternotrochanter MN": "femur/tr extend",
        "Tergotr. MN": "femur/tr extend",
        "Tr extensor MN": "femur/tr extend",
        "Ti flexor MN": "tibia flex",
        "Acc. ti flexor MN": "tibia flex",
        "Ti extensor MN": "tibia extend",
    }
    rows = {}
    for typ, muscles in FLYMIMIC_MUSCLES.items():
        mod = module_of_type[typ]
        for mu in muscles:
            a = actions[mu]
            rows.setdefault(mu, {"mn_types": [], "module": mod, "expected_effect": expected[mod]})
            rows[mu]["mn_types"].append(typ)
            rows[mu]["pitch_only_effect"] = a["pitch_only_effect"]
            rows[mu]["full_torque_effect"] = a["full_torque_effect"]
            rows[mu]["pitch_only_agrees"] = a["pitch_only_effect"] == expected[mod]
            rows[mu]["full_torque_agrees"] = a["full_torque_effect"] == expected[mod]
    return {
        "effect_convention": "+1 = protraction (coxa) / flexion (trochanter, tibia); -1 = the opposite",
        "muscles": rows,
        "pitch_only_disagreements": sorted(m for m, r in rows.items() if not r["pitch_only_agrees"]),
        "full_torque_disagreements": sorted(m for m, r in rows.items() if not r["full_torque_agrees"]),
        "expected_pitch_exceptions": list(FLYMIMIC_PITCH_EXCEPTIONS),
    }


# ----------------------------------------------------------------------------- table
def operating_posture(clip) -> dict[str, np.ndarray]:
    """Recorded walking mean per front-leg DOF (the posture the signs are read at)."""
    return {k: clip[k].mean(axis=0) for k in ("lf", "rf")}


def build_table(body=None, clip=None) -> dict:
    from flybrain.authors_mcns import WTABLE_PATH, load_authors_wtable
    from organism.flybody_tethered import TetheredFlyBody, load_walking_clip_front_legs

    table = load_authors_wtable()
    if not (table.index.values == np.arange(len(table))).all():
        raise ValueError("wTable index is not the matrix row")
    body = body or TetheredFlyBody("position")
    clip = clip or load_walking_clip_front_legs()
    posture = operating_posture(clip)
    signs = geometric_signs(body, posture)
    consistency = sign_consistency(body, clip, signs)
    mns = assign_motor_neurons(table)
    counts: dict = {}
    for r in mns:
        key = f"{r['leg']}:{r['channel'] or 'logged_not_actuated'}:{r['role']}"
        counts[key] = counts.get(key, 0) + 1
    channels = {}
    for ch, spec in CHANNELS.items():
        channels[ch] = {
            "dof_template": DOF_TEMPLATES[spec["dof"]],
            "dof_index": spec["dof"],
            "agonist": [[m, list(t) if t else "all"] for m, t in spec["agonist"]],
            "antagonist": [[m, list(t) if t else "all"] for m, t in spec["antagonist"]],
            "agonist_action": spec["agonist_action"],
            "criterion": spec["criterion"],
            "agonist_angle_sign": {leg: signs[leg][ch] for leg in LEGS},
            "sign_holds_fraction_of_clip_frames": {leg: consistency[leg][ch] for leg in LEGS},
        }
    return {
        "what": "Stage 2B motor neuron -> FlyBody joint table (design §3.3-3.4, plan step 4)",
        "status": "ENGINEERED; offset/scale/c/s are fitted in plan step 7, not here",
        "scheme": "u_j = sum R(agonists) - sum R(antagonists) [Hz]; command_j = offset_j + agonist_angle_sign_j * scale_j * (u_j - c_j) / s_j; position mode clips to the joint range",
        "wtable": str(WTABLE_PATH.relative_to(ROOT)),
        "wtable_sha256": _sha256(WTABLE_PATH),
        "legs": {leg: {"somaSide": side, "body_leg": BODY_LEG[leg]} for side, leg in SIDE_TO_LEG.items()},
        "dof_order": list(DOF_TEMPLATES),
        "held_dofs": [DOF_TEMPLATES[i] for i in HELD_DOF_INDEX],
        "held_note": "coxa yaw and roll get no neural drive (design §3.3); command = offset",
        "channels": channels,
        "sign_posture": "recorded walking mean per DOF (bundled NeuroMechFly v1 clip, own leg); kinematic probe of +0.02 rad",
        "logged_not_actuated": LOGGED_NOT_ACTUATED,
        "counts": dict(sorted(counts.items())),
        "n_motor_neurons": len(mns),
        "motor_neurons": mns,
        "flymimic_check": flymimic_agreement(),
        "engineered": [
            "module-level agonist/antagonist sums",
            "one DOF per channel; coxa yaw/roll held",
            "Ta depressor = tibia-tarsus flexion (GUESS)",
            "femur reductor -> trochanter-femur roll, posterior rotation (DOF GUESS, action NEEDS SOURCE)",
            "substrate grip logged, not actuated",
        ],
    }


# ----------------------------------------------------------------------------- runtime bridge
@dataclass
class BridgeParams:
    """Per leg (T1L/T1R), 7 values in FlyBody actuation order. Fitted in plan step 7."""

    mode: str  # "position" or "torque"
    offset: dict[str, np.ndarray]
    scale: dict[str, np.ndarray]
    c: dict[str, np.ndarray]
    s: dict[str, np.ndarray]
    lower: dict[str, np.ndarray] = field(default_factory=dict)
    upper: dict[str, np.ndarray] = field(default_factory=dict)
    fitted_by: str = "unfitted"

    @classmethod
    def unfitted(cls, mode: str = "position") -> "BridgeParams":
        nan = {leg: np.full(N_DOF, np.nan) for leg in LEGS}
        return cls(mode, {k: v.copy() for k, v in nan.items()}, {k: v.copy() for k, v in nan.items()},
                   {k: v.copy() for k, v in nan.items()}, {k: v.copy() for k, v in nan.items()})

    def validate(self) -> None:
        if self.mode not in ("position", "torque"):
            raise ValueError(f"mode must be position or torque, not {self.mode!r}")
        for leg in LEGS:
            for name in ("offset", "scale", "c", "s"):
                arr = np.asarray(getattr(self, name)[leg], dtype=np.float64)
                if arr.shape != (N_DOF,):
                    raise ValueError(f"{name}[{leg}] must have shape (7,)")
                if not np.all(np.isfinite(arr)):
                    raise ValueError(f"bridge parameter {name}[{leg}] is not fitted (plan step 7)")
            if np.any(np.asarray(self.scale[leg]) < 0):
                raise ValueError("scale must be >= 0 (direction comes from agonist_angle_sign)")
            if np.any(np.asarray(self.s[leg]) <= 0):
                raise ValueError("s must be > 0")
            if self.mode == "position" and (leg not in self.lower or leg not in self.upper):
                raise ValueError("position mode needs joint ranges (lower/upper)")

    def to_dict(self) -> dict:
        conv = lambda d: {k: np.asarray(v, dtype=float).tolist() for k, v in d.items()}  # noqa: E731
        return {"mode": self.mode, "offset": conv(self.offset), "scale": conv(self.scale), "c": conv(self.c),
                "s": conv(self.s), "lower": conv(self.lower), "upper": conv(self.upper), "fitted_by": self.fitted_by}

    @classmethod
    def from_dict(cls, d: dict) -> "BridgeParams":
        conv = lambda x: {k: np.asarray(v, dtype=np.float64) for k, v in (x or {}).items()}  # noqa: E731
        return cls(d["mode"], conv(d["offset"]), conv(d["scale"]), conv(d["c"]), conv(d["s"]),
                   conv(d.get("lower")), conv(d.get("upper")), d.get("fitted_by", "unknown"))


class MNJointBridge:
    """Rates (n_neurons,) -> per-leg drive u (Hz) and commands (rad or torque)."""

    def __init__(self, table: dict | None = None, n_neurons: int = 4310):
        from scipy.sparse import csr_matrix

        self.table = table if table is not None else json.loads(TABLE_PATH.read_text())
        self.n_neurons = int(n_neurons)
        rows, cols, vals = [], [], []
        for r in self.table["motor_neurons"]:
            if r["role"] not in ("agonist", "antagonist"):
                continue
            li = LEGS.index(r["leg"])
            j = CHANNELS[r["channel"]]["dof"]
            rows.append(li * N_DOF + j)
            cols.append(r["row"])
            vals.append(1.0 if r["role"] == "agonist" else -1.0)
        self.U = csr_matrix((vals, (rows, cols)), shape=(len(LEGS) * N_DOF, self.n_neurons))
        self.sign = {leg: np.zeros(N_DOF) for leg in LEGS}
        for ch, spec in self.table["channels"].items():
            for leg in LEGS:
                self.sign[leg][spec["dof_index"]] = float(spec["agonist_angle_sign"][leg])
        self.dof_names = {leg: dof_names(BODY_LEG[leg]) for leg in LEGS}

    def drive(self, R: np.ndarray) -> dict[str, np.ndarray]:
        """u per leg, 7 values in actuation order (Hz); held DOFs are 0."""
        u = self.U @ np.asarray(R, dtype=np.float64)
        return {leg: u[i * N_DOF:(i + 1) * N_DOF] for i, leg in enumerate(LEGS)}

    def commands(self, R: np.ndarray, params: BridgeParams) -> dict[str, np.ndarray]:
        """Commands per leg for TetheredFlyBody.set_front_leg_targets_array (T1L -> lf, T1R -> rf)."""
        params.validate()
        u = self.drive(R)
        out = {}
        for leg in LEGS:
            cmd = params.offset[leg] + self.sign[leg] * params.scale[leg] * (u[leg] - params.c[leg]) / params.s[leg]
            if params.mode == "position":
                cmd = np.clip(cmd, params.lower[leg], params.upper[leg])
            out[leg] = cmd
        return out


def main(argv: list[str] | None = None) -> int:
    import argparse
    import subprocess
    from datetime import datetime, timezone

    p = argparse.ArgumentParser(description="Build data/paper2/stage2B_mn_joint_table.json")
    p.add_argument("--out", type=Path, default=TABLE_PATH)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    if a.out.exists() and not a.overwrite:
        raise FileExistsError(f"Refusing to overwrite {a.out}; pass --overwrite")
    t = build_table()
    t["generated_utc"] = datetime.now(timezone.utc).isoformat()
    try:
        t["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
    except Exception:
        t["git_commit"] = None
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(t, indent=1) + "\n")
    print(json.dumps({k: t[k] for k in ("counts", "n_motor_neurons")}, indent=1))
    for ch, spec in t["channels"].items():
        print(ch, spec["agonist_angle_sign"], spec["sign_holds_fraction_of_clip_frames"])
    print("FlyMimic pitch disagreements:", t["flymimic_check"]["pitch_only_disagreements"],
          "full-torque disagreements:", t["flymimic_check"]["full_torque_disagreements"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
