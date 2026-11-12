"""Tethered FlyBody for the stage 2B closed loop (design §3.2, §6.1; amendment A1.2).

FlyGym 2.1 ``FlyBody`` (Vaxenburg et al. 2025) in ``TetheredWorld``: the
thorax is a mocap body (no free joint, no drift), there is no ground, and leg
adhesion is not added. Joints: ``FlyBodyJointPreset.LEGS_ONLY``. Actuated DOFs:
``FlyBodyActuatedDOFPreset.LEGS_ACTIVE_ONLY`` (7 per leg). Joint stiffness,
damping, ranges and position-actuator gains are FlyBody's own config values,
not tuned here.

- The two front legs (lf, rf) take commands from the stage 2B bridge. With
  ``front_actuation="position"`` (primary) a command is a joint-angle target
  for FlyBody's position servo (kp 80 coxa / trochanter-femur, 40 tibia /
  tarsus1). With ``"torque"`` (robustness arm R1) the front-leg actuators are
  MuJoCo ``motor`` actuators (gear 1, no control or force limit) and a command
  is a joint torque in model units.
- The other four legs are held at FlyBody's neutral pose by their position
  servos. FlyBody's neutral pose file sets only wing angles, so the neutral leg
  pose is 0 rad on every DOF, which is also every leg joint's spring
  reference.

Axis order is ``YAW_ROLL_PITCH``: the order of FlyBody's original leg joint
chain (coxa_abduct, coxa_twist, coxa; femur_twist, femur) and the only order
for which FlyGym ships a FlyBody neutral pose.

ENGINEERED: tethering with legs in air, holding the non-front legs, and the
choice between position-servo and torque actuation for the front legs.
"""

from __future__ import annotations

import warnings
from typing import Mapping

import numpy as np

FRONT_LEGS: tuple[str, str] = ("lf", "rf")
HELD_LEGS: tuple[str, ...] = ("lm", "lh", "rm", "rh")
TIMESTEP_S = 1e-4
DEFAULT_WARMUP_S = 0.1
FRONT_ACTUATIONS = ("position", "torque")

# Expected actuation order of each front leg's DOFs (stage 2B design §3.2).
FRONT_DOF_SUFFIXES: tuple[str, ...] = (
    "c_thorax-{leg}_coxa-yaw",
    "c_thorax-{leg}_coxa-roll",
    "c_thorax-{leg}_coxa-pitch",
    "{leg}_coxa-{leg}_trochanterfemur-roll",
    "{leg}_coxa-{leg}_trochanterfemur-pitch",
    "{leg}_trochanterfemur-{leg}_tibia-pitch",
    "{leg}_tibia-{leg}_tarsus1-pitch",
)


def expected_front_dof_names(leg: str) -> list[str]:
    return [s.format(leg=leg) for s in FRONT_DOF_SUFFIXES]


class TetheredFlyBody:
    """Tethered FlyBody with bridge-driven front legs and servo-held other legs.

    The co-simulation calls, once per 0.1 ms tick:
    ``set_front_leg_targets`` (or ``set_front_leg_targets_array``), ``step``,
    ``get_state`` (or ``front_leg_arrays``).
    """

    def __init__(
        self,
        front_actuation: str = "position",
        *,
        name: str = "fb2b",
        spawn_height_mm: float = 2.0,
        warmup_s: float = DEFAULT_WARMUP_S,
    ) -> None:
        if front_actuation not in FRONT_ACTUATIONS:
            raise ValueError(f"front_actuation must be one of {FRONT_ACTUATIONS}")
        from flygym import Simulation
        from flygym.compose import ActuatorType, FlyBody, KinematicPosePreset, TetheredWorld
        from flygym.flybody.anatomy_flybody import (
            FlyBodyActuatedDOFPreset,
            FlyBodyAxisOrder,
            FlyBodyJointPreset,
            FlyBodySkeleton,
        )
        from flygym.utils.math import Rotation3D

        self.front_actuation = front_actuation
        self.name = name
        self.warmup_s = float(warmup_s)
        self._AT = ActuatorType
        self.front_type = ActuatorType.POSITION if front_actuation == "position" else ActuatorType.MOTOR

        axis_order = FlyBodyAxisOrder.YAW_ROLL_PITCH
        pose = KinematicPosePreset.FLYBODY_NEUTRAL.get_pose_by_axis_order(axis_order)
        fly = FlyBody(name=name)
        skeleton = FlyBodySkeleton(axis_order=axis_order, joint_preset=FlyBodyJointPreset.LEGS_ONLY)
        with warnings.catch_warnings():
            # FlyGym warns that kp is taken from FlyBody's actuator config and that
            # motor actuators are unlimited; both are intended here.
            warnings.simplefilter("ignore")
            fly.add_joints(skeleton, neutral_pose=pose)
            actuated = skeleton.get_actuated_dofs_from_preset(FlyBodyActuatedDOFPreset.LEGS_ACTIVE_ONLY)
            front = [d for d in actuated if d.child.pos in FRONT_LEGS]
            held = [d for d in actuated if d.child.pos not in FRONT_LEGS]
            if front_actuation == "position":
                fly.add_actuators(actuated, ActuatorType.POSITION, neutral_input=pose)
            else:
                fly.add_actuators(held, ActuatorType.POSITION, neutral_input=pose)
                fly.add_actuators(front, ActuatorType.MOTOR)
            world = TetheredWorld()
            world.add_fly(fly, [0.0, 0.0, float(spawn_height_mm)], Rotation3D("quat", [1.0, 0.0, 0.0, 0.0]))
            sim = Simulation(world)
        if abs(float(sim.timestep) - TIMESTEP_S) > 1e-12:
            raise RuntimeError(f"FlyBody timestep is {sim.timestep}, expected {TIMESTEP_S}")
        self.fly, self.world, self.sim = fly, world, sim
        self.timestep = float(sim.timestep)

        joint_names = [d.name for d in fly.get_jointdofs_order()]
        self._jidx = {n: i for i, n in enumerate(joint_names)}
        self.joint_names = joint_names

        pos_order = [d.name for d in fly.get_actuated_jointdofs_order(ActuatorType.POSITION)]
        front_order = [d.name for d in fly.get_actuated_jointdofs_order(self.front_type)]
        self.dof_names: dict[str, list[str]] = {
            leg: [n for n in front_order if f"-{leg}_" in n] for leg in FRONT_LEGS
        }
        for leg in FRONT_LEGS:
            if self.dof_names[leg] != expected_front_dof_names(leg):
                raise RuntimeError(f"{leg} DOFs {self.dof_names[leg]} differ from design §3.2")
        self.held_dof_names = [n for n in pos_order if not any(f"-{leg}_" in n for leg in FRONT_LEGS)]
        if len(self.held_dof_names) != 7 * len(HELD_LEGS):
            raise RuntimeError(f"expected 28 held DOFs, got {len(self.held_dof_names)}")

        # Index maps: front-leg DOF k of leg -> slot in the front actuator array,
        # in the position actuator array (held legs), and in the joint arrays.
        self._front_slot = {leg: np.array([front_order.index(n) for n in self.dof_names[leg]]) for leg in FRONT_LEGS}
        self._held_slot = np.array([pos_order.index(n) for n in self.held_dof_names])
        self._front_jidx = {leg: np.array([self._jidx[n] for n in self.dof_names[leg]]) for leg in FRONT_LEGS}
        self._held_jidx = np.array([self._jidx[n] for n in self.held_dof_names])
        self._n_pos = len(pos_order)
        self._n_front = len(front_order)
        self._name_slot = {leg: {n: k for k, n in enumerate(self.dof_names[leg])} for leg in FRONT_LEGS}
        import mujoco

        self._qposadr = {}
        for leg in FRONT_LEGS:
            for n in self.dof_names[leg]:
                jid = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}/{n}")
                if jid < 0:
                    raise RuntimeError(f"joint {name}/{n} not found in the compiled model")
                self._qposadr[n] = int(sim.mj_model.jnt_qposadr[jid])

        # Neutral commands: held legs at FlyBody's neutral pose (all leg DOFs 0).
        lookup = pose.joint_angles_lookup_rad
        self.neutral_angle = {n: float(lookup.get(n, 0.0)) for n in pos_order + front_order}
        self._held_cmd = np.array([self.neutral_angle[n] for n in self.held_dof_names])
        self._front_neutral = {
            leg: (np.array([self.neutral_angle.get(n, 0.0) for n in self.dof_names[leg]])
                  if front_actuation == "position" else np.zeros(7))
            for leg in FRONT_LEGS
        }
        self._front_cmd = {leg: self._front_neutral[leg].copy() for leg in FRONT_LEGS}
        self.steps = 0
        self.reset()

    # ------------------------------------------------------------------ control
    def _push(self) -> None:
        AT = self._AT
        if self.front_actuation == "position":
            ctrl = np.empty(self._n_pos)
            ctrl[self._held_slot] = self._held_cmd
            for leg in FRONT_LEGS:
                ctrl[self._front_slot[leg]] = self._front_cmd[leg]
            self.sim.set_actuator_inputs(self.name, AT.POSITION, ctrl)
        else:
            held = np.empty(self._n_pos)
            held[self._held_slot] = self._held_cmd
            self.sim.set_actuator_inputs(self.name, AT.POSITION, held)
            front = np.empty(self._n_front)
            for leg in FRONT_LEGS:
                front[self._front_slot[leg]] = self._front_cmd[leg]
            self.sim.set_actuator_inputs(self.name, AT.MOTOR, front)

    def set_front_leg_targets(
        self,
        left: Mapping[str, float] | None = None,
        right: Mapping[str, float] | None = None,
    ) -> None:
        """Set commands by DOF name. DOFs not named keep their previous command.

        Position mode: joint-angle targets (rad; MuJoCo clips them to the joint
        range). Torque mode: joint torques (model units).
        """
        for leg, cmd in (("lf", left), ("rf", right)):
            if not cmd:
                continue
            slots = self._name_slot[leg]
            for name, value in cmd.items():
                if name not in slots:
                    raise KeyError(f"{name!r} is not an actuated {leg} DOF; expected one of {self.dof_names[leg]}")
                self._front_cmd[leg][slots[name]] = float(value)
        self._push()

    def set_front_leg_targets_array(self, left: np.ndarray, right: np.ndarray) -> None:
        """Set all 7 commands per front leg, in ``dof_names[leg]`` order."""
        for leg, cmd in (("lf", left), ("rf", right)):
            cmd = np.asarray(cmd, dtype=np.float64)
            if cmd.shape != (7,):
                raise ValueError(f"{leg} command must have shape (7,), got {cmd.shape}")
            self._front_cmd[leg][:] = cmd
        self._push()

    def step(self, n: int = 1) -> None:
        for _ in range(int(n)):
            self.sim.step()
        self.steps += int(n)

    @property
    def t(self) -> float:
        """Time since the end of the last reset's warm-up (s), as steps * timestep."""
        return self.steps * self.timestep

    # ------------------------------------------------------------------ state
    def front_leg_arrays(self) -> dict[str, dict[str, np.ndarray]]:
        """Per front leg: angle, velocity, actuator_force arrays in ``dof_names[leg]`` order (copies)."""
        q = self.sim.get_joint_angles(self.name)
        v = self.sim.get_joint_velocities(self.name)
        f = self.sim.get_actuator_forces(self.name, self.front_type)
        return {
            leg: {
                "angle": q[self._front_jidx[leg]].copy(),
                "velocity": v[self._front_jidx[leg]].copy(),
                "actuator_force": f[self._front_slot[leg]].copy(),
            }
            for leg in FRONT_LEGS
        }

    def get_state(self) -> dict:
        """Front-leg state keyed by DOF name, plus the DOF name lists and time."""
        arrays = self.front_leg_arrays()
        out = {"t": self.t, "dof_names": {leg: list(self.dof_names[leg]) for leg in FRONT_LEGS}}
        for leg in FRONT_LEGS:
            names = self.dof_names[leg]
            out[leg] = {key: dict(zip(names, arrays[leg][key].tolist())) for key in arrays[leg]}
        return out

    def held_leg_angles(self) -> np.ndarray:
        return self.sim.get_joint_angles(self.name)[self._held_jidx].copy()

    def held_leg_targets(self) -> np.ndarray:
        return self._held_cmd.copy()

    # ------------------------------------------------------------------ reset
    def reset(
        self,
        left: Mapping[str, float] | None = None,
        right: Mapping[str, float] | None = None,
        warmup_s: float | None = None,
        *,
        left_angles: Mapping[str, float] | None = None,
        right_angles: Mapping[str, float] | None = None,
    ) -> None:
        """Neutral keyframe, held legs at neutral, front legs at neutral or the given commands,
        then ``warmup_s`` of physics. Deterministic: the same calls give bit-identical runs.

        ``left_angles`` / ``right_angles`` optionally place front-leg joints at given
        angles (zero velocity) before the warm-up. Useful in torque mode, where the
        passive time constant (damping / stiffness) is about 1 s.
        """
        import mujoco

        self.sim.reset()
        for leg, angles in (("lf", left_angles), ("rf", right_angles)):
            for name, value in (angles or {}).items():
                if name not in self._name_slot[leg]:
                    raise KeyError(f"{name!r} is not an actuated {leg} DOF")
                self.sim.mj_data.qpos[self._qposadr[name]] = float(value)
        if left_angles or right_angles:
            mujoco.mj_forward(self.sim.mj_model, self.sim.mj_data)
        for leg in FRONT_LEGS:
            self._front_cmd[leg] = self._front_neutral[leg].copy()
        self.set_front_leg_targets(left, right)
        self._push()
        n = int(round((self.warmup_s if warmup_s is None else float(warmup_s)) / self.timestep))
        for _ in range(n):
            self.sim.step()
        self.steps = 0

    # ------------------------------------------------------------------ provenance
    def describe(self) -> dict:
        """Compiled-model facts for the front legs (for provenance files)."""
        import mujoco

        m = self.sim.mj_model
        out = {
            "body_model": "FlyGym FlyBody (Vaxenburg et al. 2025)",
            "world": "TetheredWorld (thorax mocap), no ground, no adhesion",
            "axis_order": "yaw_roll_pitch",
            "timestep_s": self.timestep,
            "gravity_mm_s2": m.opt.gravity.tolist(),
            "front_actuation": self.front_actuation,
            "warmup_s": self.warmup_s,
            "held_legs": list(HELD_LEGS),
            "held_pose": "FlyBody neutral (0 rad on every leg DOF)",
            "dofs": {},
        }
        for leg in FRONT_LEGS:
            for name in self.dof_names[leg]:
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{self.name}/{name}")
                aid = mujoco.mj_name2id(
                    m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{self.name}/{name}-{self.front_type.value}"
                )
                out["dofs"][name] = {
                    "range_rad": m.jnt_range[jid].tolist(),
                    "limited": bool(m.jnt_limited[jid]),
                    "stiffness": float(m.jnt_stiffness[jid]),
                    "springref_rad": float(m.qpos_spring[m.jnt_qposadr[jid]]),
                    "damping": float(m.dof_damping[m.jnt_dofadr[jid]]),
                    "armature": float(m.dof_armature[m.jnt_dofadr[jid]]),
                    "actuator_gain": float(m.actuator_gainprm[aid, 0]),
                    "actuator_bias": m.actuator_biasprm[aid, :3].tolist(),
                }
        return out


def load_walking_clip_front_legs(timestep: float = TIMESTEP_S) -> dict:
    """Recorded walking angles for the front legs, resampled to ``timestep``.

    Source: ``flygym_demo/ball_flybody_data/assets/ball_flybody_clip.npz``
    (NeuroMechFly v1 tethered ball walking, 1 s at 100 fps, joint angles from
    SeqIKPy-based inverse kinematics onto FlyBody). Conversion is exactly
    FlyGym's own FlyBody replay (``flybody_step_extraction.replay_clip``):
    ``MotionSnippet(angles_global2anatomical=False)``, Savitzky-Golay window
    0.05 s, cubic interpolation, DOFs matched by (parent link, child link,
    axis). No sign flips: FlyBody's leg axes are mirror-symmetric, and FlyGym
    feeds this clip to FlyBody as is.
    """
    from importlib.resources import files

    from flygym_demo.spotlight_data import MotionSnippet

    clip = files("flygym_demo.ball_flybody_data") / "assets/ball_flybody_clip.npz"
    body = _clip_dof_order()
    snippet = MotionSnippet(clip, angles_global2anatomical=False)
    angles = snippet.get_joint_angles(output_timestep=timestep, output_dof_order=body, sgfilter_window_sec=0.05)
    out = {"timestep": timestep, "source": str(clip), "data_fps": float(snippet.data_fps)}
    for leg in FRONT_LEGS:
        names = expected_front_dof_names(leg)
        cols = [i for i, d in enumerate(body) if d.name in names]
        order = [body[i].name for i in cols]
        out[leg] = angles[:, [cols[order.index(n)] for n in names]]
    return out


_CLIP_ORDER_CACHE: list | None = None


def _clip_dof_order() -> list:
    """FlyBody LEGS_ACTIVE_ONLY JointDOF objects (yaw_roll_pitch), without building a model."""
    global _CLIP_ORDER_CACHE
    if _CLIP_ORDER_CACHE is None:
        from flygym.flybody.anatomy_flybody import (
            FlyBodyActuatedDOFPreset,
            FlyBodyAxisOrder,
            FlyBodyJointPreset,
            FlyBodySkeleton,
        )

        sk = FlyBodySkeleton(axis_order=FlyBodyAxisOrder.YAW_ROLL_PITCH, joint_preset=FlyBodyJointPreset.LEGS_ONLY)
        _CLIP_ORDER_CACHE = list(sk.get_actuated_dofs_from_preset(FlyBodyActuatedDOFPreset.LEGS_ACTIVE_ONLY))
    return _CLIP_ORDER_CACHE
