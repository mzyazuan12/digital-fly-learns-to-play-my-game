"""NeuroMechFly / MuJoCo adapter for the organism loop.

The canonical body is `organism.gait.ArticulatedFly`: official FlyGym 2.1
NeuroMechFly (orange micro-CT, compound eyes), HybridTurningController as
an engineered VNC/muscle surrogate. This module only wraps that model for
VirtualFly. It does not walk by writing the free-root pose.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from organism.body import Pose
from organism.gait import ArticulatedFly, Proprioception, RootMotionForbidden
from organism.sensory import SensoryObservation, geometric_eyes


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _eye_brightness(eye: np.ndarray) -> float:
    """Mean of the active yellow/pale channel per ommatidium."""
    arr = np.asarray(eye, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    if arr.ndim >= 2 and arr.shape[-1] == 2:
        mag = np.maximum(arr[..., 0], arr[..., 1])
    else:
        mag = arr
    return _clip01(float(np.mean(mag)))


@dataclass
class FlyGymBody:
    """Organism-facing wrapper around the articulated NeuroMechFly."""

    inner: ArticulatedFly
    last_command: np.ndarray = field(default_factory=lambda: np.zeros(2))
    last_mode: str = "rest"
    world: object | None = None
    _prev_heading: float = 0.0

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def kind(self) -> str:
        return self.inner.kind

    @property
    def fly(self):
        return self.inner.fly

    @property
    def sim(self):
        return self.inner.sim

    @property
    def controller(self):
        return self.inner.controller

    @property
    def dt_s(self) -> float:
        return float(self.inner.sim.timestep)

    @dt_s.setter
    def dt_s(self, _value: float) -> None:
        # Physics timestep is owned by MuJoCo. The organism loop may assign
        # a brain-window duration here; ignore it rather than retiming the body.
        return

    @property
    def pose(self) -> Pose:
        xyz = self.inner.thorax_xyz()
        return Pose(
            x_mm=float(xyz[0]),
            y_mm=float(xyz[1]),
            z_mm=float(xyz[2]),
            heading_rad=self.inner.heading_rad(),
        )

    @pose.setter
    def pose(self, value: Pose) -> None:
        raise RootMotionForbidden(
            "Cannot assign thorax pose in embodied mode. Spawn is set when the "
            "NeuroMechFly is attached to the world; walking is physics."
        )

    def sense(self, world) -> SensoryObservation:
        pose = self.pose
        prop = self.inner.proprioception()
        contact = np.clip(np.asarray(prop.contact_found, dtype=np.float32), 0.0, 1.0)
        if contact.size < 6:
            padded = np.zeros(6, dtype=np.float32)
            padded[: contact.size] = contact
            contact = padded
        odors = world.odor_at(pose.x_mm, pose.y_mm, pose.z_mm)
        ang_vel = float(np.linalg.norm(prop.angular_velocity))
        self._prev_heading = pose.heading_rad
        ommatidia = None
        left = 0.0
        right = 0.0
        if getattr(self.inner, "_vision_ok", True):
            try:
                ommatidia = np.asarray(
                    self.inner.sim.get_ommatidia_readouts(self.inner.name),
                    dtype=np.float32,
                )
                left = _eye_brightness(ommatidia[0])
                right = _eye_brightness(ommatidia[1])
                self.inner._vision_ok = True
            except Exception:
                self.inner._vision_ok = False
                ommatidia = None
        if ommatidia is None:
            lights = world.light_xyi()
            left, right = geometric_eyes(pose.x_mm, pose.y_mm, pose.heading_rad, lights)
        return SensoryObservation(
            left_eye=left,
            right_eye=right,
            ommatidia=ommatidia,
            joint_angles=prop.joint_angles.astype(np.float32),
            joint_velocities=prop.joint_velocities.astype(np.float32),
            contact=contact[:6],
            antenna_left=_clip01(odors),
            antenna_right=_clip01(odors),
            odor=_clip01(odors),
            taste=0.0,
            angular_velocity=ang_vel,
            heading_rad=pose.heading_rad,
        )

    def proprioception(self) -> Proprioception:
        return self.inner.proprioception()

    def apply_descending(self, command) -> None:
        if hasattr(command, "left"):
            self.last_command = np.array([command.left, command.right], dtype=np.float64)
            self.last_mode = command.mode
            self.inner.last_groom_hz = float(getattr(command, "groom_hz", 0.0) or 0.0)
            self.inner.last_flight_hz = float(getattr(command, "flight_hz", 0.0) or 0.0)
        else:
            self.last_command = np.asarray(command, dtype=np.float64).reshape(2)
            self.last_mode = (
                "walk" if float(np.mean(self.last_command)) > 0.05 else "rest"
            )
        if self.last_mode == "groom":
            self.inner.set_command("groom")
        elif self.last_mode == "fly":
            self.inner.set_command("fly")
        elif self.last_mode == "rest":
            self.inner.set_command("stand")
        else:
            self.inner.set_command(self.last_command)

    def step_physics(self) -> None:
        self.inner.step()

    def teleport(self, pose: Pose) -> None:
        raise RootMotionForbidden(
            f"Refusing to teleport NeuroMechFly to ({pose.x_mm}, {pose.y_mm}). "
            "Root-transform locomotion is disabled."
        )

    def snapshot(self) -> dict:
        pose = self.pose
        snap = self.inner.snapshot()
        return {
            "kind": self.kind,
            "name": self.name,
            "x_mm": pose.x_mm,
            "y_mm": pose.y_mm,
            "z_mm": pose.z_mm,
            "heading_rad": pose.heading_rad,
            "command": self.last_command.tolist(),
            "joint_std": float(np.std(snap.joint_angles)),
            "contact": snap.contact_found.tolist(),
            "root_written": snap.root_written,
        }


def build_flygym_body(name: str = "virtual_fly", spawn: Pose | None = None) -> FlyGymBody:
    spawn = spawn or Pose(z_mm=0.6)
    inner = ArticulatedFly(
        name=name,
        spawn_xyz_mm=(spawn.x_mm, spawn.y_mm, spawn.z_mm),
        spawn_yaw_rad=spawn.heading_rad,
        render=False,
    )
    return FlyGymBody(inner=inner, _prev_heading=spawn.heading_rad)
