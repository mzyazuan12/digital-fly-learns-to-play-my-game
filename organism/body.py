"""Physical body. NeuroMechFly when FlyGym is present; otherwise a mock.

The mock is a unicycle stand-in so the organism loop can be tested without
MuJoCo. It is not an anatomically detailed fly. Do not treat mock trajectories
as NeuroMechFly results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from organism.sensory import SensoryObservation, geometric_eyes


@dataclass
class Pose:
    x_mm: float = 0.0
    y_mm: float = 0.0
    z_mm: float = 0.5
    heading_rad: float = 0.0


class Body(Protocol):
    name: str
    kind: str
    pose: Pose

    def sense(self, world) -> SensoryObservation: ...

    def apply_descending(self, command: np.ndarray) -> None: ...

    def step_physics(self) -> None: ...

    def teleport(self, pose: Pose) -> None: ...

    def snapshot(self) -> dict: ...


@dataclass
class MockBody:
    """Planar walk/turn body used when FlyGym is not imported.

    Modeling assumption (`mock_unicycle`): tank-steer from a 2-vector command.
    Left-high command turns right, matching FlyGym HybridTurningController.
    """

    name: str = "mock_fly"
    kind: str = "mock_unicycle"
    pose: Pose = field(default_factory=Pose)
    dt_s: float = 0.001
    walk_gain: float = 18.0  # mm/s at command 1; adult Drosophila walking ~10–25 mm/s
    turn_gain: float = 2.2  # rad/s at command difference 1
    last_command: np.ndarray = field(default_factory=lambda: np.zeros(2))
    last_mode: str = "rest"
    gait_phase: float = 0.0
    world: object | None = None
    _prev_heading: float = 0.0

    def sense(self, world) -> SensoryObservation:
        lights = world.light_xyi()
        left, right = geometric_eyes(
            self.pose.x_mm, self.pose.y_mm, self.pose.heading_rad, lights
        )
        odors = world.odor_at(self.pose.x_mm, self.pose.y_mm, self.pose.z_mm)
        bump = world.contact_at(self.pose.x_mm, self.pose.y_mm)
        walking = self.last_mode == "walk" or float(np.mean(self.last_command)) > 0.05
        if walking:
            self.gait_phase += 12.0 * self.dt_s * 2 * np.pi
        legs = (0.5 + 0.5 * np.sin(self.gait_phase + np.linspace(0, np.pi, 6))) * (1.0 if walking else 0.15)
        joints = np.sin(self.gait_phase + np.linspace(0, 2 * np.pi, 12)).astype(np.float32)
        if not walking:
            joints *= 0.05
        ang_vel = (self.pose.heading_rad - self._prev_heading) / max(self.dt_s, 1e-6)
        self._prev_heading = self.pose.heading_rad
        contact = legs.astype(np.float32)
        contact[0] = max(float(contact[0]), bump)
        return SensoryObservation(
            left_eye=left,
            right_eye=right,
            joint_angles=0.5 + 0.5 * joints,
            joint_velocities=np.gradient(joints).astype(np.float32),
            contact=contact,
            antenna_left=_clip01(odors * (1.0 if left >= right else 0.4)),
            antenna_right=_clip01(odors * (1.0 if right > left else 0.4)),
            odor=_clip01(odors),
            taste=0.0,
            angular_velocity=float(ang_vel),
            heading_rad=self.pose.heading_rad,
        )

    def apply_descending(self, command) -> None:
        if hasattr(command, "left"):
            self.last_command = np.array([command.left, command.right], dtype=np.float64)
            self.last_mode = command.mode
        else:
            self.last_command = np.asarray(command, dtype=np.float64).reshape(2)
            self.last_mode = "walk" if float(np.mean(self.last_command)) > 0.05 else "rest"

    def step_physics(self) -> None:
        # Unicycle integration. Forbidden as walking once NeuroMechFly/MuJoCo is
        # attached; this path exists only so CI can run without FlyGym.
        if getattr(self, "embodied", False):
            return
        left, right = self.last_command
        if self.last_mode in {"rest", "groom", "fly"} or (abs(left) + abs(right)) < 0.04:
            return
        speed = self.walk_gain * 0.5 * (left + right)
        if self.last_mode == "reverse":
            speed = -abs(speed)
        yaw_rate = self.turn_gain * (right - left)
        heading = self.pose.heading_rad + yaw_rate * self.dt_s
        x = self.pose.x_mm + speed * self.dt_s * float(np.cos(heading))
        y = self.pose.y_mm + speed * self.dt_s * float(np.sin(heading))
        if self.world is not None:
            x, y, heading = self.world.clip_pose(x, y, heading)
        self.pose.heading_rad = heading
        self.pose.x_mm = x
        self.pose.y_mm = y

    def teleport(self, pose: Pose) -> None:
        self.pose = Pose(pose.x_mm, pose.y_mm, pose.z_mm, pose.heading_rad)

    def snapshot(self) -> dict:
        return {
            "kind": self.kind,
            "name": self.name,
            "x_mm": self.pose.x_mm,
            "y_mm": self.pose.y_mm,
            "z_mm": self.pose.z_mm,
            "heading_rad": self.pose.heading_rad,
            "command": self.last_command.tolist(),
        }


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def flygym_available() -> bool:
    try:
        import flygym  # noqa: F401

        return True
    except Exception:
        return False


def build_body(*, physics: bool, name: str = "virtual_fly", spawn: Pose | None = None) -> Body:
    spawn = spawn or Pose()
    if physics:
        from organism.physics import build_flygym_body

        try:
            return build_flygym_body(name=name, spawn=spawn)
        except Exception as exc:
            raise RuntimeError(
                "physics=True requires the official FlyGym 2.1 NeuroMechFly + MuJoCo. "
                "A decorative GLB/OBJ is not a body. Install with: "
                "pip install 'flygym @ git+https://github.com/NeLy-EPFL/flygym.git@v2.1.0'"
            ) from exc
    return MockBody(name=name, pose=spawn)
