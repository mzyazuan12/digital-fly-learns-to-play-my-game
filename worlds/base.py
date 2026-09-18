"""Worlds are places. They are not brains and they do not pay the fly.

Autonomous mode has no reward, no target, no task. Experiments attach a
Task object separately; they do not live inside the sensorimotor loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Light:
    x_mm: float
    y_mm: float
    z_mm: float = 2.0
    intensity: float = 40.0


@dataclass
class Odor:
    x_mm: float
    y_mm: float
    z_mm: float = 1.0
    intensity: float = 1.0
    sigma_mm: float = 400.0


@dataclass
class Solid:
    """Axis-aligned collision proxy. Visual mesh may be richer; this is physics."""

    name: str
    x0: float
    y0: float
    x1: float
    y1: float
    z0: float = 0.0
    z1: float = 800.0
    walkable: bool = False


@dataclass
class Keycap:
    """A physical key the fly can stand on. Not a brain label."""

    label: str
    x_mm: float
    y_mm: float
    z_mm: float = 2.0
    half_mm: float = 8.5


@dataclass
class World:
    name: str
    width_mm: float = 200.0
    depth_mm: float = 200.0
    height_mm: float = 100.0
    lights: list[Light] = field(default_factory=list)
    odors: list[Odor] = field(default_factory=list)
    solids: list[Solid] = field(default_factory=list)
    keys: list[Keycap] = field(default_factory=list)
    monitor: dict | None = None
    monitor_on: bool = False
    notes: str = ""

    def light_xyi(self) -> list[tuple[float, float, float]]:
        return [(lamp.x_mm, lamp.y_mm, lamp.intensity) for lamp in self.lights]

    def odor_at(self, x_mm: float, y_mm: float, z_mm: float = 0.5) -> float:
        total = 0.0
        for odor in self.odors:
            dx = x_mm - odor.x_mm
            dy = y_mm - odor.y_mm
            dz = z_mm - odor.z_mm
            sig2 = odor.sigma_mm * odor.sigma_mm
            total += odor.intensity * float(np.exp(-(dx * dx + dy * dy + dz * dz) / (2.0 * sig2)))
        return float(min(1.0, total))

    def contact_at(self, x_mm: float, y_mm: float, radius_mm: float = 3.0) -> float:
        best = 0.0
        for solid in self.solids:
            dx = max(solid.x0 - x_mm, 0.0, x_mm - solid.x1)
            dy = max(solid.y0 - y_mm, 0.0, y_mm - solid.y1)
            dist = float(np.hypot(dx, dy))
            if dist < radius_mm:
                best = max(best, 1.0 - dist / radius_mm)
        return best

    def clip_pose(self, x_mm: float, y_mm: float, heading_rad: float, radius_mm: float = 2.0):
        x = min(max(x_mm, radius_mm), self.width_mm - radius_mm)
        y = min(max(y_mm, radius_mm), self.depth_mm - radius_mm)
        for solid in self.solids:
            if solid.walkable:
                continue
            if solid.x0 - radius_mm < x < solid.x1 + radius_mm and solid.y0 - radius_mm < y < solid.y1 + radius_mm:
                # Push to nearest face. This is collision physics, not a brain reflex.
                left = abs(x - (solid.x0 - radius_mm))
                right = abs(x - (solid.x1 + radius_mm))
                down = abs(y - (solid.y0 - radius_mm))
                up = abs(y - (solid.y1 + radius_mm))
                nearest = min(left, right, down, up)
                if nearest == left:
                    x = solid.x0 - radius_mm
                elif nearest == right:
                    x = solid.x1 + radius_mm
                elif nearest == down:
                    y = solid.y0 - radius_mm
                else:
                    y = solid.y1 + radius_mm
        return x, y, heading_rad

    def step(self) -> None:
        """Worlds may animate. Default: static."""

    def describe(self) -> dict:
        return {
            "name": self.name,
            "size_mm": [self.width_mm, self.depth_mm, self.height_mm],
            "n_lights": len(self.lights),
            "n_odors": len(self.odors),
            "n_solids": len(self.solids),
            "n_keys": len(self.keys),
            "reward": None,
            "notes": self.notes,
        }

    def present_monitor(self) -> None:
        """Turn the desk/rug monitor into a bright object. Not a navigate_to API."""
        if self.monitor_on:
            return
        self.monitor_on = True
        spec = self.monitor or {}
        self.lights.append(
            Light(
                x_mm=float(spec.get("x_mm", 3750.0)),
                y_mm=float(spec.get("y_mm", 200.0)),
                z_mm=float(spec.get("z_mm", 1280.0)),
                intensity=float(spec.get("intensity", 220.0)),
            )
        )


@dataclass
class Task:
    """Optional experimental overlay. Never imported by the fly brain loop."""

    name: str
    reward: float | None = None
    notes: str = ""


def empty_arena() -> World:
    return World(
        name="arena",
        width_mm=400.0,
        depth_mm=400.0,
        height_mm=100.0,
        notes="Small open ground. No assigned task, no reward.",
    )


def stimulus_arena(*, side: str = "left") -> World:
    """A lamp exists in the world. The brain is not told its coordinates."""
    x, y = (80.0, 120.0) if side == "left" else (80.0, -120.0 + 200.0)
    return World(
        name=f"stimulus_{side}",
        width_mm=400.0,
        depth_mm=400.0,
        height_mm=100.0,
        lights=[Light(x_mm=x, y_mm=y, intensity=80.0)],
        odors=[Odor(x_mm=x, y_mm=y, intensity=0.6, sigma_mm=80.0)],
        notes="Bright object present. No privileged target_position into the brain.",
    )
