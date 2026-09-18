"""Developer tools. Separate from autonomous neural control.

Never count teleport / reset / spawn as learned behavior.
"""

from __future__ import annotations

from organism.body import Pose
from organism.provenance import BehaviorSource, StepRecord


class DeveloperControls:
    def __init__(self, fly: "VirtualFly"):  # noqa: F821
        self.fly = fly
        self.paused = False

    def teleport(self, x_mm: float, y_mm: float, heading_rad: float = 0.0) -> StepRecord:
        if self.fly.body is None:
            raise RuntimeError("Fly is not inhabiting a world")
        pose = Pose(x_mm=x_mm, y_mm=y_mm, z_mm=self.fly.body.pose.z_mm, heading_rad=heading_rad)
        self.fly.body.teleport(pose)
        record = StepRecord(
            t_ms=self.fly.net.sim_ms,
            sources=(BehaviorSource.DEVELOPER_OVERRIDE,),
            descending=(0.0, 0.0),
            x_mm=pose.x_mm,
            y_mm=pose.y_mm,
            heading_rad=pose.heading_rad,
            left_eye=0.0,
            right_eye=0.0,
            total_spikes=int(self.fly.net.total_spikes),
            world=self.fly.world.name if self.fly.world else "",
            notes="developer teleport",
            developer=True,
        )
        self.fly.provenance.append(record)
        self.fly.history.append({"event": "teleport", "x_mm": x_mm, "y_mm": y_mm})
        return record

    def pause(self, paused: bool = True) -> None:
        self.paused = paused

    def inspect_neurons(self, indices) -> dict:
        idx = list(indices)
        return {
            "v": self.fly.net.v[idx].tolist(),
            "g": self.fly.net.g[idx].tolist(),
            "counts": self.fly.net.counts[idx].tolist(),
        }

    def inspect_efficacy_stats(self) -> dict:
        eff = self.fly.net.efficacy
        return {
            "mean": float(eff.mean()),
            "std": float(eff.std()),
            "min": float(eff.min()),
            "max": float(eff.max()),
            "changed": int((abs(eff - 1.0) > 1e-6).sum()),
        }
