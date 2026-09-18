"""Every motor act must say where it came from.

Biological-model spikes, learned efficacies, pretrained CPGs, hard-coded
reflexes, and developer teleports are different claims. Do not mix them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class BehaviorSource(str, Enum):
    BIOLOGICAL_CONNECTOME = "biological_connectome"
    LEARNED_PLASTICITY = "learned_plasticity"
    PRETRAINED_LOCOMOTION_CONTROLLER = "pretrained_locomotion_controller"
    HAND_IMPLEMENTED_TRANSITION = "hand_implemented_transition"
    DEVELOPER_OVERRIDE = "developer_override"
    # Aliases kept so older logs remain readable.
    CONNECTOME_DYNAMICS = "biological_connectome"
    PRETRAINED_LOCOMOTOR = "pretrained_locomotion_controller"
    HARDCODED_REFLEX = "hand_implemented_transition"


@dataclass
class StepRecord:
    t_ms: float
    sources: tuple[BehaviorSource, ...]
    descending: tuple[float, float]
    x_mm: float
    y_mm: float
    heading_rad: float
    left_eye: float
    right_eye: float
    total_spikes: int
    world: str
    mode: str = "rest"
    walking_drive: float = 0.0
    notes: str = ""
    developer: bool = False

    def as_dict(self) -> dict:
        return {
            "t_ms": self.t_ms,
            "sources": [s.value for s in self.sources],
            "descending": list(self.descending),
            "x_mm": self.x_mm,
            "y_mm": self.y_mm,
            "heading_rad": self.heading_rad,
            "left_eye": self.left_eye,
            "right_eye": self.right_eye,
            "total_spikes": self.total_spikes,
            "world": self.world,
            "mode": self.mode,
            "walking_drive": self.walking_drive,
            "notes": self.notes,
            "developer": self.developer,
        }


@dataclass
class ProvenanceLog:
    records: list[StepRecord] = field(default_factory=list)

    def append(self, record: StepRecord) -> None:
        self.records.append(record)

    def learned_behavior_records(self) -> list[StepRecord]:
        """Exclude developer overrides. Learned ≠ teleported."""
        return [r for r in self.records if not r.developer]

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {s.value: 0 for s in BehaviorSource}
        for record in self.records:
            for source in record.sources:
                counts[source.value] += 1
        counts["steps"] = len(self.records)
        counts["developer_steps"] = sum(1 for r in self.records if r.developer)
        return counts
