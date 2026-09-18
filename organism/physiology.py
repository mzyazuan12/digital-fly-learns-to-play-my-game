"""Internal metabolic state. Not a behavior scheduler.

Hunger, arousal, and fatigue are allowed. They may not call walk(), groom(),
or fly(). They update neuromodulators. Neuromodulators change identified
circuits. Motor output, if any, comes from the network.

Bout timers remain only behind LEGACY_SCAFFOLD for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from organism.config import LEGACY_SCAFFOLD, ScaffoldViolation
from organism.neuromodulation import Neuromodulation


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


@dataclass
class InternalState:
    arousal: float = 0.45
    hunger: float = 0.25
    thirst: float = 0.15
    fatigue: float = 0.10
    sleep_pressure: float = 0.05
    threat: float = 0.0
    novelty: float = 0.40
    # Derived / legacy readouts. Not motor commands in the default path.
    walking_drive: float = 0.0
    flight_drive: float = 0.0
    grooming_drive: float = 0.0
    walking_bout_s: float = 0.0
    grooming_bout_s: float = 0.0
    flight_bout_s: float = 0.0
    steer_bias_l: float = 0.0
    steer_bias_r: float = 0.0

    def snapshot(self) -> dict:
        return asdict(self)

    def load(self, payload: dict) -> None:
        for key, value in payload.items():
            if hasattr(self, key) and key != "last_sources":
                try:
                    setattr(self, key, float(value))
                except (TypeError, ValueError):
                    continue


class Physiology:
    """Slow metabolic variables. Modeling, not measured MaleCNS."""

    def __init__(self, seed: int = 0, *, legacy_scaffold: bool | None = None):
        self.state = InternalState()
        self.rng = np.random.default_rng(seed)
        self.seconds = 0.0
        self.legacy_scaffold = LEGACY_SCAFFOLD if legacy_scaffold is None else bool(legacy_scaffold)
        self.neuromodulation = Neuromodulation(seed=seed + 17)

    def step(
        self,
        dt_s: float,
        *,
        walking: bool,
        contact: float,
        odor: float,
        vision: float,
    ) -> InternalState:
        s = self.state
        dt_s = float(min(max(dt_s, 1e-4), 0.25))
        n = lambda scale: float(self.rng.normal(0.0, scale))
        s.arousal = _clip01(
            s.arousal
            + dt_s * (0.38 - s.arousal) / 5.0
            - 0.04 * s.fatigue * dt_s
            + n(0.04) * np.sqrt(dt_s)
        )
        s.hunger = _clip01(s.hunger + dt_s * 0.004 - 0.08 * odor * dt_s)
        s.thirst = _clip01(s.thirst + dt_s * 0.002)
        if walking:
            s.fatigue = _clip01(s.fatigue + dt_s * 0.05)
            s.sleep_pressure = _clip01(s.sleep_pressure + dt_s * 0.01)
        else:
            s.fatigue = _clip01(s.fatigue - dt_s * 0.04)
            s.sleep_pressure = _clip01(s.sleep_pressure - dt_s * 0.005)
        s.threat = _clip01(0.85 * s.threat + 0.6 * contact)
        s.novelty = _clip01(s.novelty + dt_s * (0.15 - s.novelty) / 8.0 + n(0.05) * np.sqrt(dt_s))
        self.neuromodulation.update_from_metabolic(
            dt_s,
            hunger=s.hunger,
            arousal=s.arousal,
            fatigue=s.fatigue,
            sleep_pressure=s.sleep_pressure,
            odor=odor,
        )
        if self.legacy_scaffold:
            self._legacy_bouts(dt_s, n)
        else:
            s.walking_bout_s = 0.0
            s.grooming_bout_s = 0.0
            s.flight_bout_s = 0.0
            s.flight_drive = _clip01(0.90 * s.flight_drive)
            s.grooming_drive = _clip01(0.90 * s.grooming_drive)
            s.steer_bias_l = float(
                np.clip(0.94 * s.steer_bias_l + n(0.15) * np.sqrt(max(dt_s, 1e-6)), -1.0, 1.0)
            )
            s.steer_bias_r = float(
                np.clip(0.94 * s.steer_bias_r + n(0.15) * np.sqrt(max(dt_s, 1e-6)), -1.0, 1.0)
            )
        self.seconds += dt_s
        return s

    def _legacy_bouts(self, dt_s: float, n) -> None:
        """Comparison-only personality scheduler. Not the default organism."""
        if not self.legacy_scaffold:
            raise ScaffoldViolation("bout scheduler is LEGACY_SCAFFOLD only")
        s = self.state
        was_walking_bout = s.walking_bout_s > 0.0
        if s.flight_bout_s > 0.0:
            s.flight_bout_s = max(0.0, s.flight_bout_s - dt_s)
            s.flight_drive = _clip01(0.82 + n(0.04) * np.sqrt(max(dt_s, 1e-6)))
            s.walking_bout_s = 0.0
            s.grooming_bout_s = 0.0
            s.walking_drive = _clip01(0.12 * s.walking_drive)
            s.grooming_drive = _clip01(0.70 * s.grooming_drive)
            return
        s.flight_drive = _clip01(0.90 * s.flight_drive)
        if (
            s.arousal > 0.72
            and s.novelty > 0.55
            and s.fatigue < 0.35
            and self.rng.random() < 0.02 * dt_s
        ):
            s.flight_bout_s = float(self.rng.uniform(0.4, 1.0))
            s.flight_drive = 0.8
            s.walking_bout_s = 0.0
            s.grooming_bout_s = 0.0
            s.walking_drive = 0.1
            return
        if s.grooming_bout_s > 0.0:
            s.grooming_bout_s = max(0.0, s.grooming_bout_s - dt_s)
            bout = 0.04
            s.grooming_drive = _clip01(0.72 + n(0.05) * np.sqrt(max(dt_s, 1e-6)))
        elif s.walking_bout_s > 0.0:
            s.walking_bout_s = max(0.0, s.walking_bout_s - dt_s)
            bout = 0.88
            s.grooming_drive = _clip01(0.82 * s.grooming_drive)
            if was_walking_bout and s.walking_bout_s == 0.0:
                s.grooming_bout_s = float(self.rng.uniform(0.8, 2.4))
        else:
            bout = 0.04
            start_hz = (
                0.35
                + 0.90 * s.arousal
                + 0.40 * s.hunger
                + 0.35 * s.novelty
                - 0.90 * s.fatigue
                - 1.10 * s.sleep_pressure
                - 0.40 * s.threat
            )
            if self.rng.random() < max(0.0, start_hz) * dt_s:
                s.walking_bout_s = float(self.rng.uniform(2.0, 5.5))
                bout = 0.88
            s.grooming_drive = _clip01(
                0.12 + 0.30 * s.fatigue + n(0.04) * np.sqrt(max(dt_s, 1e-6))
            )
        s.walking_drive = _clip01(
            0.35 * s.walking_drive + 0.65 * bout + n(0.04) * np.sqrt(max(dt_s, 1e-6))
        )
        s.steer_bias_l = float(
            np.clip(0.94 * s.steer_bias_l + n(0.55) * np.sqrt(max(dt_s, 1e-6)), -1.0, 1.0)
        )
        s.steer_bias_r = float(
            np.clip(0.94 * s.steer_bias_r + n(0.55) * np.sqrt(max(dt_s, 1e-6)), -1.0, 1.0)
        )

    def salient_event(self, strength: float = 1.0) -> None:
        """A world event the fly can notice. Does not choose a gait or a word."""
        s = self.state
        amp = float(np.clip(strength, 0.0, 1.5))
        s.novelty = _clip01(s.novelty + 0.55 * amp)
        s.arousal = _clip01(s.arousal + 0.40 * amp)
        if self.legacy_scaffold:
            s.walking_bout_s = max(s.walking_bout_s, 2.8 * amp)
            s.walking_drive = _clip01(max(s.walking_drive, 0.72 * amp))
            s.grooming_bout_s = 0.0

    def modulate(self, net, bridge) -> dict[str, float]:
        """Neuromodulators change circuits. Legacy path may still inject walk current."""
        sources = self.neuromodulation.modulate(net, bridge)
        if not self.legacy_scaffold:
            return sources
        s = self.state
        net.add_drive(bridge.walk_indices, 90.0 * s.walking_drive, source="legacy.walking_drive")
        net.add_drive(
            bridge.steer_left,
            8.0 * s.walking_drive + 22.0 * s.steer_bias_l,
            source="legacy.steer_L",
        )
        net.add_drive(
            bridge.steer_right,
            8.0 * s.walking_drive + 22.0 * s.steer_bias_r,
            source="legacy.steer_R",
        )
        net.add_drive(bridge.flight_indices, 70.0 * s.flight_drive, source="legacy.flight_drive")
        if s.hunger > 0.05:
            net.add_drive(bridge.chemosensory, 6.0 * s.hunger, source="legacy.hunger")
        if s.threat > 0.1:
            net.add_drive(bridge.contact_indices, 12.0 * s.threat, source="legacy.threat")
        if s.grooming_drive > 0.28 and s.walking_drive < 0.35:
            net.add_drive(bridge.walk_indices, -18.0 * s.grooming_drive, source="legacy.groom_suppress_walk")
            net.add_drive(bridge.groom_indices, 70.0 * s.grooming_drive, source="legacy.grooming_drive")
        sources["legacy_scaffold"] = 1.0
        return sources
