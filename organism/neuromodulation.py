"""Explicit neuromodulatory layer.

Internal metabolic variables (hunger, arousal, fatigue) do not call motor
actions. They change neuromodulator activity. Modulators change identified
circuits. Behavior, if any, comes out of network dynamics.

This is not a complete reconstruction of fly neuromodulation. Gap junctions,
neuropeptides, and cotransmission are largely absent from current connectomes.

Every numeric gain is an ASSUMED mapping. Do not raise these numbers until
the fly walks and then call the resulting walk emergent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from flybrain.network import LIFNetwork
from organism.config import ParameterProvenance


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


@dataclass(frozen=True)
class ModulatoryEffect:
    modulator: str
    target_population: str
    effect: str
    source: str
    confidence: str
    gain_per_unit: float
    drive_source: str

    def as_dict(self) -> dict:
        return asdict(self)


# Numeric gains are ASSUMED. Literature citations justify *which* circuit is
# touched, not the current amplitude. Do not retune to force walking.
MODULATORY_EFFECTS: tuple[ModulatoryEffect, ...] = (
    ModulatoryEffect(
        modulator="octopamine",
        target_population="DNp09",
        effect="add_excitatory_current",
        source="Insect octopamine / arousal literature (locomotor excitability). Numeric gain is not a measured receptor current.",
        confidence=ParameterProvenance.ASSUMED.value,
        gain_per_unit=14.0,
        drive_source="neuromod.octopamine.DNp09",
    ),
    ModulatoryEffect(
        modulator="octopamine",
        target_population="steer_DNs",
        effect="add_excitatory_current",
        source="Same OA-locomotion literature; assignment to DNa/DNg/DNb copies is INFERRED.",
        confidence=ParameterProvenance.ASSUMED.value,
        gain_per_unit=4.0,
        drive_source="neuromod.octopamine.steer",
    ),
    ModulatoryEffect(
        modulator="dopamine",
        target_population="chemosensory",
        effect="add_excitatory_current",
        source="Inagaki et al. 2012: starvation increases dopaminergic modulation of sugar sensing. Numeric gain ASSUMED.",
        confidence=ParameterProvenance.ASSUMED.value,
        gain_per_unit=8.0,
        drive_source="neuromod.dopamine.chemosensory",
    ),
    ModulatoryEffect(
        modulator="serotonin",
        target_population="contact_sensory",
        effect="add_excitatory_current",
        source="Placeholder slow-state map. No claim this is a measured 5-HT current onto VNC sensory cells.",
        confidence=ParameterProvenance.ASSUMED.value,
        gain_per_unit=3.0,
        drive_source="neuromod.serotonin.contact",
    ),
)


def _effect(modulator: str, target: str) -> ModulatoryEffect:
    for item in MODULATORY_EFFECTS:
        if item.modulator == modulator and item.target_population == target:
            return item
    raise KeyError(f"{modulator} → {target}")


@dataclass
class NeuromodulatoryState:
    dopamine: float = 0.20
    octopamine: float = 0.25
    serotonin: float = 0.20
    # Inspectable: how much current this step actually added, and why.
    last_sources: dict[str, float] | None = None

    def snapshot(self) -> dict:
        payload = asdict(self)
        payload["last_sources"] = dict(self.last_sources or {})
        return payload

    def load(self, payload: dict) -> None:
        for key in ("dopamine", "octopamine", "serotonin"):
            if key in payload:
                setattr(self, key, float(payload[key]))


class Neuromodulation:
    """Chemical control layer on top of the structural connectome."""

    def __init__(self, seed: int = 0):
        self.state = NeuromodulatoryState()
        self.rng = np.random.default_rng(seed)
        self.effects = MODULATORY_EFFECTS
        self.provenance = {
            "dopamine_starvation_sugar_sensing": ParameterProvenance.LITERATURE_DERIVED.value,
            "octopamine_locomotor_excitability": ParameterProvenance.LITERATURE_DERIVED.value,
            "serotonin_gain": ParameterProvenance.ASSUMED.value,
            "receptor_assignment": ParameterProvenance.INFERRED.value,
            "numeric_gains": ParameterProvenance.ASSUMED.value,
            "effects": [item.as_dict() for item in self.effects],
        }

    def update_from_metabolic(
        self,
        dt_s: float,
        *,
        hunger: float,
        arousal: float,
        fatigue: float,
        sleep_pressure: float,
        odor: float,
    ) -> NeuromodulatoryState:
        """Map slow internal state onto modulator concentrations.

        Not if-hunger-then-walk. Hunger raises dopamine; dopamine
        changes identified circuits. Walking is not selected here.
        """
        s = self.state
        dt_s = float(min(max(dt_s, 1e-4), 0.25))
        n = lambda scale: float(self.rng.normal(0.0, scale))
        # Starvation increases dopaminergic modulation of sugar sensing
        # (Inagaki et al. 2012). Numeric map is ASSUMED.
        da_target = _clip01(0.12 + 0.70 * hunger + 0.15 * odor)
        # Octopamine tracks arousal and opposes fatigue (insect locomotion
        # literature). Numeric map is ASSUMED.
        oa_target = _clip01(0.10 + 0.65 * arousal - 0.35 * fatigue - 0.20 * sleep_pressure)
        # Serotonin: placeholder slow state. ASSUMED.
        se_target = _clip01(0.18 + 0.25 * sleep_pressure + 0.10 * fatigue)
        blend = min(1.0, dt_s / 2.5)
        s.dopamine = _clip01((1.0 - blend) * s.dopamine + blend * da_target + n(0.02) * np.sqrt(dt_s))
        s.octopamine = _clip01((1.0 - blend) * s.octopamine + blend * oa_target + n(0.02) * np.sqrt(dt_s))
        s.serotonin = _clip01((1.0 - blend) * s.serotonin + blend * se_target + n(0.015) * np.sqrt(dt_s))
        return s

    def modulate(self, net: LIFNetwork, bridge) -> dict[str, float]:
        """Change identified circuits. Do not write a motor command."""
        s = self.state
        sources: dict[str, float] = {}
        oa = _effect("octopamine", "DNp09")
        oa_walk = oa.gain_per_unit * s.octopamine
        if bridge.walk_indices.size and oa_walk != 0.0:
            net.add_drive(bridge.walk_indices, oa_walk, source=oa.drive_source)
            sources[oa.drive_source] = oa_walk
        steer = _effect("octopamine", "steer_DNs")
        oa_steer = steer.gain_per_unit * s.octopamine
        if bridge.steer_left.size:
            net.add_drive(bridge.steer_left, oa_steer, source="neuromod.octopamine.steer_L")
            sources["neuromod.octopamine.steer_L"] = oa_steer
        if bridge.steer_right.size:
            net.add_drive(bridge.steer_right, oa_steer, source="neuromod.octopamine.steer_R")
            sources["neuromod.octopamine.steer_R"] = oa_steer
        da = _effect("dopamine", "chemosensory")
        da_chemo = da.gain_per_unit * s.dopamine
        if getattr(bridge, "chemosensory", np.zeros(0)).size and da_chemo != 0.0:
            net.add_drive(bridge.chemosensory, da_chemo, source=da.drive_source)
            sources[da.drive_source] = da_chemo
        se = _effect("serotonin", "contact_sensory")
        se_contact = se.gain_per_unit * s.serotonin
        if getattr(bridge, "contact_indices", np.zeros(0)).size and se_contact != 0.0:
            net.add_drive(bridge.contact_indices, se_contact, source=se.drive_source)
            sources[se.drive_source] = se_contact
        s.last_sources = sources
        net.note_drive_sources(sources)
        return sources
