"""Explicit neuromodulatory layer.

Internal metabolic variables (hunger, arousal, fatigue) do not call motor
actions. They change neuromodulator activity. Modulators change identified
circuits. Behavior, if any, comes out of network dynamics.

This is not a complete reconstruction of fly neuromodulation. Gap junctions,
neuropeptides, and cotransmission are largely absent from current connectomes.
Every gain below is labeled.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from flybrain.network import LIFNetwork
from organism.config import ParameterProvenance


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


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
        self.provenance = {
            "dopamine_starvation_sugar_sensing": ParameterProvenance.LITERATURE_DERIVED.value,
            "octopamine_locomotor_excitability": ParameterProvenance.LITERATURE_DERIVED.value,
            "serotonin_gain": ParameterProvenance.ASSUMED.value,
            "receptor_assignment": ParameterProvenance.INFERRED.value,
            "numeric_gains": ParameterProvenance.ASSUMED.value,
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

        Not: if hunger > 0.7: walk(). Hunger raises dopamine; dopamine
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
        # Octopamine raises excitability of walking-related descending neurons.
        # This is NOT a walk() call: if those cells do not spike, the fly rests.
        oa_walk = 14.0 * s.octopamine
        if bridge.walk_indices.size and oa_walk != 0.0:
            net.add_drive(bridge.walk_indices, oa_walk, source="neuromod.octopamine.DNp09")
            sources["neuromod.octopamine.DNp09"] = oa_walk
        oa_steer = 4.0 * s.octopamine
        if bridge.steer_left.size:
            net.add_drive(bridge.steer_left, oa_steer, source="neuromod.octopamine.steer_L")
            sources["neuromod.octopamine.steer_L"] = oa_steer
        if bridge.steer_right.size:
            net.add_drive(bridge.steer_right, oa_steer, source="neuromod.octopamine.steer_R")
            sources["neuromod.octopamine.steer_R"] = oa_steer
        # Dopamine: chemosensory gain under starvation (literature-inspired).
        da_chemo = 8.0 * s.dopamine
        if getattr(bridge, "chemosensory", np.zeros(0)).size and da_chemo != 0.0:
            net.add_drive(bridge.chemosensory, da_chemo, source="neuromod.dopamine.chemosensory")
            sources["neuromod.dopamine.chemosensory"] = da_chemo
        # Serotonin: weak global sensory gain change. ASSUMED.
        se_contact = 3.0 * s.serotonin
        if getattr(bridge, "contact_indices", np.zeros(0)).size and se_contact != 0.0:
            net.add_drive(bridge.contact_indices, se_contact, source="neuromod.serotonin.contact")
            sources["neuromod.serotonin.contact"] = se_contact
        s.last_sources = sources
        net.note_drive_sources(sources)
        return sources
