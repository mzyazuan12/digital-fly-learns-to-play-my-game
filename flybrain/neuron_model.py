"""Per-neuron dynamics metadata.

MaleCNS does not specify membranes. Do not treat every cell as the same LIF
unit. VNC premotor neurons are often nonspiking in insects; walking models
therefore use analog graded_release (not a firing rate) for substantial
parts of those circuits.

Every assigned parameter is labeled MEASURED, LITERATURE_DERIVED, INFERRED,
or ASSUMED.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from enum import Enum

from flybrain.loader import Connectome


class ParameterProvenance(str, Enum):
    MEASURED = "MEASURED"
    LITERATURE_DERIVED = "LITERATURE_DERIVED"
    INFERRED = "INFERRED"
    ASSUMED = "ASSUMED"


class NeuronKind(str, Enum):
    SPIKING_LIF = "spiking_lif"
    GRADED_RATE = "graded_rate"


# Superclasses / type fragments treated as graded (nonspiking) by default.
# Literature: insect VNC walking premotor neurons are often nonspiking
# (see connectome CPG modeling of fly walking). Not a MaleCNS measurement.
_GRADED_SUPERCLASS = {
    "vnc_intrinsic",
    "vnc_premotor",
    "vnc_interneuron",
}
_GRADED_TYPE_FRAGMENTS = (
    "IN",
    "premotor",
    "Int",
)


@dataclass
class NeuronModelTable:
    kind: np.ndarray  # object[str] NeuronKind values
    provenance: np.ndarray  # object[str]
    notes: dict

    def graded_mask(self) -> np.ndarray:
        return self.kind == NeuronKind.GRADED_RATE.value

    def spiking_mask(self) -> np.ndarray:
        return self.kind == NeuronKind.SPIKING_LIF.value

    def snapshot(self) -> dict:
        graded = int(self.graded_mask().sum())
        return {
            "n": int(self.kind.size),
            "n_spiking_lif": int(self.kind.size) - graded,
            "n_graded_rate": graded,
            "notes": self.notes,
        }


def assign_neuron_models(connectome: Connectome) -> NeuronModelTable:
    """Classify cells. Default is ASSUMED identical LIF; VNC premotor is graded."""
    n = connectome.n
    kind = np.full(n, NeuronKind.SPIKING_LIF.value, dtype=object)
    provenance = np.full(n, ParameterProvenance.ASSUMED.value, dtype=object)
    superclasses = connectome.superclass
    types = connectome.cell_type
    classes = connectome.cell_class
    graded = 0
    for i in range(n):
        sc = str(superclasses[i] or "")
        typ = str(types[i] or "")
        cls = str(classes[i] or "")
        if sc in _GRADED_SUPERCLASS or cls in {"premotor", "vnc_premotor"}:
            kind[i] = NeuronKind.GRADED_RATE.value
            provenance[i] = ParameterProvenance.LITERATURE_DERIVED.value
            graded += 1
            continue
        if sc.startswith("vnc") and any(frag in typ for frag in _GRADED_TYPE_FRAGMENTS):
            kind[i] = NeuronKind.GRADED_RATE.value
            provenance[i] = ParameterProvenance.LITERATURE_DERIVED.value
            graded += 1
    notes = {
        "default_kind": NeuronKind.SPIKING_LIF.value,
        "default_provenance": ParameterProvenance.ASSUMED.value,
        "graded_rule": (
            "VNC premotor / local interneuron labels use analog graded_release "
            "(literature: many insect walking premotor neurons are nonspiking). "
            "This is not a firing rate. All other cells are identical "
            "current-based LIF (ASSUMED)."
        ),
        "n_graded": graded,
        "identical_lif_for_all_cells": False,
    }
    return NeuronModelTable(kind=kind, provenance=provenance, notes=notes)
