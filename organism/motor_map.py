"""Motor-neuron → muscle → FlyBody actuator mapping.

MaleCNS / MANC contain annotated motor neurons. Published muscle-target
atlases (including female FANC) let us attach some of those cells to
FlyBody joints. The mapping is incomplete. Cross-specimen / cross-sex
transfers are labeled as inference, not measurements from the MaleCNS
specimen.

This module does not by itself walk the fly. MODE_ENGINEERED_CPG still
uses the FlyGym HybridTurningController. MODE_NEURAL_MOTOR is the
research target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from flybrain.loader import Connectome
from organism.config import MotorMode, ParameterProvenance, motor_fidelity_level

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP = ROOT / "data" / "motor_neuron_muscle_map.json"


@dataclass
class MotorNeuronMuscleEntry:
    malecns_body_id: int | None
    manc_type: str
    motor_neuron_type: str
    body_side: str
    target_body_part: str
    target_muscle: str
    joint_action: str
    data_source: str
    confidence: float
    mapping_kind: str  # direct | inferred_cross_specimen | inferred_cross_sex
    provenance: str = ParameterProvenance.INFERRED.value
    index: int | None = None
    resolved_type: str = ""

    def as_dict(self) -> dict:
        return {
            "malecns_body_id": self.malecns_body_id,
            "manc_type": self.manc_type,
            "motor_neuron_type": self.motor_neuron_type,
            "body_side": self.body_side,
            "target_body_part": self.target_body_part,
            "target_muscle": self.target_muscle,
            "joint_action": self.joint_action,
            "data_source": self.data_source,
            "confidence": self.confidence,
            "mapping_kind": self.mapping_kind,
            "provenance": self.provenance,
            "index": self.index,
            "resolved_type": self.resolved_type,
        }


def _norm_side(value: object) -> str:
    text = str(value or "").strip().upper()
    if text in {"L", "LEFT", "LHS"}:
        return "L"
    if text in {"R", "RIGHT", "RHS"}:
        return "R"
    return ""


class MotorNeuronMuscleMap:
    """Resolve annotated motor neurons onto FlyBody actuators when possible."""

    def __init__(self, connectome: Connectome, path: Path | None = None):
        self.connectome = connectome
        self.path = Path(path or DEFAULT_MAP)
        raw = json.loads(self.path.read_text()) if self.path.exists() else {"entries": []}
        self.catalog_note = str(raw.get("note", ""))
        self.references = list(raw.get("references", []))
        self.entries: list[MotorNeuronMuscleEntry] = []
        for item in raw.get("entries", []):
            self.entries.append(
                MotorNeuronMuscleEntry(
                    malecns_body_id=item.get("malecns_body_id"),
                    manc_type=str(item.get("manc_type") or ""),
                    motor_neuron_type=str(item.get("motor_neuron_type") or ""),
                    body_side=_norm_side(item.get("body_side")),
                    target_body_part=str(item.get("target_body_part") or ""),
                    target_muscle=str(item.get("target_muscle") or ""),
                    joint_action=str(item.get("joint_action") or ""),
                    data_source=str(item.get("data_source") or ""),
                    confidence=float(item.get("confidence") or 0.0),
                    mapping_kind=str(item.get("mapping_kind") or "inferred_cross_specimen"),
                    provenance=str(item.get("provenance") or ParameterProvenance.INFERRED.value),
                )
            )
        self._resolve()
        self.vnc_motor_indices = self._lookup_vnc_motor()
        self.notes = {
            "n_catalog": len(self.entries),
            "n_resolved_body_id": int(sum(1 for e in self.entries if e.index is not None)),
            "n_vnc_motor_annotated": int(self.vnc_motor_indices.size),
            "not_complete_muscle_recovery": True,
            "cross_sex_inference_present": any(
                e.mapping_kind == "inferred_cross_sex" for e in self.entries
            ),
            "catalog": self.path.name,
        }

    def _lookup_vnc_motor(self) -> np.ndarray:
        chunks = [
            self.connectome.lookup(superclass="vnc_motor"),
            self.connectome.lookup(superclass="vnc_motor_neuron"),
            self.connectome.lookup(cell_class="motor"),
            self.connectome.lookup(cell_class="motor_neuron"),
            self.connectome.lookup(type_prefix="MN"),
        ]
        chunks = [c for c in chunks if c.size]
        if not chunks:
            return np.zeros(0, dtype=np.int32)
        return np.unique(np.concatenate(chunks)).astype(np.int32)

    def _resolve(self) -> None:
        ids = self.connectome.neuron_ids
        for entry in self.entries:
            if entry.malecns_body_id is None:
                idx = self.connectome.lookup(type_prefix=entry.manc_type)
                if idx.size == 0:
                    idx = self.connectome.lookup(type_exact=entry.motor_neuron_type)
                if entry.body_side and idx.size:
                    keep = [
                        i
                        for i in idx
                        if _norm_side(self.connectome.side[i]) == entry.body_side
                    ]
                    idx = np.asarray(keep, dtype=np.int32)
                if idx.size:
                    entry.index = int(idx[0])
                    entry.malecns_body_id = int(ids[entry.index])
                    entry.resolved_type = str(self.connectome.cell_type[entry.index])
                continue
            body = np.uint64(entry.malecns_body_id)
            i = int(np.searchsorted(ids, body))
            if i < len(ids) and ids[i] == body:
                entry.index = i
                entry.resolved_type = str(self.connectome.cell_type[i])

    def activity(self, counts: np.ndarray, duration_s: float) -> list[dict]:
        rows = []
        for entry in self.entries:
            hz = 0.0
            if entry.index is not None and duration_s > 0:
                hz = float(counts[entry.index]) / duration_s
            rows.append({**entry.as_dict(), "hz": hz})
        return rows

    def snapshot(self) -> dict:
        return {
            "notes": self.notes,
            "references": self.references,
            "entries": [e.as_dict() for e in self.entries],
        }


def command_for_mode(
    mode: MotorMode,
    *,
    left: float,
    right: float,
    walk_mode: str,
    mn_activity: list[dict],
) -> dict:
    """Document which controller is allowed to move the body.

    MODE_ENGINEERED_CPG: FlyGym CPG amplitudes from descending rates.
    MODE_HYBRID_VNC: CPG still executes; MN rates are logged beside it.
    MODE_NEURAL_CPG: timing is the per-leg E1/E2/I1 motif; joints are not
    actuated from that motif yet, and FlyGym PreprogrammedSteps is not the
    walk generator.
    MODE_NEURAL_MOTOR: reserved; muscle actuation does not exist yet.
    """
    neural_cpg = mode is MotorMode.NEURAL_CPG
    neural_motor = mode is MotorMode.NEURAL_MOTOR
    return {
        "motor_mode": mode.value,
        "motor_fidelity_level": motor_fidelity_level(mode),
        "cpg_left": float(left),
        "cpg_right": float(right),
        "body_mode": walk_mode,
        "mn_n": len(mn_activity),
        "mn_active": int(sum(1 for row in mn_activity if row.get("hz", 0) > 0.2)),
        "legs_still_cpg": mode in (MotorMode.ENGINEERED_CPG, MotorMode.HYBRID_VNC),
        "neural_cpg_timing": neural_cpg or neural_motor,
        "joints_from_neural_cpg": False,
        "neural_motor_ready": False,
    }
