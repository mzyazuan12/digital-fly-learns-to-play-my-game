"""Identified MaleCNS output pathways → low-level locomotor commands.

Populations are resolved from annotations (type + side + body ID), not from
anonymous index ranges. The NeuroMechFly HybridTurningController still executes
joints; that is pretrained neuromuscular scaffolding, not VNC→MN→muscle.

Literature (not a claim that our LIF model reproduces those recordings):

- DNa02 / DNa01 / DNg13 / DNb05: ipsiversive walking steering;
  R−L DNa02 difference tracks rotational velocity
  (Rayshubskiy et al.; Yang et al. 2023; Feng et al. 2024).
- DNb06: contraversive steering (Yang et al. 2023).
- DNp09 (P9): walking initiation (Bidaye et al. 2020).
- MDN: backward walking (Bidaye et al. 2014).
- DNg02: wingbeat / flight-related descending population (Namiki catalogue).

Janelia MaleCNS v1.0 body IDs we resolve when present:

- DNa02  L=523769 R=10360
- DNa01  L=10442  R=10760
- DNg13  L=11074  R=512006
- DNb05  L=10118  R=10065
- DNb06  L=10888  R=11067
- DNp09  L=10783  R=11177
- MDN    L=11288,12348  R=10763,11332
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np

from flybrain.loader import Connectome


# Documented expected body IDs in MaleCNS v1.0 (uint64). Used for provenance,
# not as a hardcoded index map — lookup is still by type/side first.
EXPECTED_BODY_IDS = {
    "DNa02": {"L": (523769,), "R": (10360,)},
    "DNa01": {"L": (10442,), "R": (10760,)},
    "DNg13": {"L": (11074,), "R": (512006,)},
    "DNb05": {"L": (10118,), "R": (10065,)},
    "DNb06": {"L": (10888,), "R": (11067,)},
    "DNp09": {"L": (10783,), "R": (11177,)},
    "MDN": {"L": (11288, 12348), "R": (10763, 11332)},
}


@dataclass(frozen=True)
class Pathway:
    name: str
    types: tuple[str, ...]
    side: str | None
    literature: str
    maps_to: str
    body_ids: tuple[int, ...] = ()
    indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    resolved_types: tuple[str, ...] = ()


@dataclass
class MotorCommand:
    left: float
    right: float
    mode: str  # rest | walk | reverse | groom | fly
    walk_hz: float
    steer_l_hz: float
    steer_r_hz: float
    pathways_used: tuple[str, ...]
    flight_hz: float = 0.0
    groom_hz: float = 0.0


def _norm_side(value: object) -> str:
    text = str(value or "").strip().upper()
    if text in {"L", "LEFT", "LHS"}:
        return "L"
    if text in {"R", "RIGHT", "RHS"}:
        return "R"
    return ""


def _lookup_type(connectome: Connectome, typename: str) -> np.ndarray:
    exact = connectome.lookup(type_exact=typename)
    if exact.size:
        return exact
    return connectome.lookup(type_prefix=typename)


def _side_filter(connectome: Connectome, indices: np.ndarray, side: str | None) -> np.ndarray:
    if side is None or indices.size == 0:
        return indices.astype(np.int32)
    keep = [_norm_side(connectome.side[i]) == side for i in indices]
    out = indices[np.array(keep, dtype=bool)]
    return out.astype(np.int32)


def _group_rate(counts: np.ndarray, indices: np.ndarray, duration_s: float) -> float:
    if indices.size == 0 or duration_s <= 0:
        return 0.0
    return float(counts[indices].sum()) / (duration_s * max(1, indices.size))


class MotorBridge:
    """MaleCNS identified DNs → 2-vector command for the walking controller."""

    WALK_THRESHOLD = 0.18

    def __init__(self, connectome: Connectome):
        self.connectome = connectome
        self.pathways: list[Pathway] = []
        self.notes: dict = {"engineered_gain": True, "not_motor_neuron_control": True}
        self.walk_indices = self._resolve(
            "walk_initiation",
            ("DNp09",),
            None,
            maps_to="forward_speed",
            literature="Bidaye et al. 2020: DNp09/P9 walking initiation",
        )
        self.steer_left = self._resolve(
            "steer_left",
            ("DNa02", "DNa01", "DNg13", "DNb05"),
            "L",
            maps_to="ipsiversive_left",
            literature="Rayshubskiy/Yang/Feng: DNa02, DNa01, DNg13, DNb05 ipsiversive steering",
        )
        self.steer_right = self._resolve(
            "steer_right",
            ("DNa02", "DNa01", "DNg13", "DNb05"),
            "R",
            maps_to="ipsiversive_right",
            literature="Rayshubskiy/Yang/Feng: right copies predict right turns",
        )
        self.contra_left = self._resolve(
            "contra_left",
            ("DNb06",),
            "L",
            maps_to="contraversive_from_left",
            literature="Yang et al. 2023: DNb06 contraversive steering",
        )
        self.contra_right = self._resolve(
            "contra_right",
            ("DNb06",),
            "R",
            maps_to="contraversive_from_right",
            literature="Yang et al. 2023: DNb06 contraversive steering",
        )
        self.reverse_indices = self._resolve(
            "reverse",
            ("MDN",),
            None,
            maps_to="backward_walk",
            literature="Bidaye et al. 2014: MDN backward walking",
        )
        self.flight_indices = self._resolve(
            "flight",
            ("DNg02", "DNg02_a", "DNg02_b", "DNg02_c"),
            None,
            maps_to="wingbeat_and_takeoff",
            literature="Namiki DN catalogue: DNg02 wing/flight-related",
        )
        self.groom_indices = self._resolve(
            "groom",
            ("aDN1", "aDN", "DNg12", "DNg11"),
            None,
            maps_to="antennal_groom",
            literature="Antennal grooming DNs; joint CPG is an engineered VNC surrogate",
        )
        chemo = connectome.lookup(superclass="cb_sensory")
        contact = connectome.lookup(superclass="vnc_sensory")
        self.chemosensory = chemo.astype(np.int32) if chemo.size else np.zeros(0, np.int32)
        self.contact_indices = contact.astype(np.int32) if contact.size else np.zeros(0, np.int32)
        self.last_command = MotorCommand(0.0, 0.0, "rest", 0.0, 0.0, 0.0, ())
        mapping_path = Path(__file__).resolve().parents[1] / "data" / "motor_bridge_mapping.json"
        if mapping_path.exists():
            self.notes["mapping_file"] = str(mapping_path)
            self.notes["mapping_architecture"] = json.loads(mapping_path.read_text()).get("architecture")
        self.notes["pathways"] = [
            {
                "name": p.name,
                "types": p.types,
                "side": p.side,
                "n": int(p.indices.size),
                "body_ids": [int(connectome.neuron_ids[i]) for i in p.indices[:12]],
                "resolved_types": p.resolved_types,
                "literature": p.literature,
                "maps_to": p.maps_to,
            }
            for p in self.pathways
        ]
        self.notes["fallback"] = self._fallback_note()

    def _fallback_note(self) -> str:
        if self.walk_indices.size and self.steer_left.size and self.steer_right.size:
            return "identified_types"
        return "superclass_descending_split_lr"

    def _resolve(
        self,
        name: str,
        types: tuple[str, ...],
        side: str | None,
        *,
        maps_to: str,
        literature: str,
    ) -> np.ndarray:
        chunks = []
        found_types = []
        for typename in types:
            idx = _side_filter(self.connectome, _lookup_type(self.connectome, typename), side)
            if idx.size:
                chunks.append(idx)
                found_types.append(typename)
        if chunks:
            indices = np.unique(np.concatenate(chunks)).astype(np.int32)
        elif name in {"walk_initiation", "steer_left", "steer_right"}:
            indices = self._superclass_fallback(side)
            found_types = ("descending_neuron_superclass_fallback",)
        else:
            indices = np.zeros(0, dtype=np.int32)
            found_types = ()
        body_ids = tuple(int(self.connectome.neuron_ids[i]) for i in indices[:16])
        self.pathways.append(
            Pathway(
                name=name,
                types=types,
                side=side,
                literature=literature,
                maps_to=maps_to,
                body_ids=body_ids,
                indices=indices,
                resolved_types=tuple(found_types),
            )
        )
        return indices

    def _superclass_fallback(self, side: str | None) -> np.ndarray:
        dns = self.connectome.lookup(superclass="descending_neuron")
        if dns.size == 0:
            dns = self.connectome.lookup(superclass="descending_neuron_tbc")
        return _side_filter(self.connectome, dns, side)

    def read(
        self,
        counts: np.ndarray,
        duration_s: float,
        walking_drive: float = 0.0,
        walking_bout_s: float = 0.0,
        grooming_drive: float = 0.0,
        flight_drive: float = 0.0,
    ) -> MotorCommand:
        walk_hz = _group_rate(counts, self.walk_indices, duration_s)
        left_hz = _group_rate(counts, self.steer_left, duration_s)
        right_hz = _group_rate(counts, self.steer_right, duration_s)
        reverse_hz = _group_rate(counts, self.reverse_indices, duration_s)
        flight_hz = _group_rate(counts, self.flight_indices, duration_s)
        groom_hz = _group_rate(counts, self.groom_indices, duration_s)
        # DNb06 is contraversive: left DNb06 contributes to right turn.
        left_hz = left_hz + 0.5 * _group_rate(counts, self.contra_right, duration_s)
        right_hz = right_hz + 0.5 * _group_rate(counts, self.contra_left, duration_s)

        speed = np.clip(0.22 * walk_hz, 0.0, 1.2)
        if reverse_hz > walk_hz and reverse_hz > 2.0:
            speed = -np.clip(0.15 * reverse_hz, 0.0, 0.8)
        elif walking_bout_s > 0.05:
            # DNp09 is two cells; a short LIF window often records 0 Hz.
            # An open walking bout still has to drive the legs.
            speed = max(float(speed), float(np.clip(0.85 * walking_drive, 0.0, 1.15)))
        steer = np.clip(0.08 * (right_hz - left_hz), -0.8, 0.8)
        left = float(np.clip(abs(speed) * (1.0 - steer), 0.0, 1.4))
        right = float(np.clip(abs(speed) * (1.0 + steer), 0.0, 1.4))
        # Large DN pools spike tonically in the LIF. Takeoff / antennal
        # grooming follow the physiological bout that biased those cells,
        # not the raw group rate sitting above a few Hz.
        want_fly = flight_drive > 0.55 and walking_bout_s <= 0.05 and walking_drive < 0.42
        want_groom = grooming_drive > 0.55 and walking_bout_s <= 0.05 and walking_drive < 0.35
        if want_fly and abs(speed) < 0.45:
            mode = "fly"
            left, right = 0.0, 0.0
        elif want_groom and abs(speed) < self.WALK_THRESHOLD:
            mode = "groom"
            left, right = 0.0, 0.0
        elif abs(speed) < self.WALK_THRESHOLD:
            mode = "rest"
            left, right = 0.0, 0.0
        elif speed < 0:
            mode = "reverse"
        else:
            mode = "walk"
        used = tuple(
            p.name for p in self.pathways if p.indices.size and _group_rate(counts, p.indices, duration_s) > 0.2
        )
        cmd = MotorCommand(
            left=left,
            right=right,
            mode=mode,
            walk_hz=walk_hz,
            steer_l_hz=left_hz,
            steer_r_hz=right_hz,
            pathways_used=used,
            flight_hz=flight_hz,
            groom_hz=groom_hz,
        )
        self.last_command = cmd
        return cmd
