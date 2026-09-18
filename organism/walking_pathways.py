"""Identified walking DNs and the published VNC rhythm circuit.

This module looks up real MaleCNS types. It does not call walk(), stand(),
or groom(). Names in WALKING_CIRCUIT_TYPES tag cells for measurement and
lesion; they never mean `if DNg100: walk()`. Joint motion still comes from
the engineered FlyGym CPG in MODE_ENGINEERED_CPG. MODE_NEURAL_CPG records
timing from the per-leg E1/E2/I1 motif and does not actuate FlyBody yet.

Literature (not a claim that our LIF model reproduces those recordings):

- DNp09 / P9: forward walking with turning; strong activation can freeze
  (Bidaye et al. 2020; Zacarias et al. 2018).
- DNg100 / BDN2: walking command that accesses a VNC rhythm generator even
  in headless flies (Sapkal et al. 2024; Pugliese et al. 2025).
- DNb08: rhythmic searching/flailing, not coordinated walking
  (Pugliese et al. 2025).
- oDN1 / DNg97: bolt-related forward walking DN (Sapkal et al. 2024).
- DNa01 / DNa02: ipsiversive steering (Yang / Rayshubskiy / Feng).
- MDN: backward walking (Bidaye et al. 2014).
- Halt: Foxglove/CB0890 walk-OFF, Bluebell/DNg60 walk-OFF, Brake/AN19A018
  (Sapkal et al. 2024). Foxglove is a FlyWire type; MaleCNS may not label it.
- Core CPG, one copy per leg neuropil (Pugliese et al. 2025 published):
  E1=IN17A001, E2=INXXX466, I1=IN16B036, I2=IN19A007, E3=IN19B012,
  E4=IN03A006, E5=INXXX464. An older preprint passage appears to call E5
  INXXX466 (the E2 type); canonical mapping is the published article.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from flybrain.loader import Connectome


@dataclass(frozen=True)
class PathwaySpec:
    name: str
    types: tuple[str, ...]
    aliases: tuple[str, ...]
    role: str
    maps_to: str
    literature: str
    family: str  # command_dn | halt | cpg | motor


WALKING_DNS: tuple[PathwaySpec, ...] = (
    PathwaySpec(
        name="DNp09",
        types=("DNp09",),
        aliases=("P9",),
        role="forward_walk_and_turn",
        maps_to="forward_locomotor_drive",
        literature="Bidaye et al. 2020 walking initiation; Zacarias et al. 2018 freeze",
        family="command_dn",
    ),
    PathwaySpec(
        name="DNg100",
        types=("DNg100",),
        aliases=("BDN2",),
        role="walking_command_cpg_access",
        maps_to="forward_locomotor_drive",
        literature="Pugliese et al. 2025; Sapkal et al. 2024 BDN2 walking, including headless",
        family="command_dn",
    ),
    PathwaySpec(
        name="DNb08",
        types=("DNb08",),
        aliases=(),
        role="rhythmic_searching_flailing",
        maps_to="observe_only",
        literature="Pugliese et al. 2025: leg rhythms resembling searching, not walking",
        family="command_dn",
    ),
    PathwaySpec(
        name="oDN1",
        types=("DNg97",),
        aliases=("oDN1", "DNxl054"),
        role="bolt_forward_walk",
        maps_to="forward_locomotor_drive",
        literature="Sapkal et al. 2024 oDN1; FlyBase DNg97. Bolt-pathway descending node.",
        family="command_dn",
    ),
    PathwaySpec(
        name="DNa01",
        types=("DNa01",),
        aliases=(),
        role="ipsiversive_steering",
        maps_to="ipsiversive_steer",
        literature="Yang et al. 2023: DNa01 walking steering",
        family="command_dn",
    ),
    PathwaySpec(
        name="DNa02",
        types=("DNa02",),
        aliases=(),
        role="ipsiversive_steering",
        maps_to="ipsiversive_steer",
        literature="Rayshubskiy / Yang / Feng: DNa02 rotational velocity",
        family="command_dn",
    ),
    PathwaySpec(
        name="MDN",
        types=("MDN",),
        aliases=("moonwalker",),
        role="backward_walk",
        maps_to="backward_walk",
        literature="Bidaye et al. 2014: MDN backward walking",
        family="command_dn",
    ),
)

HALT_PATHWAYS: tuple[PathwaySpec, ...] = (
    PathwaySpec(
        name="foxglove",
        types=("CB0890",),
        aliases=("Foxglove", "FG"),
        role="walk_OFF_forward",
        maps_to="halt_observe",
        literature="Sapkal et al. 2024 Foxglove GABAergic walk-OFF; FlyWire CB0890. May be unlabeled in MaleCNS.",
        family="halt",
    ),
    PathwaySpec(
        name="bluebell",
        types=("DNg60",),
        aliases=("Bluebell", "BB", "mesa", "snail"),
        role="walk_OFF_turning",
        maps_to="halt_observe",
        literature="Sapkal et al. 2024 Bluebell; MaleCNS type DNg60",
        family="halt",
    ),
    PathwaySpec(
        name="brake",
        types=("AN19A018",),
        aliases=("Brake", "BRK", "AN_GNG_53", "AN_GNG_54", "AN_GNG_76"),
        role="brake_halt",
        maps_to="halt_observe",
        literature="Sapkal et al. 2024 Brake; FlyBase adult AN19A018 / BRK",
        family="halt",
    ),
)

CPG_INTERNEURONS: tuple[PathwaySpec, ...] = (
    PathwaySpec(
        name="E1",
        types=("IN17A001",),
        aliases=("E1",),
        role="cpg_excitatory",
        maps_to="observe_only",
        literature="Pugliese et al. 2025: E1 (IN17A001), one per leg neuropil",
        family="cpg",
    ),
    PathwaySpec(
        name="E2",
        types=("INXXX466",),
        aliases=("E2",),
        role="cpg_excitatory",
        maps_to="observe_only",
        literature="Pugliese et al. 2025 published: E2 (INXXX466). Not E5.",
        family="cpg",
    ),
    PathwaySpec(
        name="I1",
        types=("IN16B036",),
        aliases=("I1",),
        role="cpg_inhibitory",
        maps_to="observe_only",
        literature="Pugliese et al. 2025: I1 (IN16B036) in the MANC three-neuron core",
        family="cpg",
    ),
    PathwaySpec(
        name="E3",
        types=("IN19B012",),
        aliases=("E3",),
        role="cpg_excitatory_fanc",
        maps_to="observe_only",
        literature="Pugliese et al. 2025 FANC extra excitatory cell",
        family="cpg",
    ),
    PathwaySpec(
        name="E4",
        types=("IN03A006",),
        aliases=("E4",),
        role="dnb08_relay",
        maps_to="observe_only",
        literature="Pugliese et al. 2025: E4 (IN03A006) DNb08 relay onto E1",
        family="cpg",
    ),
    PathwaySpec(
        name="E5",
        types=("INXXX464",),
        aliases=("E5",),
        role="dnb08_relay",
        maps_to="observe_only",
        literature=(
            "Pugliese et al. 2025 published article: E5 = INXXX464. "
            "An older preprint passage appears to identify E5 as INXXX466 "
            "(published E2). Canonical mapping is INXXX464; the discrepancy "
            "is recorded, not used to pick whichever cell oscillates."
        ),
        family="cpg",
    ),
    PathwaySpec(
        name="I2",
        types=("IN19A007",),
        aliases=("I2",),
        role="cpg_inhibitory_alt",
        maps_to="observe_only",
        literature="Pugliese et al. 2025: I2 (IN19A007) in the DNb08 five-cell motif",
        family="cpg",
    ),
)

ALL_WALKING_SPECS: tuple[PathwaySpec, ...] = WALKING_DNS + HALT_PATHWAYS + CPG_INTERNEURONS

FORWARD_WALK_NAMES = ("DNp09", "DNg100", "oDN1")
ENGINEERED_CPG_STILL_EXECUTES = True
NEURAL_CPG_DRIVES_JOINTS = False

# Names only. Never `if DNg100: walk()`.
WALKING_CIRCUIT_TYPES = {
    "DNg100": "DNg100",
    "DNb08": "DNb08",
    "E1": "IN17A001",
    "E2": "INXXX466",
    "I1": "IN16B036",
    "I2": "IN19A007",
    "E3": "IN19B012",
    "E4": "IN03A006",
    "E5": "INXXX464",
}

E5_TYPE_PROVENANCE = {
    "canonical": "INXXX464",
    "source": "Pugliese et al. 2025 published article (PMC13142387)",
    "older_preprint_discrepancy": (
        "An older PDF/preprint passage appears to identify E5 as INXXX466, "
        "which is the published E2 type. Canonical mapping uses the published "
        "article (E5 = INXXX464). The discrepancy is recorded rather than "
        "silently picking whichever type makes a simulation oscillate."
    ),
    "rejected_alias": "INXXX466",
}

# One motif copy per leg neuropil. Do not average all E1 into one scalar.
LEG_SLOTS = ("FL", "FR", "ML", "MR", "HL", "HR")
LEG_LAYOUT = {
    "FL": ("L", "T1"),
    "FR": ("R", "T1"),
    "ML": ("L", "T2"),
    "MR": ("R", "T2"),
    "HL": ("L", "T3"),
    "HR": ("R", "T3"),
}
CPG_ROLES = ("E1", "E2", "I1", "I2", "E3", "E4", "E5")
# High somaLocation Z within a side is treated as anterior (T1). INFERRED.
SOMA_Z_ANTERIOR_IS_HIGH = True

# Documented MaleCNS v1.0 body IDs (uint64). Lookup is still by type.
EXPECTED_WALKING_BODY_IDS = {
    "DNp09": {"L": (10783,), "R": (11177,)},
    "DNg100": {"L": (10045,), "R": (10056,)},
    "DNb08": {"L": (12189, 12550), "R": (12044, 12075)},
    "oDN1": {"L": (13805,), "R": (230783,)},
    "DNa01": {"L": (10442,), "R": (10760,)},
    "DNa02": {"L": (523769,), "R": (10360,)},
    "MDN": {"L": (11288, 12348), "R": (10763, 11332)},
    "bluebell": {"L": (11374,), "R": (188947,)},
}


def _norm_side(value: object) -> str:
    text = str(value or "").strip().upper()
    if text in {"L", "LEFT", "LHS"}:
        return "L"
    if text in {"R", "RIGHT", "RHS"}:
        return "R"
    return ""


def lookup_exact_types(connectome: Connectome, types: tuple[str, ...]) -> np.ndarray:
    """Exact type matches only. Prefix matching would conflate DNg10 with DNg100."""
    chunks = []
    for typename in types:
        if not typename:
            continue
        idx = connectome.lookup(type_exact=typename)
        if idx.size:
            chunks.append(idx)
    if not chunks:
        return np.zeros(0, dtype=np.int32)
    return np.unique(np.concatenate(chunks)).astype(np.int32)


def _group_rate(counts: np.ndarray, indices: np.ndarray, duration_s: float) -> float:
    if indices.size == 0 or duration_s <= 0:
        return 0.0
    return float(counts[indices].sum()) / (duration_s * max(1, indices.size))


def _mean_analog(analog: np.ndarray | None, indices: np.ndarray) -> float:
    if analog is None or indices.size == 0:
        return 0.0
    return float(np.mean(analog[indices]))


def contacts_between(connectome: Connectome, pre: np.ndarray, post: np.ndarray) -> dict:
    pre = np.asarray(pre, dtype=np.int32)
    post = np.asarray(post, dtype=np.int32)
    if pre.size == 0 or post.size == 0:
        return {"n_edges": 0, "contacts": 0, "n_pre": int(pre.size), "n_post": int(post.size)}
    want = np.zeros(connectome.n, dtype=bool)
    want[post] = True
    n_edges = 0
    contacts = 0
    for i in pre.tolist():
        start = int(connectome.pre_ptr[i])
        end = int(connectome.pre_ptr[i + 1])
        if end <= start:
            continue
        mask = want[connectome.post[start:end]]
        if not np.any(mask):
            continue
        n_edges += int(mask.sum())
        contacts += int(connectome.anatomical[start:end][mask].sum())
    return {
        "n_edges": n_edges,
        "contacts": contacts,
        "n_pre": int(pre.size),
        "n_post": int(post.size),
    }


def contacts_pair(connectome: Connectome, pre_i: int, post_i: int) -> int:
    start = int(connectome.pre_ptr[int(pre_i)])
    end = int(connectome.pre_ptr[int(pre_i) + 1])
    if end <= start:
        return 0
    posts = connectome.post[start:end]
    mask = posts == np.int64(post_i)
    if not np.any(mask):
        return 0
    return int(connectome.anatomical[start:end][mask].sum())


def _try_soma_xyz() -> dict | None:
    try:
        from organism.soma import load_soma_xyz

        return load_soma_xyz()
    except (FileNotFoundError, RuntimeError, OSError, ValueError):
        return None


def _soma_z(connectome: Connectome, indices: list[int], soma: dict | None) -> np.ndarray:
    from organism.soma import lookup_xyz

    if soma is None or not indices:
        return np.full(len(indices), np.nan, dtype=np.float32)
    xyz = lookup_xyz(connectome.neuron_ids[np.asarray(indices, dtype=np.int32)], soma)
    return xyz[:, 2]


def _side_of(connectome: Connectome, index: int) -> str:
    return _norm_side(connectome.side[int(index)])


def _rank_ipsilateral(
    connectome: Connectome,
    indices: np.ndarray,
    side: str,
    soma: dict | None,
) -> list[int]:
    members = [int(i) for i in np.asarray(indices, dtype=np.int32).tolist() if _side_of(connectome, int(i)) == side]
    if not members:
        return []
    z = _soma_z(connectome, members, soma)
    if np.isfinite(z).all():
        order = np.argsort(-z) if SOMA_Z_ANTERIOR_IS_HIGH else np.argsort(z)
        return [members[int(i)] for i in order.tolist()]
    ids = [int(connectome.neuron_ids[i]) for i in members]
    return [m for _, m in sorted(zip(ids, members))]


def _assign_e1_slots(connectome: Connectome, e1: np.ndarray, soma: dict | None) -> dict[str, int | None]:
    slots: dict[str, int | None] = {name: None for name in LEG_SLOTS}
    for side, front, mid, hind in (("L", "FL", "ML", "HL"), ("R", "FR", "MR", "HR")):
        ranked = _rank_ipsilateral(connectome, e1, side, soma)
        for slot, idx in zip((front, mid, hind), ranked):
            slots[slot] = int(idx)
    return slots


def _assign_role_to_e1(
    connectome: Connectome,
    cells: np.ndarray,
    e1_by_slot: dict[str, int | None],
    soma: dict | None,
    side_slots: dict[str, tuple[str, ...]],
) -> dict[str, int | None]:
    assigned: dict[str, int | None] = {name: None for name in LEG_SLOTS}
    remaining = [int(i) for i in np.asarray(cells, dtype=np.int32).tolist()]
    scores: list[tuple[int, int, str]] = []
    for cell in remaining:
        for slot, e1 in e1_by_slot.items():
            if e1 is None:
                continue
            w = contacts_pair(connectome, cell, e1) + contacts_pair(connectome, e1, cell)
            if w > 0:
                scores.append((w, cell, slot))
    scores.sort(key=lambda row: row[0], reverse=True)
    used_cells: set[int] = set()
    used_slots: set[str] = set()
    for weight, cell, slot in scores:
        if cell in used_cells or slot in used_slots:
            continue
        assigned[slot] = cell
        used_cells.add(cell)
        used_slots.add(slot)
    leftovers = [c for c in remaining if c not in used_cells]
    if leftovers:
        for side, slots in side_slots.items():
            ranked = _rank_ipsilateral(connectome, np.asarray(leftovers, dtype=np.int32), side, soma)
            empty = [s for s in slots if assigned[s] is None]
            for slot, idx in zip(empty, ranked):
                assigned[slot] = int(idx)
                leftovers = [c for c in leftovers if c != idx]
    return assigned


def top_partners(connectome: Connectome, pre: np.ndarray, *, k: int = 12, min_contacts: int = 1) -> list[dict]:
    pre = np.asarray(pre, dtype=np.int32)
    if pre.size == 0:
        return []
    weights = np.zeros(connectome.n, dtype=np.int64)
    for i in pre.tolist():
        start = int(connectome.pre_ptr[i])
        end = int(connectome.pre_ptr[i + 1])
        if end <= start:
            continue
        posts = connectome.post[start:end]
        np.add.at(weights, posts, connectome.anatomical[start:end].astype(np.int64))
    order = np.argsort(weights)[::-1]
    rows = []
    for j in order.tolist():
        w = int(weights[j])
        if w < min_contacts:
            break
        rows.append(
            {
                "index": int(j),
                "body_id": int(connectome.neuron_ids[j]),
                "type": str(connectome.cell_type[j] or ""),
                "superclass": str(connectome.superclass[j] or ""),
                "side": _norm_side(connectome.side[j]),
                "contacts": w,
            }
        )
        if len(rows) >= k:
            break
    return rows


@dataclass
class ResolvedPathway:
    spec: PathwaySpec
    indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    resolved_types: tuple[str, ...] = ()

    @property
    def n(self) -> int:
        return int(self.indices.size)

    def as_dict(self, connectome: Connectome) -> dict:
        body_ids = [int(connectome.neuron_ids[i]) for i in self.indices[:16]]
        sides = [_norm_side(connectome.side[i]) for i in self.indices[:16]]
        superclasses = sorted({str(connectome.superclass[i] or "") for i in self.indices.tolist()})
        return {
            "name": self.spec.name,
            "types": list(self.spec.types),
            "aliases": list(self.spec.aliases),
            "role": self.spec.role,
            "maps_to": self.spec.maps_to,
            "literature": self.spec.literature,
            "family": self.spec.family,
            "n": self.n,
            "resolved": self.n > 0,
            "resolved_types": list(self.resolved_types),
            "body_ids": body_ids,
            "sides": sides,
            "superclasses": superclasses,
        }


class WalkingCircuit:
    """Resolved walking DNs + published VNC CPG types on one connectome."""

    def __init__(self, connectome: Connectome):
        self.connectome = connectome
        self.pathways: dict[str, ResolvedPathway] = {}
        for spec in ALL_WALKING_SPECS:
            idx = lookup_exact_types(connectome, spec.types)
            found = tuple(t for t in spec.types if connectome.lookup(type_exact=t).size)
            self.pathways[spec.name] = ResolvedPathway(spec=spec, indices=idx, resolved_types=found)
        self.vnc_motor = self._lookup_vnc_motor()
        self.vnc_premotor = self._lookup_vnc_premotor()
        self.forward_walk_indices = self._union(FORWARD_WALK_NAMES)
        self.halt_indices = self._union(("foxglove", "bluebell", "brake"))
        self.cpg_core_indices = self._union(("E1", "E2", "I1"))

    def _union(self, names: tuple[str, ...]) -> np.ndarray:
        chunks = [self.pathways[name].indices for name in names if self.pathways[name].n]
        if not chunks:
            return np.zeros(0, dtype=np.int32)
        return np.unique(np.concatenate(chunks)).astype(np.int32)

    def _lookup_vnc_motor(self) -> np.ndarray:
        chunks = [
            self.connectome.lookup(superclass="vnc_motor"),
            self.connectome.lookup(superclass="vnc_motor_neuron"),
            self.connectome.lookup(cell_class="motor"),
            self.connectome.lookup(type_prefix="MN"),
        ]
        chunks = [c for c in chunks if c.size]
        if not chunks:
            return np.zeros(0, dtype=np.int32)
        return np.unique(np.concatenate(chunks)).astype(np.int32)

    def _lookup_vnc_premotor(self) -> np.ndarray:
        chunks = [
            self.connectome.lookup(superclass="vnc_premotor"),
            self.connectome.lookup(superclass="vnc_intrinsic"),
            self.connectome.lookup(cell_class="premotor"),
        ]
        chunks = [c for c in chunks if c.size]
        if not chunks:
            return np.zeros(0, dtype=np.int32)
        return np.unique(np.concatenate(chunks)).astype(np.int32)

    def indices(self, name: str) -> np.ndarray:
        return self.pathways[name].indices

    def catalog(self) -> dict:
        return {name: p.as_dict(self.connectome) for name, p in self.pathways.items()}

    def anatomy(self) -> dict:
        dng = self.indices("DNg100")
        dnb = self.indices("DNb08")
        dnp = self.indices("DNp09")
        odn = self.indices("oDN1")
        e1 = self.indices("E1")
        e2 = self.indices("E2")
        i1 = self.indices("I1")
        e4 = self.indices("E4")
        bb = self.indices("bluebell")
        brk = self.indices("brake")
        fg = self.indices("foxglove")
        mn = self.vnc_motor
        dng100_to_e1 = contacts_between(self.connectome, dng, e1)
        return {
            "engineered_cpg_still_executes_joints": ENGINEERED_CPG_STILL_EXECUTES,
            "neural_vnc_cpg_drives_joints": False,
            "dng100_present": int(dng.size) > 0,
            "core_cpg_present": bool(e1.size and e2.size and i1.size),
            "dng100_to_E1": dng100_to_e1,
            "dng100_to_E2": contacts_between(self.connectome, dng, e2),
            "dng100_to_I1": contacts_between(self.connectome, dng, i1),
            "dng100_to_vnc_motor": contacts_between(self.connectome, dng, mn),
            "dnb08_to_E1": contacts_between(self.connectome, dnb, e1),
            "dnb08_to_E4": contacts_between(self.connectome, dnb, e4),
            "odn1_to_E1": contacts_between(self.connectome, odn, e1),
            "dnp09_to_E1": contacts_between(self.connectome, dnp, e1),
            "dnp09_to_dng100": contacts_between(self.connectome, dnp, dng),
            "bluebell_to_odn1": contacts_between(self.connectome, bb, odn),
            "bluebell_to_dng100": contacts_between(self.connectome, bb, dng),
            "brake_to_dng100": contacts_between(self.connectome, brk, dng),
            "foxglove_to_odn1": contacts_between(self.connectome, fg, odn),
            "E1_to_E2": contacts_between(self.connectome, e1, e2),
            "E2_to_E1": contacts_between(self.connectome, e2, e1),
            "E1_to_I1": contacts_between(self.connectome, e1, i1),
            "E2_to_I1": contacts_between(self.connectome, e2, i1),
            "I1_to_E1": contacts_between(self.connectome, i1, e1),
            "I1_to_E2": contacts_between(self.connectome, i1, e2),
            "cpg_core_to_vnc_motor": contacts_between(self.connectome, self.cpg_core_indices, mn),
            "dng100_top_partners": top_partners(self.connectome, dng, k=12),
            "notes": [
                "Anatomy is MEASURED synapse counts. Oscillation is not implied.",
                "Pugliese et al. 2025: DNg100 → E1 is the main walking-CPG entry.",
                "DNp09 does not have to synapse on E1; it can recruit DNg100.",
                "DNb08 is weakly onto E1 and stronger onto E4.",
                "Foxglove (CB0890) may be absent from MaleCNS type labels.",
            ],
        }

    def rates(self, counts: np.ndarray, duration_s: float, analog: np.ndarray | None = None) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for name, pathway in self.pathways.items():
            out[name] = {
                "hz": _group_rate(counts, pathway.indices, duration_s),
                "analog": _mean_analog(analog, pathway.indices),
                "n": pathway.n,
            }
        out["vnc_motor"] = {
            "hz": _group_rate(counts, self.vnc_motor, duration_s),
            "analog": _mean_analog(analog, self.vnc_motor),
            "n": int(self.vnc_motor.size),
        }
        out["forward_walk"] = {
            "hz": _group_rate(counts, self.forward_walk_indices, duration_s),
            "analog": _mean_analog(analog, self.forward_walk_indices),
            "n": int(self.forward_walk_indices.size),
        }
        return out
