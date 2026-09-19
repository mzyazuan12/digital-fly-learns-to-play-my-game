"""Map individual VNC cells onto T1/T2/T3 × L/R leg slots.

Each bodyId is assigned from *that cell's* LegNp PreSyn+PostSyn counts.
Type-level ROI pages pool the six segmental copies and must not be used.

Soma XYZ is not used. If the best neuropil is not dominant, the cell is
AMBIGUOUS. There is no silent soma-Z or somaNeuromere fallback.

DNg100 itself is two descending neurons. Soma side is contralateral to VNC
innervation. The six-copy types are E1/E2/I1 and the other named CPG cells.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

from flybrain.loader import ANN_FILE, DEFAULT_DATA, Connectome
from organism.roi_innervation import (
    CPG_TYPES as ROI_CPG_TYPES,
    JSON_CACHE,
    LEGNP_RE,
    MAPPING_PATH,
    PAPER_TO_INTERNAL,
    PARQUET_CACHE,
    assign_leg_from_row,
    load_cpg_mapping,
    load_roi_json,
    malecns_body_id_of,
    paper_slot_of,
    parse_legnp,
    rois_to_row,
)

ROOT = Path(__file__).resolve().parents[1]
ANN_PATH = DEFAULT_DATA / ANN_FILE
if not ANN_PATH.exists():
    ANN_PATH = ROOT / ANN_FILE
ROI_CACHE = JSON_CACHE

LEG_SLOTS = ("FL", "FR", "ML", "MR", "HL", "HR")
LEG_LAYOUT = {
    "FL": ("L", "T1"),
    "FR": ("R", "T1"),
    "ML": ("L", "T2"),
    "MR": ("R", "T2"),
    "HL": ("L", "T3"),
    "HR": ("R", "T3"),
}
SLOT_FROM_NEUROMERE = {value: key for key, value in LEG_LAYOUT.items()}
NEUROMERES = ("T1", "T2", "T3")
SIDES = ("L", "R")
CPG_TYPES = ROI_CPG_TYPES
DNG100_TYPE = "DNg100"


def _norm_side(value: object) -> str:
    text = str(value or "").strip().upper()
    if text in {"L", "LEFT", "LHS"}:
        return "L"
    if text in {"R", "RIGHT", "RHS"}:
        return "R"
    return ""


def _norm_neuromere(value: object) -> str:
    text = str(value or "").strip().upper()
    if text in NEUROMERES:
        return text
    return ""


def slot_of(side: str, neuromere: str) -> str | None:
    paper = paper_slot_of(side, neuromere)
    return PAPER_TO_INTERNAL.get(paper) if paper else None


@dataclass
class CellNeuropil:
    body_id: int
    cell_type: str
    side: str
    soma_neuromere: str
    roi_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    roi_slot: str | None = None
    mn_contacts: dict[str, int] = field(default_factory=dict)
    mn_slot: str | None = None
    assigned_slot: str | None = None
    assigned_leg: str | None = None
    confidence: float = 0.0
    second_best: str | None = None
    assignment_status: str = "unassigned"
    assignment_source: str = ""

    def as_dict(self) -> dict:
        return {
            "body_id": self.body_id,
            "type": self.cell_type,
            "side": self.side,
            "soma_neuromere": self.soma_neuromere,
            "roi_counts": self.roi_counts,
            "roi_slot": self.roi_slot,
            "mn_contacts": self.mn_contacts,
            "mn_slot": self.mn_slot,
            "assigned_slot": self.assigned_slot,
            "assigned_leg": self.assigned_leg,
            "confidence": self.confidence,
            "second_best": self.second_best,
            "assignment_status": self.assignment_status,
            "assignment_source": self.assignment_source,
            "soma_z_used": False,
            "fallback_used": False,
        }


def load_annotation_neuropil(*, types: tuple[str, ...] | None = None) -> dict[int, CellNeuropil]:
    """somaSide + somaNeuromere from the official MaleCNS annotation table."""
    if not ANN_PATH.exists():
        return {}
    wanted = set(types or (CPG_TYPES + (DNG100_TYPE,)))
    table = feather.read_table(
        ANN_PATH,
        columns=["bodyId", "type", "somaSide", "somaNeuromere", "superclass"],
    )
    frame = table.to_pandas()
    keep = frame["type"].isin(wanted)
    if "superclass" in frame:
        retain = frame["superclass"].notna() & frame["superclass"].astype(str).ne("")
        keep = keep & retain
    out: dict[int, CellNeuropil] = {}
    for row in frame.loc[keep].itertuples(index=False):
        body = int(row.bodyId)
        out[body] = CellNeuropil(
            body_id=body,
            cell_type=str(row.type or ""),
            side=_norm_side(getattr(row, "somaSide", "")),
            soma_neuromere=_norm_neuromere(getattr(row, "somaNeuromere", "")),
        )
    return out


def load_roi_cache() -> dict[int, dict[str, dict[str, int]]]:
    cached = load_roi_json()
    if cached:
        return cached
    if not ROI_CACHE.exists():
        return {}
    payload = json.loads(ROI_CACHE.read_text())
    bodies = payload.get("bodies") or payload
    out: dict[int, dict[str, dict[str, int]]] = {}
    for key, rois in bodies.items():
        if not isinstance(rois, dict):
            continue
        try:
            body = int(key)
        except (TypeError, ValueError):
            continue
        out[body] = {str(roi): {str(k): int(v) for k, v in counts.items()} for roi, counts in rois.items() if isinstance(counts, dict)}
    return out


def roi_slot_from_counts(rois: dict[str, dict[str, int]]) -> tuple[str | None, dict[str, int]]:
    """Per-bodyId LegNp pre+post → internal slot, or None if AMBIGUOUS."""
    assigned = assign_leg_from_row(rois_to_row(rois))
    scores = {PAPER_TO_INTERNAL[slot]: int(count) for slot, count in assigned["synapses_in"].items()}
    if assigned["status"] != "ok" or not assigned["assigned_slot"]:
        return None, scores
    return PAPER_TO_INTERNAL[assigned["assigned_slot"]], scores


def _unique_argmax(scores: dict[str, int], *, min_ratio: float = 1.15) -> str | None:
    if not scores:
        return None
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_key, best = ranked[0]
    if best <= 0:
        return None
    second = ranked[1][1] if len(ranked) > 1 else 0
    if second > 0 and best < min_ratio * second:
        return None
    return best_key


def mn_slot_from_contacts(contacts: dict[str, int]) -> tuple[str | None, dict[str, int]]:
    winner = _unique_argmax(contacts)
    return winner, contacts


def assign_cell(cell: CellNeuropil) -> CellNeuropil:
    """Assign from this bodyId's LegNp input synapses only.

    MN connectivity is a diagnostic, not a fallback. somaNeuromere / soma Z
    are never used to fill an AMBIGUOUS ROI.
    """
    assigned = assign_leg_from_row(rois_to_row(cell.roi_counts))
    paper_slot = assigned["assigned_slot"]
    cell.confidence = float(assigned["confidence"])
    cell.second_best = assigned["second_best"]
    cell.assigned_leg = assigned["assigned_leg"]
    cell.assignment_status = assigned["status"]
    cell.roi_slot = PAPER_TO_INTERNAL.get(paper_slot) if paper_slot else None
    mn_slot, _ = mn_slot_from_contacts(cell.mn_contacts)
    cell.mn_slot = mn_slot
    if assigned["status"] == "ok" and paper_slot:
        cell.assigned_slot = PAPER_TO_INTERNAL[paper_slot]
        cell.assignment_source = "roi_innervation_per_bodyId"
        return cell
    cell.assigned_slot = None
    cell.assignment_source = "AMBIGUOUS" if assigned["status"] == "AMBIGUOUS" else "unassigned"
    return cell


def attach_roi(cells: dict[int, CellNeuropil], rois: dict[int, dict[str, dict[str, int]]] | None = None) -> dict[int, CellNeuropil]:
    rois = rois if rois is not None else load_roi_cache()
    for body, cell in cells.items():
        cell.roi_counts = rois.get(body, {})
        assign_cell(cell)
    return cells


def motor_slot_table(connectome: Connectome, cells: dict[int, CellNeuropil]) -> dict[int, str]:
    """body_id → slot for vnc_motor cells using somaNeuromere/side annotations."""
    motor_bodies = set()
    for cell in cells.values():
        motor_bodies.add(cell.body_id)
    # Prefer the annotation table for all vnc_motor cells, not only CPG types.
    if not ANN_PATH.exists():
        return {}
    table = feather.read_table(
        ANN_PATH,
        columns=["bodyId", "somaSide", "somaNeuromere", "superclass"],
    ).to_pandas()
    motor = table[table["superclass"].astype(str).eq("vnc_motor")]
    out: dict[int, str] = {}
    for row in motor.itertuples(index=False):
        slot = slot_of(row.somaSide, row.somaNeuromere)
        if slot:
            out[int(row.bodyId)] = slot
    return out


def score_mn_contacts(connectome: Connectome, cells: dict[int, CellNeuropil]) -> dict[int, CellNeuropil]:
    if connectome.n < 1:
        return cells
    motor_slots = motor_slot_table(connectome, cells)
    if not motor_slots:
        for cell in cells.values():
            assign_cell(cell)
        return cells
    id_to_index = {int(body): i for i, body in enumerate(connectome.neuron_ids.tolist())}
    for body, cell in cells.items():
        if body not in id_to_index:
            assign_cell(cell)
            continue
        idx = id_to_index[body]
        start = int(connectome.pre_ptr[idx])
        end = int(connectome.pre_ptr[idx + 1])
        scores = {slot: 0 for slot in LEG_SLOTS}
        if end > start:
            posts = connectome.post[start:end]
            weights = connectome.anatomical[start:end]
            for post, weight in zip(posts.tolist(), weights.tolist()):
                target = int(connectome.neuron_ids[int(post)])
                slot = motor_slots.get(target)
                if slot:
                    scores[slot] += int(weight)
        cell.mn_contacts = scores
        assign_cell(cell)
    return cells


def apply_cpg_mapping(cells: dict[int, CellNeuropil], mapping: dict | None = None) -> dict[int, CellNeuropil]:
    """Fill assigned slots from the human-readable per-bodyId mapping file."""
    mapping = mapping if mapping is not None else load_cpg_mapping()
    inverse: dict[int, tuple[str, str]] = {}
    for typename in CPG_TYPES:
        block = mapping.get(typename) or {}
        for paper_slot, body in (block.get("neurons") or {}).items():
            malecns_id = malecns_body_id_of(body)
            if malecns_id is None:
                continue
            inverse[int(malecns_id)] = (typename, paper_slot)
    for body, cell in cells.items():
        hit = inverse.get(int(body))
        if hit is None:
            continue
        typename, paper_slot = hit
        cell.assigned_slot = PAPER_TO_INTERNAL.get(paper_slot, paper_slot)
        cell.assigned_leg = {"LF": "T1", "RF": "T1", "LM": "T2", "RM": "T2", "LH": "T3", "RH": "T3"}.get(paper_slot)
        record = ((mapping.get(typename) or {}).get("neurons") or {}).get(paper_slot) or {}
        if isinstance(record, dict):
            conf = record.get("assignment_confidence", record.get("confidence"))
            if conf is not None:
                cell.confidence = float(conf)
            if record.get("second_best"):
                cell.second_best = record["second_best"]
            if record.get("assigned_segment"):
                cell.assigned_leg = record["assigned_segment"]
        cell.assignment_status = "ok"
        cell.assignment_source = "cpg_mapping.json"
        cell.roi_slot = cell.assigned_slot
    return cells


def resolve_cpg_cells(connectome: Connectome | None = None) -> dict[int, CellNeuropil]:
    cells = load_annotation_neuropil()
    attach_roi(cells)
    if PARQUET_CACHE.exists() or MAPPING_PATH.exists():
        apply_cpg_mapping(cells)
    if connectome is not None and connectome.n >= 10_000:
        score_mn_contacts(connectome, cells)
        # Re-apply mapping after MN diagnostic so ROI/mapping still wins.
        if PARQUET_CACHE.exists() or MAPPING_PATH.exists():
            apply_cpg_mapping(cells)
    else:
        for cell in cells.values():
            if not cell.assigned_slot:
                assign_cell(cell)
    return cells


def assign_indices(
    connectome: Connectome,
    indices: np.ndarray,
    cells: dict[int, CellNeuropil] | None = None,
) -> tuple[dict[str, int | None], dict[str, CellNeuropil | None]]:
    """Map graph indices of one type onto the six leg slots. No soma XYZ / soma-Z."""
    cells = cells if cells is not None else resolve_cpg_cells(connectome)
    slots: dict[str, int | None] = {name: None for name in LEG_SLOTS}
    details: dict[str, CellNeuropil | None] = {name: None for name in LEG_SLOTS}
    leftovers: list[int] = []
    for raw in np.asarray(indices, dtype=np.int32).tolist():
        idx = int(raw)
        body = int(connectome.neuron_ids[idx])
        cell = cells.get(body)
        if cell is None:
            cell = CellNeuropil(
                body_id=body,
                cell_type=str(connectome.cell_type[idx] or ""),
                side=_norm_side(connectome.side[idx]),
                soma_neuromere="",
            )
            assign_cell(cell)
        slot = cell.assigned_slot
        if slot and slots[slot] is None:
            slots[slot] = idx
            details[slot] = cell
        else:
            leftovers.append(idx)
    if leftovers:
        _toy_or_ipsilateral_t1(connectome, leftovers, slots, details)
    return slots, details


def _toy_or_ipsilateral_t1(
    connectome: Connectome,
    leftovers: list[int],
    slots: dict[str, int | None],
    details: dict[str, CellNeuropil | None],
) -> None:
    """Miniature graphs have one ipsilateral copy: treat it as T1 of that side."""
    by_side: dict[str, list[int]] = {"L": [], "R": []}
    for idx in leftovers:
        side = _norm_side(connectome.side[int(idx)])
        if side in by_side:
            by_side[side].append(int(idx))
    for side, members in by_side.items():
        if len(members) != 1:
            continue
        slot = slot_of(side, "T1")
        if slot and slots[slot] is None:
            idx = members[0]
            slots[slot] = idx
            details[slot] = CellNeuropil(
                body_id=int(connectome.neuron_ids[idx]),
                cell_type=str(connectome.cell_type[idx] or ""),
                side=side,
                soma_neuromere="T1",
                assigned_slot=slot,
                assignment_source="toy_single_ipsilateral_as_T1",
            )
