"""Map individual VNC cells onto T1/T2/T3 × L/R leg slots.

Soma XYZ rank is not used. MaleCNS already annotates these CPG types across
the three leg neuropils (somaNeuromere T1/T2/T3 and somaSide L/R). Per-cell
LegNp ROI innervation and connectivity onto motor neurons of each neuromere
are independent checks of that annotation.

DNg100 itself is two descending neurons (one per side), not six CPG copies.
The six-copy types are E1/E2/I1 and the other named CPG interneurons.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

from flybrain.loader import ANN_FILE, DEFAULT_DATA, Connectome

ROOT = Path(__file__).resolve().parents[1]
ANN_PATH = DEFAULT_DATA / ANN_FILE
if not ANN_PATH.exists():
    ANN_PATH = ROOT / ANN_FILE
ROI_CACHE = DEFAULT_DATA / "normalized" / "cpg_leg_roi.json"
NEUROMERE_CACHE = DEFAULT_DATA / "normalized" / "leg_neuropil.npz"

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
CPG_TYPES = (
    "IN17A001",
    "INXXX466",
    "IN16B036",
    "IN19A007",
    "IN19B012",
    "IN03A006",
    "INXXX464",
)
DNG100_TYPE = "DNg100"
LEGNP_RE = re.compile(r"LegNp\((T[123])\)\(([LR])\)")


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


def parse_legnp(roi: str) -> tuple[str, str] | None:
    match = LEGNP_RE.search(str(roi or ""))
    if not match:
        return None
    return match.group(1), match.group(2)


def slot_of(side: str, neuromere: str) -> str | None:
    return SLOT_FROM_NEUROMERE.get((_norm_side(side), _norm_neuromere(neuromere)))


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
            "assignment_source": self.assignment_source,
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
    if not ROI_CACHE.exists():
        return {}
    payload = json.loads(ROI_CACHE.read_text())
    bodies = payload.get("bodies") or payload
    out: dict[int, dict[str, dict[str, int]]] = {}
    for key, rois in bodies.items():
        out[int(key)] = {str(roi): {str(k): int(v) for k, v in counts.items()} for roi, counts in rois.items()}
    return out


def roi_slot_from_counts(rois: dict[str, dict[str, int]]) -> tuple[str | None, dict[str, int]]:
    scores = {slot: 0 for slot in LEG_SLOTS}
    for name, counts in rois.items():
        parsed = parse_legnp(name)
        if parsed is None:
            continue
        neuromere, side = parsed
        slot = slot_of(side, neuromere)
        if slot is None:
            continue
        scores[slot] += int(counts.get("pre", 0)) + int(counts.get("post", 0))
    winner = _unique_argmax(scores)
    return winner, scores


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
    """Prefer ROI innervation, then MN connectivity, then somaNeuromere annotation."""
    roi_slot, _ = roi_slot_from_counts(cell.roi_counts)
    cell.roi_slot = roi_slot
    mn_slot, _ = mn_slot_from_contacts(cell.mn_contacts)
    cell.mn_slot = mn_slot
    annotated = slot_of(cell.side, cell.soma_neuromere)
    if roi_slot:
        cell.assigned_slot = roi_slot
        cell.assignment_source = "roi_innervation"
        return cell
    if mn_slot:
        cell.assigned_slot = mn_slot
        cell.assignment_source = "mn_connectivity"
        return cell
    if annotated:
        cell.assigned_slot = annotated
        cell.assignment_source = "somaNeuromere_annotation"
        return cell
    cell.assigned_slot = None
    cell.assignment_source = "unassigned"
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


def resolve_cpg_cells(connectome: Connectome | None = None) -> dict[int, CellNeuropil]:
    cells = load_annotation_neuropil()
    attach_roi(cells)
    if connectome is not None and connectome.n >= 10_000:
        score_mn_contacts(connectome, cells)
    else:
        for cell in cells.values():
            assign_cell(cell)
    return cells


def assign_indices(
    connectome: Connectome,
    indices: np.ndarray,
    cells: dict[int, CellNeuropil] | None = None,
) -> tuple[dict[str, int | None], dict[str, CellNeuropil | None]]:
    """Map graph indices of one type onto the six leg slots. No soma XYZ."""
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
