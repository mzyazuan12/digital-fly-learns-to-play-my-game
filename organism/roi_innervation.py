"""Per-bodyId MaleCNS LegNp innervation. Never type-level ROI totals.

Identity comes only from the official annotation feather:

  body-annotations-male-cns-v1.0-minconf-0.5.feather
  column bodyId  (not the NT table's `body`, not a derived parquet)

Syn-points use column `body`. Rename both to malecns_body_id before joining.

Pugliese DNg100_Stim is MANC_T1 matrix index 31 / MANC body 10093.
That is recorded only under pugliese_reference. MaleCNS DNg100 is
annotations[type == "DNg100"]. A curated mancBodyid/mancType field, if
present, is stored separately as correspondence — not as identity.

Soma XYZ / somaNeuromere are never a fallback.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.feather as feather
import pyarrow.parquet as pq

from flybrain.loader import ANN_FILE, DEFAULT_DATA, NT_FILE

ROOT = Path(__file__).resolve().parents[1]
ANN_PATH = DEFAULT_DATA / ANN_FILE
if not ANN_PATH.exists():
    ANN_PATH = ROOT / ANN_FILE
NT_PATH = DEFAULT_DATA / NT_FILE
if not NT_PATH.exists():
    NT_PATH = ROOT / NT_FILE

JSON_CACHE = DEFAULT_DATA / "normalized" / "cpg_leg_roi.json"
PARQUET_CACHE = DEFAULT_DATA / "cpg_roi_innervation.parquet"
LEGACY_PARQUET_CACHE = DEFAULT_DATA / "neuron_roi_innervation.parquet"
MAPPING_PATH = DEFAULT_DATA / "cpg_mapping.json"
INVALID_MAPPING_PATH = DEFAULT_DATA / "cpg_mapping.INVALID_pre_roi.json"
INVALID_NAMESPACE_MAPPING_PATH = DEFAULT_DATA / "cpg_mapping.INVALID_pre_namespace_fix.json"
INVALID_NAMESPACE_PARQUET_PATH = DEFAULT_DATA / "male_cpg.INVALID_pre_namespace_fix.parquet"
METADATA_PATH = DEFAULT_DATA / "neuron_metadata.parquet"
SYN_POINTS = DEFAULT_DATA / "syn-points-male-cns-v1.0-minconf-0.5.feather"
PUGLIESE_MANC_T1_TABLE = (
    ROOT
    / "third_party"
    / "Pugliese_2026"
    / "data"
    / "manc t1 connectome data"
    / "wTable_20250813_DNtoMN_unsorted_withModules.csv"
)

LEGNP_RE = re.compile(r"LegNp\((T[123])\)\(([LR])\)")
NEUROMERES = ("T1", "T2", "T3")
SIDES = ("L", "R")
ROI_COLUMNS = tuple(f"{seg}_{side}_{kind}" for seg in NEUROMERES for side in SIDES for kind in ("pre", "post"))
ASSIGNMENT_METHOD = "LegNp synaptic innervation"
DNG100_ASSIGNMENT_METHOD = "MaleCNS annotation type == DNg100"
FORBIDDEN_KEYS = {"id", "dng100_body_id"}
PUGLIESE_MANC_STIM_INDEX = 31
PUGLIESE_MANC_STIM_BODY = 10093
MALECNS_DNG100_BODY_IDS = {10045, 10056}

PAPER_SLOTS = ("LF", "RF", "LM", "RM", "LH", "RH")
PAPER_LAYOUT = {
    "LF": ("L", "T1"),
    "RF": ("R", "T1"),
    "LM": ("L", "T2"),
    "RM": ("R", "T2"),
    "LH": ("L", "T3"),
    "RH": ("R", "T3"),
}
PAPER_TO_INTERNAL = {"LF": "FL", "RF": "FR", "LM": "ML", "RM": "MR", "LH": "HL", "RH": "HR"}
INTERNAL_TO_PAPER = {value: key for key, value in PAPER_TO_INTERNAL.items()}

# Published core motif. Rebuild identity from these types + DNg100, never from parquet.
CORE_CPG_TYPES = {
    "E1": "IN17A001",
    "E2": "INXXX466",
    "E3": "IN19B012",
    "I1": "IN16B036",
    "I2": "IN19A007",
}
EXTENDED_CPG_TYPES = {
    "E4": "IN03A006",
    "E5": "INXXX464",
}
CPG_TYPES = tuple(CORE_CPG_TYPES[role] for role in ("E1", "E2", "I1", "I2", "E3")) + tuple(
    EXTENDED_CPG_TYPES[role] for role in ("E4", "E5")
)
ROLE_FOR_TYPE = {
    "DNg100": "walking_command",
    "IN17A001": "E1",
    "INXXX466": "E2",
    "IN16B036": "I1",
    "IN19A007": "I2",
    "IN19B012": "E3",
    "IN03A006": "E4",
    "INXXX464": "E5",
}
TYPE_FOR_ROLE = {role: typename for typename, role in ROLE_FOR_TYPE.items() if role != "walking_command"}
TYPE_FOR_ROLE["DNg100"] = "DNg100"
EXPECTED_COUNTS = {
    "IN17A001": 6,
    "INXXX466": 6,
    "IN19B012": 6,
    "IN16B036": 6,
    "IN19A007": 6,
}
EXPECTED_COPIES = {
    **EXPECTED_COUNTS,
    "IN03A006": 6,
    "INXXX464": 6,
    "DNg100": 2,
}

I2_TYPE_PROVENANCE = {
    "canonical": "IN19A007",
    "role": "I2",
    "rejected_alias": "IN19B007",
    "source": (
        "Pugliese et al. 2025 (PMC13142387): E1=IN17A001, E2=INXXX466, "
        "E3=IN19B012, I1=IN16B036, I2=IN19A007"
    ),
    "note": (
        "I2 is IN19A007: six T1/T2/T3 copies, predicted GABAergic. "
        "IN19B007 exists in MaleCNS (two neurons) and was an incorrect earlier I2 assignment."
    ),
}
MALE_CNS_DATASETS = {"MaleCNS_v1.0", "MaleCNS_v1"}
MANC_DATASETS = {"MANC_T1", "MANC"}

# Best neuromere / (T1+T2+T3). Below this → AMBIGUOUS.
CONFIDENCE_THRESHOLD = 0.70

def _manc_record(
    *,
    source_matrix_index: int,
    source_body_id: int,
    type: str,
    predicted_nt: str = "",
    note: str | None = None,
) -> dict:
    """MANC-only identity. Never put these integers in malecns_body_id."""
    rec = {
        "source_dataset": "MANC_T1",
        "source_matrix_index": int(source_matrix_index),
        "source_body_id": int(source_body_id),
        "type": type,
        "predicted_nt": predicted_nt,
        # Legacy aliases for older tests; mapping JSON uses the source_* keys.
        "dataset": "MANC_T1",
        "matrix_index": int(source_matrix_index),
        "body_id": int(source_body_id),
    }
    if note:
        rec["note"] = note
    return rec


def namespaced_manc_only(record: dict) -> dict:
    """Drop legacy aliases so mapping JSON cannot be read as MaleCNS."""
    out = {
        "source_dataset": "MANC_T1",
        "source_matrix_index": int(record.get("source_matrix_index", record.get("matrix_index"))),
        "source_body_id": int(record.get("source_body_id", record.get("body_id"))),
        "type": record.get("type"),
    }
    if record.get("predicted_nt"):
        out["predicted_nt"] = record["predicted_nt"]
    if record.get("note"):
        out["note"] = record["note"]
    return out


# Pugliese configs/experiment/DNg100_Stim.yaml: stimNeurons: [[31]]
PUGLIESE_DNG100_STIM = _manc_record(
    source_matrix_index=PUGLIESE_MANC_STIM_INDEX,
    source_body_id=PUGLIESE_MANC_STIM_BODY,
    type="DNg100",
    predicted_nt="acetylcholine",
)
MANC_VMS16 = _manc_record(
    source_matrix_index=16,
    source_body_id=10056,
    type="vMS16",
    predicted_nt="gaba",
    note=(
        "Pugliese MANC T1 table row 16 is vMS16, not DNg100. MaleCNS annotations "
        "independently give bodyId 10056 type DNg100 (DNg100_R). Same integer, "
        "different reconstructed specimen/dataset. Do not join on the integer. "
        "MaleCNS DNg100_R's curated mancBodyid/mancType is correspondence to "
        "MANC DNg100 10093, not to this vMS16 row."
    ),
)
# Kept so existing imports keep working. Same object as MANC_VMS16.
MANC_NOT_DNG100_10056 = MANC_VMS16


def namespaced_ids(
    *,
    source_dataset: str,
    source_matrix_index: int | None = None,
    source_body_id: int | None = None,
    malecns_body_id: int | None = None,
) -> dict:
    """Four ID fields. There is no generic `id` or `dng100_body_id`."""
    if source_dataset not in {"MANC_T1", "MaleCNS_v1", "MaleCNS_v1.0"}:
        raise ValueError(f"unknown source_dataset {source_dataset!r}")
    return {
        "source_dataset": source_dataset,
        "source_matrix_index": source_matrix_index,
        "source_body_id": source_body_id,
        "malecns_body_id": None if malecns_body_id is None else int(malecns_body_id),
    }


def malecns_body_id_of(entry: object) -> int | None:
    """Read a MaleCNS body ID. Never treat a MANC record as MaleCNS."""
    if entry is None:
        return None
    if isinstance(entry, dict):
        for forbidden in FORBIDDEN_KEYS:
            if forbidden in entry:
                raise ValueError(f"generic field {forbidden!r} is forbidden")
        dataset = str(entry.get("source_dataset") or entry.get("dataset") or "")
        if dataset in {"MANC_T1", "MANC"}:
            return None
        if entry.get("malecns_body_id") is not None:
            return int(entry["malecns_body_id"])
        return None
    return int(entry)


def _norm_side(value: object) -> str:
    text = str(value or "").strip().upper()
    if text in {"L", "LEFT", "LHS"}:
        return "L"
    if text in {"R", "RIGHT", "RHS"}:
        return "R"
    return ""


def _norm_neuromere(value: object) -> str:
    text = str(value or "").strip().upper()
    return text if text in NEUROMERES else ""


def parse_legnp(roi: str) -> tuple[str, str] | None:
    match = LEGNP_RE.search(str(roi or ""))
    if not match:
        return None
    return match.group(1), match.group(2)


def paper_slot_of(side: str, neuromere: str) -> str | None:
    side = _norm_side(side)
    neuromere = _norm_neuromere(neuromere)
    for slot, (slot_side, slot_seg) in PAPER_LAYOUT.items():
        if slot_side == side and slot_seg == neuromere:
            return slot
    return None


def empty_roi_row() -> dict[str, int]:
    return {name: 0 for name in ROI_COLUMNS}


def rois_to_row(rois: dict[str, dict[str, int]]) -> dict[str, int]:
    row = empty_roi_row()
    for name, counts in (rois or {}).items():
        parsed = parse_legnp(name)
        if parsed is None:
            continue
        neuromere, side = parsed
        row[f"{neuromere}_{side}_pre"] += int(counts.get("pre", 0) or 0)
        row[f"{neuromere}_{side}_post"] += int(counts.get("post", 0) or 0)
    return row


def _kind_to_polarity(kind: object) -> str | None:
    text = str(kind or "").lower().replace("_", "").replace("-", "")
    if text in {"presyn", "pre", "tbar", "presynaptic"}:
        return "pre"
    if text in {"postsyn", "post", "psd", "postsynaptic"}:
        return "post"
    return None


def synapses_in_by_slot(row: dict[str, int]) -> dict[str, int]:
    """Input synapses (PostSyn) in each of the six LegNp ROIs. Provenance only."""
    out = {}
    for slot, (side, neuromere) in PAPER_LAYOUT.items():
        out[slot] = int(row.get(f"{neuromere}_{side}_post", 0) or 0)
    return out


def synapses_out_by_slot(row: dict[str, int]) -> dict[str, int]:
    out = {}
    for slot, (side, neuromere) in PAPER_LAYOUT.items():
        out[slot] = int(row.get(f"{neuromere}_{side}_pre", 0) or 0)
    return out


def neuromere_scores(row: dict[str, int]) -> dict[str, int]:
    """T1/T2/T3 = PreSyn + PostSyn, pooled across L/R."""
    return {
        neuromere: sum(
            int(row.get(f"{neuromere}_{side}_{kind}", 0) or 0) for side in SIDES for kind in ("pre", "post")
        )
        for neuromere in NEUROMERES
    }


def neuromere_polarity_counts(row: dict[str, int]) -> dict[str, dict[str, int]]:
    """User-facing ROI block: T1/T2/T3 × {pre, post}, L+R pooled."""
    return {
        neuromere: {
            "pre": int(row.get(f"{neuromere}_L_pre", 0) or 0) + int(row.get(f"{neuromere}_R_pre", 0) or 0),
            "post": int(row.get(f"{neuromere}_L_post", 0) or 0) + int(row.get(f"{neuromere}_R_post", 0) or 0),
        }
        for neuromere in NEUROMERES
    }


def innervation_side_of(row: dict[str, int]) -> str:
    left = sum(
        int(row.get(f"{neuromere}_L_{kind}", 0) or 0) for neuromere in NEUROMERES for kind in ("pre", "post")
    )
    right = sum(
        int(row.get(f"{neuromere}_R_{kind}", 0) or 0) for neuromere in NEUROMERES for kind in ("pre", "post")
    )
    if left > right:
        return "L"
    if right > left:
        return "R"
    return ""


def side_scores(row: dict[str, int], neuromere: str) -> dict[str, int]:
    return {
        side: int(row.get(f"{neuromere}_{side}_pre", 0) or 0) + int(row.get(f"{neuromere}_{side}_post", 0) or 0)
        for side in SIDES
    }


def assign_leg_from_row(row: dict[str, int]) -> dict:
    """Map one bodyId onto a walking slot from that cell's LegNp pre+post counts.

    T#_score = pre_T# + post_T# (L+R). Winner neuromere, then L vs R inside it.
    Low-confidence winners are AMBIGUOUS. Soma Z / somaNeuromere are never consulted.
    """
    t_scores = neuromere_scores(row)
    ranked_t = sorted(t_scores.items(), key=lambda item: item[1], reverse=True)
    best_t, best = ranked_t[0]
    second_t, second = ranked_t[1]
    total = int(sum(t_scores.values()))
    confidence = float(best / total) if total > 0 else 0.0
    sides = side_scores(row, best_t) if best > 0 else {"L": 0, "R": 0}
    ranked_side = sorted(sides.items(), key=lambda item: item[1], reverse=True)
    best_side, side_best = ranked_side[0]
    _other_side, side_second = ranked_side[1]
    side_total = int(side_best + side_second)
    side_confidence = float(side_best / side_total) if side_total > 0 else 0.0
    t_ok = total > 0 and best > 0 and confidence >= CONFIDENCE_THRESHOLD
    side_ok = t_ok and side_best > 0 and side_confidence >= CONFIDENCE_THRESHOLD
    if t_ok and side_ok:
        status = "ok"
        assigned_segment = best_t
        assigned_side = best_side
        slot = paper_slot_of(best_side, best_t)
    elif t_ok:
        status = "AMBIGUOUS"
        assigned_segment = best_t
        assigned_side = None
        slot = None
    else:
        status = "AMBIGUOUS"
        assigned_segment = None
        assigned_side = None
        slot = None
    return {
        "assigned_slot": slot,
        "assigned_segment": assigned_segment,
        "assigned_leg": assigned_segment,
        "assigned_side": assigned_side,
        "confidence": confidence,
        "assignment_confidence": confidence,
        "side_confidence": side_confidence,
        "second_best": second_t if second > 0 else None,
        "second_best_count": int(second),
        "best_count": int(best),
        "total_legnp": total,
        "status": status,
        "assignment_status": status,
        "assignment_method": ASSIGNMENT_METHOD,
        "fallback_used": False,
        "t_scores": t_scores,
        "side_scores": sides,
        "roi_counts": neuromere_polarity_counts(row),
        "synapses_in": synapses_in_by_slot(row),
        "synapses_out": synapses_out_by_slot(row),
        "soma_z_used": False,
    }


def cpg_wanted_types() -> set[str]:
    return {"DNg100", *CORE_CPG_TYPES.values(), *EXTENDED_CPG_TYPES.values()}


def load_raw_annotation_table():
    """Official MaleCNS annotation feather. Not parquet. Not the NT table."""
    import pandas as pd

    if not ANN_PATH.exists():
        raise FileNotFoundError(ANN_PATH)
    name = ANN_PATH.name.lower()
    if ANN_PATH.suffix == ".parquet" or "neurotransmitter" in name or "syn-points" in name:
        raise AssertionError(
            f"refusing to treat {ANN_PATH} as MaleCNS annotations. "
            "Identity is body-annotations-*.feather column bodyId."
        )
    raw = pd.read_feather(ANN_PATH)
    if "bodyId" not in raw.columns:
        raise AssertionError(
            f"bodyId missing from {ANN_PATH}; columns={list(raw.columns)}; "
            f"index.name={raw.index.name}. Wrong DataFrame: the NT table and "
            "syn-points use column 'body', not 'bodyId'. Do not use the index."
        )
    assert_expected_annotation_counts(raw)
    if CORE_CPG_TYPES["I2"] != "IN19A007":
        raise AssertionError("I2 must be IN19A007")
    if "IN19B007" in CORE_CPG_TYPES.values():
        raise AssertionError("IN19B007 is not I2")
    return raw


def assert_expected_annotation_counts(raw) -> None:
    """Stop the mapping build rather than guessing if a core type has the wrong copy count."""
    if CORE_CPG_TYPES["I2"] != "IN19A007":
        raise AssertionError("I2 must be IN19A007")
    if "IN19B007" in CORE_CPG_TYPES.values():
        raise AssertionError("IN19B007 is not I2")
    types = raw["type"].astype(str).str.strip()
    dng_ids = set(int(v) for v in raw.loc[types.eq("DNg100"), "bodyId"])
    if dng_ids != MALECNS_DNG100_BODY_IDS:
        raise AssertionError(
            f"MaleCNS DNg100 bodyIds {dng_ids} != {MALECNS_DNG100_BODY_IDS}"
        )
    for cell_type, expected in EXPECTED_COUNTS.items():
        found = raw.loc[types.eq(cell_type)]
        if len(found) != expected:
            raise AssertionError(f"{cell_type}: expected {expected}, got {len(found)}")


def raw_dng100_annotation_rows():
    """The only authoritative MaleCNS DNg100 lookup: type == DNg100 on the feather."""
    raw = load_raw_annotation_table()
    cols = [c for c in ("bodyId", "type", "somaSide", "rootSide", "mancBodyid", "mancType", "instance") if c in raw.columns]
    return raw.loc[raw["type"].astype(str).str.strip().eq("DNg100"), cols].copy()


def malecns_dng100_body_ids() -> dict[str, int]:
    """L/R malecns_body_id from annotations[type == DNg100]. Nothing else."""
    dng = raw_dng100_annotation_rows()
    out: dict[str, int] = {}
    for rec in dng.itertuples(index=False):
        side = _norm_side(getattr(rec, "somaSide", ""))
        if not side:
            instance = str(getattr(rec, "instance", "") or "")
            if instance.endswith("_L"):
                side = "L"
            elif instance.endswith("_R"):
                side = "R"
        if side not in SIDES:
            raise AssertionError(f"DNg100 bodyId {int(rec.bodyId)} has no L/R side")
        if side in out:
            raise AssertionError(f"duplicate MaleCNS DNg100 side {side}")
        out[side] = int(rec.bodyId)
    if set(out) != {"L", "R"}:
        raise AssertionError(f"MaleCNS DNg100 sides {out} != {{L, R}}")
    return out


def lookup_malecns(body_id: int) -> dict:
    """One MaleCNS annotation row. bodyId is only unique inside MaleCNS_v1.0."""
    import pandas as pd

    raw = load_raw_annotation_table()
    hit = raw.loc[raw["bodyId"] == int(body_id)]
    if len(hit) != 1:
        raise KeyError(f"MaleCNS bodyId {body_id} n={len(hit)}")
    row = hit.iloc[0]
    manc = row.get("mancBodyid")
    manc_id = None if manc is None or pd.isna(manc) else int(manc)
    return {
        "source_dataset": "MaleCNS_v1.0",
        "malecns_body_id": int(body_id),
        "type": str(row["type"] or "").strip(),
        "side": _norm_side(row.get("somaSide")),
        "instance": str(row.get("instance") or ""),
        "manc_body_id": manc_id,
        "manc_type": str(row.get("mancType") or "") or None,
    }


def lookup_manc(body_id: int) -> dict:
    """One Pugliese MANC T1 table row. bodyId is only unique inside MANC_T1."""
    import pandas as pd

    if not PUGLIESE_MANC_T1_TABLE.exists():
        raise FileNotFoundError(PUGLIESE_MANC_T1_TABLE)
    table = pd.read_csv(PUGLIESE_MANC_T1_TABLE)
    hit = table.loc[table["bodyId"] == int(body_id)]
    if hit.empty:
        raise KeyError(f"MANC T1 bodyId {body_id} n=0")
    row = hit.iloc[0]
    return {
        "source_dataset": "MANC_T1",
        "source_matrix_index": int(row.name),
        "source_body_id": int(body_id),
        "type": str(row.get("type") or "").strip(),
        "predicted_nt": str(row.get("predictedNt") or ""),
    }


def assert_every_body_id_has_dataset(obj: object, *, path: str = "$") -> None:
    """A body ID integer is meaningless without source_dataset."""
    if isinstance(obj, dict):
        if "body_id" in obj:
            raise ValueError(
                f"{path}: generic body_id is forbidden in mapping JSON; "
                "use malecns_body_id with source_dataset=MaleCNS_v1.0 or "
                "source_body_id with source_dataset=MANC_T1"
            )
        if obj.get("malecns_body_id") is not None:
            if obj.get("source_dataset") not in MALE_CNS_DATASETS:
                raise ValueError(f"{path}: malecns_body_id requires source_dataset MaleCNS_v1.0")
        if obj.get("source_body_id") is not None:
            if obj.get("source_dataset") not in MANC_DATASETS:
                raise ValueError(f"{path}: source_body_id requires source_dataset MANC_T1")
        for key, value in obj.items():
            assert_every_body_id_has_dataset(value, path=f"{path}.{key}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            assert_every_body_id_has_dataset(item, path=f"{path}[{i}]")


def load_annotations(types: tuple[str, ...] | None = None) -> list[dict]:
    """MaleCNS identity from raw annotations[bodyId, type, ...]. ROI is joined later."""
    import pandas as pd

    raw = load_raw_annotation_table()
    wanted = set(types) if types is not None else cpg_wanted_types()
    keep = raw["type"].astype(str).str.strip().isin(wanted)
    cells = raw.loc[keep].copy()
    cells = cells.rename(columns={"bodyId": "malecns_body_id"})
    rows = []
    for rec in cells.itertuples(index=False):
        manc = getattr(rec, "mancBodyid", None)
        manc_id = None if manc is None or (isinstance(manc, float) and pd.isna(manc)) else int(manc)
        rows.append(
            {
                "source_dataset": "MaleCNS_v1.0",
                "malecns_body_id": int(rec.malecns_body_id),
                "type": str(rec.type or "").strip(),
                "instance": str(getattr(rec, "instance", "") or ""),
                "side": _norm_side(getattr(rec, "somaSide", "")),
                "root_side": _norm_side(getattr(rec, "rootSide", "")),
                "soma_neuromere": _norm_neuromere(getattr(rec, "somaNeuromere", "")),
                "superclass": str(getattr(rec, "superclass", "") or ""),
                "manc_body_id": manc_id,
                "manc_type": str(getattr(rec, "mancType", "") or "") or None,
            }
        )
    return rows


def annotation_synpoint_rows(types: tuple[str, ...] | None = None) -> list[dict]:
    """Join raw annotation bodyId onto syn-points ROI counts.

    Never reads the derived CPG parquet for identity. syn-points column is
    `body`; annotations column is `bodyId`; both become malecns_body_id.
    """
    anns = load_annotations(types)
    rois = load_roi_json()
    missing = [row["malecns_body_id"] for row in anns if row["malecns_body_id"] not in rois]
    if missing and not rois:
        raise FileNotFoundError(
            f"{JSON_CACHE} missing; scan syn-points first: "
            "python scripts/cache_malecns_roi_innervation.py --from-syn-points --write-mapping"
        )
    records = []
    for ann in anns:
        row = rois_to_row(rois.get(ann["malecns_body_id"], {}))
        records.append({**ann, **row})
    return records


def load_predicted_nt(body_ids: set[int] | None = None) -> dict[int, dict]:
    if not NT_PATH.exists():
        return {}
    table = feather.read_table(NT_PATH)
    frame = table.to_pandas()
    body_col = "body" if "body" in frame.columns else "bodyId"
    wanted = None if body_ids is None else {int(b) for b in body_ids}
    out: dict[int, dict] = {}
    for rec in frame.itertuples(index=False):
        body = int(getattr(rec, body_col))
        if wanted is not None and body not in wanted:
            continue
        out[body] = {
            "predicted_nt": str(getattr(rec, "predicted_nt", "") or getattr(rec, "consensus_nt", "") or ""),
            "predicted_nt_confidence": float(getattr(rec, "predicted_nt_confidence", 0.0) or 0.0),
            "celltype_predicted_nt": str(getattr(rec, "celltype_predicted_nt", "") or ""),
        }
    return out


def load_roi_json() -> dict[int, dict[str, dict[str, int]]]:
    if not JSON_CACHE.exists():
        return {}
    payload = json.loads(JSON_CACHE.read_text())
    bodies = payload.get("bodies") or payload
    out: dict[int, dict[str, dict[str, int]]] = {}
    for key, rois in bodies.items():
        if key in {"n_bodies", "batches", "seconds"}:
            continue
        try:
            body = int(key)
        except (TypeError, ValueError):
            continue
        if not isinstance(rois, dict):
            continue
        out[body] = {
            str(roi): {str(k): int(v) for k, v in (counts or {}).items()}
            for roi, counts in rois.items()
            if isinstance(counts, dict)
        }
    return out


def parquet_from_json_and_annotations() -> pa.Table:
    records = annotation_synpoint_rows()
    if not records:
        schema_names = [
            "malecns_body_id",
            "type",
            "instance",
            "side",
            "soma_neuromere",
            "superclass",
            *ROI_COLUMNS,
        ]
        return pa.table({name: [] for name in schema_names})
    return pa.Table.from_pylist(records)


def write_roi_parquet(table: pa.Table, path: Path = PARQUET_CACHE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    if path.resolve() != LEGACY_PARQUET_CACHE.resolve():
        pq.write_table(table, LEGACY_PARQUET_CACHE)
    return path


def load_roi_parquet(path: Path | None = None) -> list[dict]:
    """Derived cache only. Mapping identity must use annotation_synpoint_rows()."""
    return annotation_synpoint_rows()


def load_roi_rows_by_body(path: Path | None = None) -> dict[int, dict]:
    return {int(row["malecns_body_id"]): row for row in load_roi_parquet(path)}


def verify_pugliese_manc_t1_dng100(path: Path = PUGLIESE_MANC_T1_TABLE) -> dict:
    """Read the authors' checked-in MANC T1 table. Do not guess IDs."""
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    stim = table.iloc[PUGLIESE_DNG100_STIM["matrix_index"]]
    not_dng = table.iloc[MANC_VMS16["matrix_index"]]
    stim_body = int(stim["bodyId"])
    stim_type = str(stim["type"])
    other_body = int(not_dng["bodyId"])
    other_type = str(not_dng["type"])
    if stim_body != PUGLIESE_DNG100_STIM["body_id"] or stim_type != "DNg100":
        raise ValueError(
            f"MANC T1 row {PUGLIESE_DNG100_STIM['matrix_index']} is "
            f"bodyId={stim_body} type={stim_type}, expected DNg100 "
            f"{PUGLIESE_DNG100_STIM['body_id']}"
        )
    if other_body != MANC_VMS16["body_id"] or other_type != "vMS16":
        raise ValueError(
            f"MANC T1 row {MANC_VMS16['matrix_index']} is "
            f"bodyId={other_body} type={other_type}, expected vMS16 {MANC_VMS16['body_id']}"
        )
    dng_rows = table[table["type"].astype(str).eq("DNg100")]
    return {
        "pugliese_reference": namespaced_manc_only(PUGLIESE_DNG100_STIM),
        "manc_vms16": namespaced_manc_only(
            {
                "source_dataset": "MANC_T1",
                "source_matrix_index": int(not_dng.name),
                "source_body_id": other_body,
                "type": other_type,
                "predicted_nt": str(not_dng.get("predictedNt") or ""),
                "note": MANC_VMS16.get("note"),
            }
        ),
        "manc_t1_dng100": [
            namespaced_manc_only(
                {
                    "source_dataset": "MANC_T1",
                    "source_matrix_index": int(idx),
                    "source_body_id": int(row["bodyId"]),
                    "type": "DNg100",
                    "predicted_nt": str(row.get("predictedNt") or ""),
                }
            )
            for idx, row in dng_rows.iterrows()
        ],
        "manc_body_ids": {int(v) for v in table["bodyId"].tolist()},
    }


def _manc_correspondence(row: dict) -> dict | None:
    mid = row.get("manc_body_id")
    if mid is None:
        return None
    return {
        "source": "MaleCNS annotation fields mancBodyid / mancType",
        "source_dataset": "MANC_T1",
        "manc_body_id": int(mid),
        "manc_type": row.get("manc_type") or None,
        "note": (
            "Curated cross-dataset correspondence. Same annotated cell type "
            "on a different reconstructed specimen. Not integer identity of malecns_body_id."
        ),
    }


def _neuron_record(row: dict, assigned: dict) -> dict:
    """One MaleCNS CPG cell. malecns_body_id is from raw annotations.bodyId."""
    rec = {
        "source_dataset": "MaleCNS_v1.0",
        "malecns_body_id": int(row["malecns_body_id"]),
        "type": row.get("type") or "",
        "side": row.get("side") or "",
        "identity_method": f"raw annotation type == {row.get('type') or ''}",
        "roi_counts": assigned["roi_counts"],
        "assigned_segment": assigned["assigned_segment"],
        "assignment_confidence": assigned["assignment_confidence"],
        "assignment_method": ASSIGNMENT_METHOD,
        "fallback_used": False,
        "assignment_status": assigned["assignment_status"],
        "assigned_side": assigned["assigned_side"],
        "assigned_slot": assigned["assigned_slot"],
    }
    corr = _manc_correspondence(row)
    if corr:
        rec["manc_correspondence"] = corr
    return rec


def _dng100_malecns_record(row: dict, assigned: dict, nt: dict) -> dict:
    side = _norm_side(row.get("side"))
    if not side:
        instance = str(row.get("instance") or "")
        if instance.endswith("_L"):
            side = "L"
        elif instance.endswith("_R"):
            side = "R"
    body = int(row["malecns_body_id"])
    nt_row = nt.get(body) or {}
    conf = nt_row.get("predicted_nt_confidence")
    rec = {
        "source_dataset": "MaleCNS_v1.0",
        "malecns_body_id": body,
        "type": "DNg100",
        "side": side,
        "identity_method": "raw annotation type == DNg100",
        "root_side": row.get("root_side") or "",
        "instance": row.get("instance") or "",
        "predicted_nt": nt_row.get("predicted_nt") or nt_row.get("celltype_predicted_nt") or "",
        "predicted_nt_confidence": None if conf in (None, "") else float(conf),
        "roi_counts": assigned["roi_counts"],
        "assigned_segment": None,
        "assignment_confidence": assigned["assignment_confidence"],
        "assignment_method": DNG100_ASSIGNMENT_METHOD,
        "fallback_used": False,
        "assignment_status": "AMBIGUOUS",
        "innervation_side": innervation_side_of(row),
        "manc_correspondence": _manc_correspondence(row),
        "note": (
            "MaleCNS annotations[type == DNg100]. Same annotated cell type as "
            "Pugliese's MANC DNg100, different reconstructed specimen. "
            "manc_correspondence is the curated mancBodyid field, not identity."
        ),
    }
    return rec


def build_cpg_mapping(rows: list[dict] | None = None) -> dict:
    # Identity first: crash unless the raw feather has exactly two DNg100 bodyIds.
    malecns_dng100_body_ids()
    rows = rows if rows is not None else annotation_synpoint_rows()
    by_type: dict[str, list[dict]] = {}
    for row in rows:
        by_type.setdefault(str(row["type"]), []).append(row)

    if PUGLIESE_MANC_T1_TABLE.exists():
        pugliese = verify_pugliese_manc_t1_dng100()
    else:
        pugliese = {
            "pugliese_reference": namespaced_manc_only(PUGLIESE_DNG100_STIM),
            "manc_vms16": namespaced_manc_only(MANC_VMS16),
            "manc_t1_dng100": [],
            "manc_body_ids": set(),
        }

    mapping: dict = {
        "dataset": "MaleCNS v1.0",
        "roles": {
            "E1": CORE_CPG_TYPES["E1"],
            "E2": CORE_CPG_TYPES["E2"],
            "E3": CORE_CPG_TYPES["E3"],
            "I1": CORE_CPG_TYPES["I1"],
            "I2": CORE_CPG_TYPES["I2"],
            "E4": EXTENDED_CPG_TYPES["E4"],
            "E5": EXTENDED_CPG_TYPES["E5"],
            "DNg100": "DNg100",
        },
        "source": (
            "Identity: body-annotations-male-cns-v1.0-minconf-0.5.feather "
            "[type] via bodyId renamed malecns_body_id. "
            "ROI: syn-points-male-cns-v1.0-minconf-0.5.feather column body, "
            "joined on malecns_body_id. Not the derived CPG parquet. "
            "Not type-level ROI totals. Soma XYZ / somaNeuromere unused. "
            "Pugliese MANC IDs appear only under DNg100.pugliese_reference."
        ),
        "assignment": {
            "signal": "T#_score = PreSyn+PostSyn in LegNp(T#)(L)+LegNp(T#)(R); then L vs R",
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "ambiguous_policy": "assigned_segment=null, assignment_status=AMBIGUOUS; never soma-Z or somaNeuromere",
            "fallback_used": False,
            "assignment_method": ASSIGNMENT_METHOD,
            "type_level_roi_pooling": False,
        },
        "i2_type_provenance": dict(I2_TYPE_PROVENANCE),
    }

    mapping["DNg100"] = _dng100_block(by_type.get("DNg100", []), pugliese)

    for typename in CPG_TYPES:
        role = ROLE_FOR_TYPE[typename]
        neurons: dict[str, dict | None] = {slot: None for slot in PAPER_SLOTS}
        unassigned: list[dict] = []
        for row in by_type.get(typename, []):
            assigned = assign_leg_from_row(row)
            record = _neuron_record(row, assigned)
            slot = assigned["assigned_slot"]
            if assigned["assignment_status"] == "ok" and slot:
                if neurons[slot] is not None:
                    record["assignment_status"] = "AMBIGUOUS"
                    record["assigned_segment"] = None
                    record["assigned_slot"] = None
                    record["note"] = (
                        f"slot {slot} already filled by "
                        f"{neurons[slot]['malecns_body_id']}"
                    )
                    unassigned.append(record)
                    continue
                neurons[slot] = record
            else:
                if assigned["assigned_segment"] is None:
                    record["assigned_segment"] = None
                unassigned.append(record)
        mapping[typename] = {
            "role": role,
            "type": typename,
            "neurons": neurons,
            "unassigned": unassigned,
            "n_assigned": sum(1 for value in neurons.values() if value is not None),
            "n_expected": EXPECTED_COPIES.get(typename, 6),
            "n_in_annotations": len(by_type.get(typename, [])),
        }
    validate_cpg_mapping(mapping)
    return mapping


def _dng100_block(rows: list[dict], pugliese: dict) -> dict:
    nt = load_predicted_nt({int(row["malecns_body_id"]) for row in rows})
    left = None
    right = None
    extras: list[dict] = []
    for row in rows:
        assigned = assign_leg_from_row(row)
        record = _dng100_malecns_record(row, assigned, nt)
        side = record["side"]
        if side == "L" and left is None:
            left = record
        elif side == "R" and right is None:
            right = record
        else:
            extras.append(record)
    return {
        "pugliese_reference": namespaced_manc_only(pugliese["pugliese_reference"]),
        "manc_vms16": namespaced_manc_only(pugliese.get("manc_vms16") or MANC_VMS16),
        "malecns": {
            "left": left,
            "right": right,
            "unassigned": extras,
            "n_expected": 2,
            "n_in_annotations": len(rows),
            "note": (
                "Looked up by type == DNg100 in the official MaleCNS annotation feather. "
                "Same annotated cell type as Pugliese MANC DNg100, different specimen. "
                "manc_correspondence copies mancBodyid/mancType; it is not identity. "
                "MANC table body 10056 is vMS16. MaleCNS bodyId 10056 is independently DNg100_R."
            ),
        },
    }


def collect_malecns_body_ids(obj: object) -> list[int]:
    """Values of keys literally named malecns_body_id."""
    found: list[int] = []
    if isinstance(obj, dict):
        if obj.get("malecns_body_id") is not None:
            found.append(int(obj["malecns_body_id"]))
        for value in obj.values():
            found.extend(collect_malecns_body_ids(value))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(collect_malecns_body_ids(item))
    return found


def iter_cpg_neuron_records(mapping: dict) -> list[dict]:
    records: list[dict] = []
    for typename in CPG_TYPES:
        block = mapping.get(typename) or {}
        for rec in (block.get("neurons") or {}).values():
            if isinstance(rec, dict):
                records.append(rec)
        for rec in block.get("unassigned") or []:
            if isinstance(rec, dict):
                records.append(rec)
    return records


def iter_dng100_malecns_records(mapping: dict) -> list[dict]:
    malecns = (mapping.get("DNg100") or {}).get("malecns") or {}
    records = []
    for key in ("left", "right"):
        rec = malecns.get(key)
        if isinstance(rec, dict):
            records.append(rec)
    for rec in malecns.get("unassigned") or []:
        if isinstance(rec, dict):
            records.append(rec)
    return records


def validate_cpg_mapping(mapping: dict, *, manc_body_ids: set[int] | None = None) -> None:
    """Crash unless identity is proven by the raw annotation feather."""
    import pandas as pd

    _assert_forbidden_keys(mapping)
    assert_every_body_id_has_dataset(mapping)
    if CORE_CPG_TYPES["I2"] != "IN19A007":
        raise AssertionError("I2 must be IN19A007")
    if "IN19B007" in CORE_CPG_TYPES.values():
        raise AssertionError("IN19B007 is not I2")

    raw = load_raw_annotation_table()
    if "bodyId" not in raw.columns:
        raise AssertionError("bodyId missing from raw annotations")
    dng_raw = raw.loc[raw["type"].astype(str).str.strip().eq("DNg100")]
    raw_dng_ids = {int(v) for v in dng_raw["bodyId"]}
    if raw_dng_ids != MALECNS_DNG100_BODY_IDS:
        raise AssertionError(
            f"raw DNg100 bodyIds {raw_dng_ids} != {MALECNS_DNG100_BODY_IDS}"
        )

    dng_records = iter_dng100_malecns_records(mapping)
    if len(dng_records) != 2:
        raise ValueError(f"MaleCNS DNg100 entries: {len(dng_records)}")
    mapped_dng_ids = {int(rec["malecns_body_id"]) for rec in dng_records}
    if mapped_dng_ids != raw_dng_ids:
        raise AssertionError(
            f"MaleCNS DNg100 mapping {mapped_dng_ids} != raw annotations[type==DNg100] {raw_dng_ids}"
        )
    for rec in dng_records:
        if rec.get("type") != "DNg100":
            raise ValueError(f"MaleCNS DNg100 record has type={rec.get('type')!r}")
        if rec.get("source_dataset") not in {"MaleCNS_v1.0", "MaleCNS_v1"}:
            raise ValueError("MaleCNS DNg100 source_dataset must be MaleCNS_v1.0")
        if rec.get("assignment_method") != DNG100_ASSIGNMENT_METHOD:
            raise ValueError(f"DNg100 assignment_method={rec.get('assignment_method')!r}")
        if rec.get("identity_method") != "raw annotation type == DNg100":
            raise ValueError(f"DNg100 identity_method={rec.get('identity_method')!r}")
        body = int(rec["malecns_body_id"])
        source = raw.loc[raw["bodyId"] == body]
        if len(source) != 1:
            raise AssertionError(f"raw annotations bodyId={body} n={len(source)}")
        if str(source.iloc[0]["type"]).strip() != "DNg100":
            raise AssertionError(
                f"malecns_body_id {body} is type {source.iloc[0]['type']} in raw annotations, not DNg100"
            )
        if rec.get("fallback_used"):
            raise ValueError(f"DNg100 {body} used a fallback")
        if rec.get("assigned_segment") is not None:
            raise ValueError(f"DNg100 {body} must not be assigned a leg segment")
        if "body_id" in rec or "dng100_body_id" in rec:
            raise AssertionError("MaleCNS DNg100 must use malecns_body_id, not body_id")
        pug = rec.get("manc_correspondence") or {}
        if pug.get("manc_body_id") is not None:
            expected = source.iloc[0]["mancBodyid"]
            if pd.notna(expected) and int(expected) != int(pug["manc_body_id"]):
                raise AssertionError("manc_correspondence does not match raw mancBodyid")

    mapping_rows = list(iter_cpg_neuron_records(mapping)) + dng_records
    for rec in mapping_rows:
        mid = int(rec["malecns_body_id"])
        claimed = str(rec.get("type") or "").strip()
        source = raw.loc[raw["bodyId"] == mid]
        if len(source) != 1:
            raise AssertionError(f"mapping malecns_body_id {mid} not unique in raw annotations")
        if str(source.iloc[0]["type"]).strip() != claimed:
            raise AssertionError(
                f"malecns_body_id {mid}: raw type {source.iloc[0]['type']} != mapping {claimed}"
            )

    for rec in iter_cpg_neuron_records(mapping):
        if rec.get("fallback_used"):
            raise ValueError(f"CPG cell {rec.get('malecns_body_id')} used a fallback")
        method = rec.get("assignment_method") or ""
        if "soma" in method.lower() or method != ASSIGNMENT_METHOD:
            raise ValueError(f"CPG cell {rec.get('malecns_body_id')} assignment_method={method!r}")
        if rec.get("assignment_status") == "AMBIGUOUS" and rec.get("assigned_slot"):
            raise ValueError(f"CPG cell {rec.get('malecns_body_id')} is AMBIGUOUS but slotted")

    i2 = mapping.get("IN19A007") or {}
    if i2.get("role") != "I2" or i2.get("type") != "IN19A007":
        raise ValueError("I2 must be IN19A007")
    if mapping.get("IN19B007"):
        raise ValueError("IN19B007 must not appear as a mapping type key")
    if (mapping.get("roles") or {}).get("I2") != "IN19A007":
        raise ValueError("roles.I2 must be IN19A007")

    listed = {int(v) for v in collect_malecns_body_ids(mapping)}
    if PUGLIESE_MANC_STIM_BODY in listed and PUGLIESE_MANC_STIM_BODY not in raw_dng_ids:
        raise AssertionError(
            "MANC stim body 10093 must not appear as malecns_body_id unless "
            "raw MaleCNS annotations[type==DNg100] independently contain it "
            f"(MaleCNS bodyId 10093 is {raw.loc[raw['bodyId'] == PUGLIESE_MANC_STIM_BODY]['type'].tolist()})"
        )
    pug = mapping.get("DNg100", {}).get("pugliese_reference") or {}
    if pug.get("source_body_id") != PUGLIESE_MANC_STIM_BODY or pug.get("source_dataset") != "MANC_T1":
        raise AssertionError(
            f"Pugliese reference must be MANC_T1 source_body_id={PUGLIESE_MANC_STIM_BODY}"
        )
    if pug.get("source_matrix_index") != PUGLIESE_MANC_STIM_INDEX:
        raise AssertionError(
            f"Pugliese reference must be source_matrix_index={PUGLIESE_MANC_STIM_INDEX}"
        )
    if "malecns_body_id" in pug:
        raise AssertionError("pugliese_reference must not carry malecns_body_id")


def archive_invalid_namespace_artifacts() -> dict[str, Path]:
    """Keep forensic copies. Never overwrite an existing INVALID_* file."""
    written: dict[str, Path] = {}
    if MAPPING_PATH.exists() and not INVALID_NAMESPACE_MAPPING_PATH.exists():
        INVALID_NAMESPACE_MAPPING_PATH.write_text(MAPPING_PATH.read_text())
        written["mapping"] = INVALID_NAMESPACE_MAPPING_PATH
    if PARQUET_CACHE.exists() and not INVALID_NAMESPACE_PARQUET_PATH.exists():
        INVALID_NAMESPACE_PARQUET_PATH.write_bytes(PARQUET_CACHE.read_bytes())
        written["parquet"] = INVALID_NAMESPACE_PARQUET_PATH
    return written


def write_cpg_mapping(mapping: dict | None = None, path: Path = MAPPING_PATH) -> Path:
    mapping = mapping if mapping is not None else build_cpg_mapping()
    validate_cpg_mapping(mapping)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping, indent=2) + "\n")
    return path


def archive_invalid_mapping(current: Path = MAPPING_PATH) -> Path | None:
    """Keep a forensic copy of the previous mapping. Do not overwrite the archive."""
    if INVALID_MAPPING_PATH.exists() or not current.exists():
        return INVALID_MAPPING_PATH if INVALID_MAPPING_PATH.exists() else None
    INVALID_MAPPING_PATH.write_text(current.read_text())
    return INVALID_MAPPING_PATH


def _assert_forbidden_keys(obj: object) -> None:
    if isinstance(obj, dict):
        for forbidden in FORBIDDEN_KEYS:
            if forbidden in obj:
                raise ValueError(f"generic field {forbidden!r} is forbidden")
        for value in obj.values():
            _assert_forbidden_keys(value)
    elif isinstance(obj, list):
        for item in obj:
            _assert_forbidden_keys(item)


def load_cpg_mapping(path: Path = MAPPING_PATH) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    mapping = build_cpg_mapping()
    if mapping.get("IN17A001", {}).get("n_assigned"):
        write_cpg_mapping(mapping, path)
    return mapping


def malecns_dng100_by_annotation_side(mapping: dict | None = None, side: str = "L") -> int | None:
    """MaleCNS DNg100 malecns_body_id for annotation side L/R."""
    mapping = mapping if mapping is not None else load_cpg_mapping()
    key = "left" if _norm_side(side) == "L" else "right" if _norm_side(side) == "R" else ""
    rec = ((mapping.get("DNg100") or {}).get("malecns") or {}).get(key) or {}
    return malecns_body_id_of(rec)


def malecns_dng100_by_vnc_innervation(mapping: dict | None = None, side: str = "L") -> int | None:
    """MaleCNS DNg100 whose LegNp synapses prefer this VNC side."""
    mapping = mapping if mapping is not None else load_cpg_mapping()
    want = _norm_side(side)
    for rec in iter_dng100_malecns_records(mapping):
        if rec.get("innervation_side") == want:
            return malecns_body_id_of(rec)
    return None


def malecns_dng100_matching_manc_body(manc_body_id: int, mapping: dict | None = None) -> int | None:
    """MaleCNS DNg100 whose curated mancBodyid equals this MANC body."""
    mapping = mapping if mapping is not None else load_cpg_mapping()
    want = int(manc_body_id)
    for rec in iter_dng100_malecns_records(mapping):
        corr = rec.get("manc_correspondence") or {}
        if corr.get("manc_body_id") == want:
            return malecns_body_id_of(rec)
    return None


def left_vnc_dng100_malecns_body_id(mapping: dict | None = None) -> int | None:
    """MaleCNS DNg100 that innervates the left VNC. Not Pugliese matrix index 31."""
    return malecns_dng100_by_vnc_innervation(mapping, "L")


def scan_syn_points(
    body_ids: set[int],
    syn_path: Path = SYN_POINTS,
    progress: bool = True,
    batch_size: int = 65536,
) -> dict[int, dict[str, dict[str, int]]]:
    """Aggregate per-bodyId ROI counts from syn-points without loading the file into RAM.

    Reads only body / kind / primary / primary_label. `primary` holds names
    such as LegNp(T1)(L); `primary_label` is the integer code.
    """
    if not syn_path.exists():
        raise FileNotFoundError(syn_path)
    wanted = {int(b) for b in body_ids}
    counts: dict[int, dict[str, dict[str, int]]] = {int(b): {} for b in wanted}
    dataset = ds.dataset(str(syn_path), format="feather")
    columns = [name for name in ("body", "kind", "primary", "primary_label") if name in dataset.schema.names]
    scanner = dataset.scanner(columns=columns, batch_size=batch_size)
    n_batches = 0
    for batch in scanner.to_batches():
        n_batches += 1
        frame = batch.to_pandas()
        frame = frame[frame["body"].isin(wanted)]
        if frame.empty:
            if progress and n_batches % 500 == 0:
                print(f"  syn-points batch {n_batches}", flush=True)
            continue
        roi_col = "primary" if "primary" in frame.columns else "primary_label"
        for rec in frame.itertuples(index=False):
            body = int(rec.body)
            polarity = _kind_to_polarity(getattr(rec, "kind", ""))
            if polarity is None:
                continue
            roi = str(getattr(rec, roi_col, "") or "")
            if not roi or roi == "<unspecified>":
                continue
            bucket = counts[body].setdefault(roi, {"pre": 0, "post": 0})
            bucket[polarity] += 1
        if progress and n_batches % 500 == 0:
            print(
                f"  syn-points batch {n_batches} hits so far "
                f"{sum(sum(v['pre']+v['post'] for v in rois.values()) for rois in counts.values())}",
                flush=True,
            )
    if progress:
        print(f"  syn-points done batches={n_batches} bodies={len(wanted)}", flush=True)
    return counts


def write_neuron_metadata(connectome, path: Path = METADATA_PATH) -> Path:
    """malecns_body_id → graph index plus annotations. Not ROI."""
    records = []
    for i in range(connectome.n):
        records.append(
            {
                "index": i,
                "malecns_body_id": int(connectome.neuron_ids[i]),
                "type": str(connectome.cell_type[i] or ""),
                "side": str(connectome.side[i] or ""),
                "superclass": str(connectome.superclass[i] or ""),
                "class": str(connectome.cell_class[i] or ""),
                "neurotransmitter": str(connectome.neurotransmitter[i] or ""),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), path)
    return path
