"""Four hard checks: mapping identity is the raw MaleCNS annotation feather."""

from organism.roi_innervation import (
    ANN_PATH,
    ASSIGNMENT_METHOD,
    CORE_CPG_TYPES,
    DNG100_ASSIGNMENT_METHOD,
    EXPECTED_COUNTS,
    INVALID_MAPPING_PATH,
    INVALID_NAMESPACE_MAPPING_PATH,
    INVALID_NAMESPACE_PARQUET_PATH,
    MANC_VMS16,
    MAPPING_PATH,
    PUGLIESE_DNG100_STIM,
    collect_malecns_body_ids,
    iter_cpg_neuron_records,
    iter_dng100_malecns_records,
    load_cpg_mapping,
    load_raw_annotation_table,
    lookup_malecns,
    lookup_manc,
    malecns_dng100_body_ids,
    raw_dng100_annotation_rows,
    validate_cpg_mapping,
)
from organism.walking_pathways import EXPECTED_WALKING_BODY_IDS
from organism.bridge import EXPECTED_BODY_IDS


def test_raw_annotations_have_bodyId_and_two_dng100():
    raw = load_raw_annotation_table()
    assert "bodyId" in raw.columns
    assert ANN_PATH.exists()
    dng = raw_dng100_annotation_rows()
    assert list(dng.columns)  # bodyId is a column, not the index
    assert "bodyId" in dng.columns
    assert len(dng) == 2
    ids = malecns_dng100_body_ids()
    assert ids == {"L": 10045, "R": 10056}
    assert 10093 not in set(ids.values())
    # Curated correspondence, not identity:
    right = dng.loc[dng["somaSide"].astype(str).str.upper().eq("R")].iloc[0]
    left = dng.loc[dng["somaSide"].astype(str).str.upper().eq("L")].iloc[0]
    assert int(right["bodyId"]) == 10056
    assert int(left["bodyId"]) == 10045
    assert int(right["mancBodyid"]) == 10093
    assert str(right["mancType"]) == "DNg100"
    assert int(left["mancBodyid"]) == 10339
    assert str(left["mancType"]) == "DNg100"
    am1 = raw.loc[raw["bodyId"] == 10093]
    assert len(am1) == 1
    assert str(am1.iloc[0]["type"]).strip() == "Am1"


def test_integer_10056_is_malecns_dng100_and_manc_vms16():
    raw = load_raw_annotation_table()
    malecns = raw.loc[raw["bodyId"] == 10056].iloc[0]
    assert str(malecns["type"]).strip() == "DNg100"
    assert str(malecns["somaSide"]).strip().upper() == "R"
    assert MANC_VMS16["source_dataset"] == "MANC_T1"
    assert MANC_VMS16["source_body_id"] == 10056
    assert MANC_VMS16["type"] == "vMS16"
    mapping = load_cpg_mapping()
    assert mapping["DNg100"]["malecns"]["right"]["malecns_body_id"] == 10056
    assert mapping["DNg100"]["malecns"]["left"]["malecns_body_id"] == 10045
    assert mapping["DNg100"]["manc_vms16"]["source_dataset"] == "MANC_T1"
    assert mapping["DNg100"]["manc_vms16"]["source_body_id"] == 10056
    assert mapping["DNg100"]["pugliese_reference"]["source_body_id"] == 10093
    assert "malecns_body_id" not in mapping["DNg100"]["pugliese_reference"]
    assert EXPECTED_WALKING_BODY_IDS["DNg100"]["L"] == (10045,)
    assert EXPECTED_WALKING_BODY_IDS["DNg100"]["R"] == (10056,)
    assert EXPECTED_BODY_IDS["DNg100"]["L"] == (10045,)
    assert EXPECTED_BODY_IDS["DNg100"]["R"] == (10056,)


def test_four_mapping_assertions():
    mapping = load_cpg_mapping()
    raw = load_raw_annotation_table()
    dng = iter_dng100_malecns_records(mapping)
    assert len(dng) == 2
    for rec in dng:
        assert rec["type"] == "DNg100"
        body = int(rec["malecns_body_id"])
        source = raw.loc[raw["bodyId"] == body]
        assert len(source) == 1
        assert str(source.iloc[0]["type"]).strip() == "DNg100"
        assert rec.get("fallback_used") is False
        assert rec["assigned_segment"] is None
        assert rec["source_dataset"] == "MaleCNS_v1.0"
        assert rec["identity_method"] == "raw annotation type == DNg100"
        assert rec["assignment_method"] == DNG100_ASSIGNMENT_METHOD

    for rec in iter_cpg_neuron_records(mapping):
        assert rec.get("fallback_used") is False
        assert rec.get("assignment_method") == ASSIGNMENT_METHOD
        assert "soma" not in rec.get("assignment_method", "").lower()
        assert rec.get("type") != "IN19B007"
        mid = rec["malecns_body_id"]
        source = raw.loc[raw["bodyId"] == mid]
        assert len(source) == 1
        assert str(source.iloc[0]["type"]).strip() == rec["type"]

    assert CORE_CPG_TYPES["I2"] == "IN19A007"
    assert "IN19B007" not in CORE_CPG_TYPES.values()
    assert mapping["IN19A007"]["role"] == "I2"
    assert mapping["roles"]["I2"] == "IN19A007"
    assert "IN19B007" not in mapping

    listed = collect_malecns_body_ids(mapping)
    assert 10093 not in listed
    pug = mapping["DNg100"]["pugliese_reference"]
    assert pug["source_dataset"] == "MANC_T1"
    assert pug["source_matrix_index"] == 31
    assert pug["source_body_id"] == 10093
    validate_cpg_mapping(mapping)


def test_dng100_manc_correspondence_is_not_identity():
    mapping = load_cpg_mapping()
    pug = mapping["DNg100"]["pugliese_reference"]
    assert pug["source_body_id"] == PUGLIESE_DNG100_STIM["source_body_id"] == 10093
    left = mapping["DNg100"]["malecns"]["left"]
    right = mapping["DNg100"]["malecns"]["right"]
    assert left["malecns_body_id"] != 10093
    assert right["malecns_body_id"] != 10093
    assert left["side"] == "L"
    assert right["side"] == "R"
    assert (right.get("manc_correspondence") or {}).get("manc_body_id") == 10093
    assert (left.get("manc_correspondence") or {}).get("manc_body_id") == 10339
    assert mapping["DNg100"]["manc_vms16"]["source_body_id"] == 10056
    assert mapping["DNg100"]["manc_vms16"]["type"] == "vMS16"
    assert "dng100_body_id" not in mapping["DNg100"]
    assert "body_id" not in left
    assert "body_id" not in right
    assert "malecns_body_id" not in pug
    raw = load_raw_annotation_table()
    by_side = {
        str(row.somaSide).strip().upper(): int(row.bodyId)
        for row in raw.loc[raw["type"].astype(str).str.strip().eq("DNg100")].itertuples()
    }
    assert left["malecns_body_id"] == by_side["L"]
    assert right["malecns_body_id"] == by_side["R"]


def test_cpg_records_are_per_body_roi_not_soma():
    mapping = load_cpg_mapping()
    e1 = mapping["IN17A001"]
    assert e1["n_assigned"] == 6
    for slot, rec in e1["neurons"].items():
        assert rec is not None, slot
        assert rec["fallback_used"] is False
        assert rec["assignment_method"] == ASSIGNMENT_METHOD
        assert rec["source_dataset"] == "MaleCNS_v1.0"
        winner = rec["assigned_segment"]
        others = [seg for seg in ("T1", "T2", "T3") if seg != winner]
        best = rec["roi_counts"][winner]["pre"] + rec["roi_counts"][winner]["post"]
        rest = sum(rec["roi_counts"][seg]["pre"] + rec["roi_counts"][seg]["post"] for seg in others)
        assert best > rest
    i2 = mapping["IN19A007"]
    assert i2["role"] == "I2"
    assert i2["n_in_annotations"] == 6
    assert i2["n_assigned"] + len(i2["unassigned"]) == 6
    for rec in i2["unassigned"]:
        assert rec["type"] == "IN19A007"
        assert rec["fallback_used"] is False
        assert rec["assignment_status"] == "AMBIGUOUS"


def test_invalid_derived_files_were_archived():
    assert INVALID_MAPPING_PATH.exists()
    assert INVALID_NAMESPACE_MAPPING_PATH.exists()
    assert INVALID_NAMESPACE_PARQUET_PATH.exists()
    assert MAPPING_PATH.exists()
    assert '"dng100_body_id"' not in MAPPING_PATH.read_text()
