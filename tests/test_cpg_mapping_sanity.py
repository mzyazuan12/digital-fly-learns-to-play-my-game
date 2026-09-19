"""Four hard checks on the generated CPG / DNg100 mapping."""

from organism.roi_innervation import (
    ASSIGNMENT_METHOD,
    CPG_TYPES,
    INVALID_MAPPING_PATH,
    MAPPING_PATH,
    PUGLIESE_DNG100_STIM,
    collect_malecns_body_ids,
    iter_cpg_neuron_records,
    iter_dng100_malecns_records,
    load_annotations,
    load_cpg_mapping,
    validate_cpg_mapping,
    verify_pugliese_manc_t1_dng100,
)


def test_four_mapping_assertions():
    mapping = load_cpg_mapping()
    anns = {int(row["malecns_body_id"]): row for row in load_annotations(types=("DNg100", *CPG_TYPES, "IN19A007"))}

    dng = iter_dng100_malecns_records(mapping)
    assert dng, "MaleCNS DNg100 entries missing"
    for rec in dng:
        assert rec["type"] == "DNg100"
        body = int(rec["body_id"])
        assert anns[body]["type"] == "DNg100", body
        assert rec.get("fallback_used") is False
        assert rec["assigned_segment"] is None
        assert rec["assignment_status"] == "AMBIGUOUS"
        assert "malecns_body_id" not in rec

    for rec in iter_cpg_neuron_records(mapping):
        assert rec.get("fallback_used") is False
        assert rec.get("assignment_method") == ASSIGNMENT_METHOD
        assert "soma" not in rec.get("assignment_method", "").lower()
        assert rec.get("type") != "IN19A007"

    i2 = mapping["IN19B007"]
    assert i2["role"] == "I2"
    assert i2["type"] == "IN19B007"
    assert mapping["roles"]["I2"] == "IN19B007"
    assert mapping["roles"]["E1"] == "IN17A001"
    assert mapping["roles"]["E2"] == "INXXX466"
    assert mapping["roles"]["E3"] == "IN19B012"
    assert mapping["roles"]["I1"] == "IN16B036"
    assert "IN19A007" not in mapping

    manc_ids = verify_pugliese_manc_t1_dng100()["manc_body_ids"]
    listed = collect_malecns_body_ids(mapping)
    assert 10093 not in listed
    overlap = set(listed) & manc_ids
    assert not overlap, overlap
    validate_cpg_mapping(mapping, manc_body_ids=manc_ids)


def test_dng100_is_type_not_body_identity():
    mapping = load_cpg_mapping()
    pug = mapping["DNg100"]["pugliese_reference"]
    assert pug["dataset"] == "MANC_T1"
    assert pug["matrix_index"] == 31
    assert pug["body_id"] == 10093
    assert pug["type"] == "DNg100"
    assert pug["body_id"] == PUGLIESE_DNG100_STIM["body_id"]
    left = mapping["DNg100"]["malecns"]["left"]
    right = mapping["DNg100"]["malecns"]["right"]
    assert left["type"] == right["type"] == "DNg100"
    assert left["side"] == "L"
    assert right["side"] == "R"
    assert left["body_id"] != 10093
    assert right["body_id"] != 10093
    assert "dng100_body_id" not in mapping["DNg100"]
    assert mapping["DNg100"]["manc_vms16"]["body_id"] == 10056
    assert mapping["DNg100"]["manc_vms16"]["type"] == "vMS16"
    # Integer 10056 may appear as MaleCNS DNg100_R; that is not MANC vMS16.
    if right["body_id"] == 10056:
        anns = {int(row["malecns_body_id"]): row for row in load_annotations(types=("DNg100",))}
        assert anns[10056]["type"] == "DNg100"
        assert anns[10056]["side"] == "R"


def test_cpg_records_are_per_body_roi_not_soma():
    mapping = load_cpg_mapping()
    e1 = mapping["IN17A001"]
    assert e1["n_assigned"] == 6
    for slot, rec in e1["neurons"].items():
        assert rec is not None, slot
        assert rec["fallback_used"] is False
        assert rec["assignment_method"] == ASSIGNMENT_METHOD
        assert rec["assignment_status"] == "ok"
        assert rec["assigned_segment"] in {"T1", "T2", "T3"}
        assert set(rec["roi_counts"]) == {"T1", "T2", "T3"}
        assert set(rec["roi_counts"]["T1"]) == {"pre", "post"}
        winner = rec["assigned_segment"]
        others = [seg for seg in ("T1", "T2", "T3") if seg != winner]
        best = rec["roi_counts"][winner]["pre"] + rec["roi_counts"][winner]["post"]
        rest = sum(rec["roi_counts"][seg]["pre"] + rec["roi_counts"][seg]["post"] for seg in others)
        assert best > rest
    i2_unassigned = mapping["IN19B007"]["unassigned"]
    assert i2_unassigned
    for rec in i2_unassigned:
        assert rec["type"] == "IN19B007"
        assert rec["fallback_used"] is False
        assert rec["assignment_status"] == "AMBIGUOUS"
        assert rec["assigned_slot"] is None


def test_invalid_pre_roi_mapping_was_archived_not_deleted():
    assert INVALID_MAPPING_PATH.exists()
    assert MAPPING_PATH.exists()
    old = INVALID_MAPPING_PATH.read_text()
    new = MAPPING_PATH.read_text()
    assert old != new
    assert '"dng100_body_id"' not in new
