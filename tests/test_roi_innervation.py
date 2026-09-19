"""Per-bodyId LegNp assignment. Type-level ROI totals are forbidden."""

from organism.roi_innervation import (
    CONFIDENCE_THRESHOLD,
    assign_leg_from_row,
    empty_roi_row,
    paper_slot_of,
)
from organism.neuropil import assign_cell, CellNeuropil


def test_paper_slots_are_front_middle_hind_not_type_pools():
    assert paper_slot_of("L", "T1") == "LF"
    assert paper_slot_of("R", "T3") == "RH"


def test_per_bodyid_t1_left_is_left_front():
    row = empty_roi_row()
    row["T1_L_post"] = 91
    row["T2_L_post"] = 5
    row["T3_L_post"] = 2
    assigned = assign_leg_from_row(row)
    assert assigned["assigned_slot"] == "LF"
    assert assigned["assigned_leg"] == "T1"
    assert assigned["status"] == "ok"
    assert assigned["confidence"] > CONFIDENCE_THRESHOLD
    assert assigned["second_best"] == "T2"
    assert assigned["t_scores"]["T1"] == 91
    assert assigned["soma_z_used"] is False
    assert assigned["fallback_used"] is False
    assert assigned["assigned_segment"] == "T1"


def test_per_bodyid_t2_left_is_left_middle_not_type_average():
    row = empty_roi_row()
    row["T1_L_post"] = 3
    row["T2_L_post"] = 93
    row["T3_L_post"] = 2
    assigned = assign_leg_from_row(row)
    assert assigned["assigned_slot"] == "LM"
    assert assigned["assigned_leg"] == "T2"


def test_mixed_innervation_is_ambiguous_not_soma_z():
    row = empty_roi_row()
    row["T1_L_post"] = 40
    row["T2_L_post"] = 35
    row["T3_L_post"] = 25
    assigned = assign_leg_from_row(row)
    assert assigned["status"] == "AMBIGUOUS"
    assert assigned["assigned_slot"] is None
    assert assigned["assigned_segment"] is None
    assert assigned["soma_z_used"] is False
    assert assigned["fallback_used"] is False
    cell = CellNeuropil(body_id=1, cell_type="IN17A001", side="L", soma_neuromere="T3")
    cell.roi_counts = {
        "LegNp(T1)(L)": {"pre": 0, "post": 40},
        "LegNp(T2)(L)": {"pre": 0, "post": 35},
        "LegNp(T3)(L)": {"pre": 0, "post": 25},
    }
    assign_cell(cell)
    assert cell.assigned_slot is None
    assert cell.assignment_source == "AMBIGUOUS"
    # somaNeuromere T3 must not silently win.
    assert cell.assigned_leg is None


def test_pre_plus_post_scores_are_not_post_only():
    row = empty_roi_row()
    row["T1_L_pre"] = 80
    row["T1_L_post"] = 11
    row["T2_L_post"] = 20
    assigned = assign_leg_from_row(row)
    assert assigned["assigned_leg"] == "T1"
    assert assigned["t_scores"]["T1"] == 91
    assert assigned["synapses_in"]["LF"] == 11
    assert assigned["synapses_out"]["LF"] == 80


def test_bilateral_t3_without_side_winner_is_ambiguous():
    row = empty_roi_row()
    row["T3_L_pre"] = 217
    row["T3_L_post"] = 417
    row["T3_R_pre"] = 133
    row["T3_R_post"] = 481
    assigned = assign_leg_from_row(row)
    assert assigned["assigned_leg"] == "T3"
    assert assigned["assigned_side"] is None
    assert assigned["assigned_slot"] is None
    assert assigned["status"] == "AMBIGUOUS"
    assert assigned["soma_z_used"] is False
    assert assigned["fallback_used"] is False
    assert assigned["assigned_segment"] == "T3"


def test_dng100_like_three_neuropils_on_one_side_is_ambiguous():
    row = empty_roi_row()
    row["T1_L_post"] = 124
    row["T2_L_post"] = 111
    row["T3_L_post"] = 171
    assigned = assign_leg_from_row(row)
    assert assigned["status"] == "AMBIGUOUS"
    assert assigned["assigned_slot"] is None
