"""Stage 2B sensory inclusion reproduces the design's counts and side checks."""

import json
from pathlib import Path

import pytest

TABLE = Path(__file__).resolve().parents[1] / "data" / "paper2" / "stage2B_sensory_table.json"
pytestmark = pytest.mark.skipif(not TABLE.exists(), reason="run experiment/paper2_sensory_table.py first")


def test_counts_match_design():
    d = json.loads(TABLE.read_text())
    assert d["included_counts"] == {
        "T1L:chordotonal_femur_tibia": 8,
        "T1L:hair_plate_coxa": 7,
        "T1L:hair_plate_trochanter": 3,
        "T1R:campaniform_trochanter": 1,
        "T1R:chordotonal_femur_tibia": 4,
        "T1R:hair_plate_coxa": 5,
        "T1R:hair_plate_trochanter": 3,
    }


def test_included_afferents_are_ipsilateral_and_front_leg():
    d = json.loads(TABLE.read_text())
    for r in d["afferents"]:
        if not r["included_2B"]:
            continue
        assert r["entryNerve"] in {"ProLN", "VProN", "DProN", "ProAN"}
        assert r["bodyId"] != 1078099767
        total = r["mn_synapses_L"] + r["mn_synapses_R"]
        if total >= 10:
            assert r["ipsilateral_fraction"] >= 0.8
        assert r["entryNerve"] not in {"PrN", "MesoLN", "MetaLN"}
