"""Published DNg100 → per-leg VNC CPG rhythm experiment.

Does not add walk(). Does not retune the graph when the trace is tonic.
"""

import inspect

import numpy as np

from organism.config import MotorMode, motor_fidelity_level
from organism.cpg_rhythm import rhythmicity_score
from organism.motor_map import command_for_mode
from organism.toy import miniature_connectome
from organism.walking_pathways import (
    CPG_INTERNEURONS,
    E5_TYPE_PROVENANCE,
    I2_TYPE_PROVENANCE,
    WALKING_CIRCUIT_TYPES,
    WalkingCircuit,
)
from organism.neuropil import LEG_SLOTS as NEURO_SLOTS
from organism.neuropil import assign_cell, attach_roi, load_annotation_neuropil
from experiment.dng100_cpg_rhythm import run
from experiment import dng100_cpg_rhythm


def test_walking_circuit_types_use_published_e5():
    assert WALKING_CIRCUIT_TYPES["E1"] == "IN17A001"
    assert WALKING_CIRCUIT_TYPES["E2"] == "INXXX466"
    assert WALKING_CIRCUIT_TYPES["I1"] == "IN16B036"
    assert WALKING_CIRCUIT_TYPES["I2"] == "IN19B007"
    assert WALKING_CIRCUIT_TYPES["E3"] == "IN19B012"
    assert WALKING_CIRCUIT_TYPES["E4"] == "IN03A006"
    assert WALKING_CIRCUIT_TYPES["E5"] == "INXXX464"
    assert E5_TYPE_PROVENANCE["canonical"] == "INXXX464"
    assert E5_TYPE_PROVENANCE["rejected_alias"] == "INXXX466"
    assert I2_TYPE_PROVENANCE["canonical"] == "IN19B007"
    assert I2_TYPE_PROVENANCE["rejected_alias"] == "IN19A007"
    e2 = next(p for p in CPG_INTERNEURONS if p.name == "E2")
    e5 = next(p for p in CPG_INTERNEURONS if p.name == "E5")
    i2 = next(p for p in CPG_INTERNEURONS if p.name == "I2")
    assert "E5" not in e2.aliases
    assert e2.types == ("INXXX466",)
    assert e5.types == ("INXXX464",)
    assert i2.types == ("IN19B007",)


def test_toy_has_six_leg_slots_and_fills_front_left():
    graph = miniature_connectome(1)
    circuit = WalkingCircuit(graph)
    catalog = circuit.catalog()
    for name in ("DNg100", "DNb08", "E1", "E2", "I1", "I2", "E3", "E4", "E5"):
        assert catalog[name]["resolved"], name
    assert circuit.legs["FL"].filled
    assert circuit.legs["FL"].cells["E1"] is not None
    assert circuit.legs["FL"].cells["E2"] is not None
    assert circuit.legs["FL"].cells["I1"] is not None
    assert circuit.legs["FL"].cells["E5"] is not None
    assert circuit.legs["FR"].filled is False
    anatomy = circuit.anatomy()
    assert anatomy["dnb08_to_E5"]["contacts"] > 0
    assert anatomy["E5_to_E1"]["contacts"] > 0
    assert anatomy["n_leg_copies_with_E1"] == 1
    rates = circuit.rates(np.zeros(graph.n, dtype=np.int32), 0.01)
    legs = circuit.leg_activity(np.zeros(graph.n, dtype=np.int32), 0.01)
    assert "E1" in rates
    assert legs["FL"]["E1"]["index"] == circuit.legs["FL"].cells["E1"]
    assert legs["FR"]["E1"]["index"] is None


def test_rhythmicity_detects_sine_tonic_and_explosion():
    dt = 1.0
    t = np.arange(400)
    sine = 0.5 + 0.4 * np.sin(2 * np.pi * 10.0 * t / 1000.0)
    scored = rhythmicity_score(sine, dt)
    assert scored["oscillatory"]
    assert 7.0 <= float(scored["peak_hz"]) <= 15.0
    assert scored["exploding"] is False
    tonic = np.full(400, 0.8)
    plateau = rhythmicity_score(tonic, dt)
    assert plateau["tonic_plateau"]
    assert plateau["oscillatory"] is False
    boom = np.linspace(-52.0, -400.0, 400)
    exploded = rhythmicity_score(boom, dt)
    assert exploded["exploding"]
    assert exploded["oscillatory"] is False


def test_neural_cpg_mode_does_not_hand_joints_to_flygym():
    assert motor_fidelity_level(MotorMode.NEURAL_CPG) == 2
    report = command_for_mode(MotorMode.NEURAL_CPG, left=1.0, right=1.0, walk_mode="walk", mn_activity=[])
    assert report["legs_still_cpg"] is False
    assert report["neural_cpg_timing"] is True
    assert report["joints_from_neural_cpg"] is False


def test_dng100_rhythm_experiment_on_toy_does_not_call_walk():
    source = inspect.getsource(dng100_cpg_rhythm)
    assert "def walk" not in source
    assert "if DNg100" not in source
    result = run(connectome="toy", seed=1, current=40.0, steps=80, warmup=10, lesions=True)
    assert result["walk_api_called"] is False
    assert result["engineered_cpg_used"] is False
    assert result["graph_modified"] is False
    assert result["dynamics_retuned"] is False
    assert result["frequency_forced"] is False
    assert result["motor_mode"] == "MODE_NEURAL_CPG"
    assert result["walking_circuit_types"]["E5"] == "INXXX464"
    assert result["dng100_n"] == 2
    assert len(result["stimulated"]["body_ids"]) == 1
    intact = result["conditions"]["intact"]["summary"]
    assert "legs" in intact
    assert "FL" in intact["legs"]
    assert "E1" in intact["legs"]["FL"]
    assert "lesion_E1" in result["conditions"]
    assert "lesion_I2" in result["conditions"]
    assert "scrambled_connectome" in result["conditions"]
    assert result["answer"] in {"yes", "no"}
    if not result["oscillation_reproduced"]:
        assert "Do not change the graph" in result["next_step"] or "do not change the graph" in result["next_step"].lower()
    assert result["dng100_is_six_neurons"] is False
    source = inspect.getsource(__import__("organism.walking_pathways", fromlist=["walking_pathways"]))
    assert "somaLocation Z" not in source
    assert "SOMA_Z" not in source
    assert "somaNeuromere_annotation" not in inspect.getsource(assign_cell)


def test_cpg_mapping_json_keeps_pugliese_and_malecns_ids_apart():
    from pathlib import Path
    import json
    from organism.roi_innervation import PUGLIESE_DNG100_STIM, malecns_body_id_of

    path = Path("data/malecns_v1/cpg_mapping.json")
    if not path.exists():
        return
    mapping = json.loads(path.read_text())
    assert mapping["assignment"]["fallback_used"] is False
    assert mapping["assignment"]["type_level_roi_pooling"] is False
    stim = mapping["DNg100"]["pugliese_reference"]
    assert stim["source_matrix_index"] == 31
    assert stim["source_body_id"] == 10093
    assert stim["source_dataset"] == "MANC_T1"
    assert stim["type"] == "DNg100"
    assert PUGLIESE_DNG100_STIM["source_body_id"] == 10093
    assert mapping["DNg100"]["manc_vms16"]["source_body_id"] == 10056
    assert mapping["DNg100"]["manc_vms16"]["type"] == "vMS16"
    e1 = mapping["IN17A001"]["neurons"]
    assert set(e1) == {"LF", "RF", "LM", "RM", "LH", "RH"}
    e1_ids = [malecns_body_id_of(value) for value in e1.values()]
    if any(e1_ids):
        assert len({i for i in e1_ids if i is not None}) == 6
    assert mapping["roles"]["I2"] == "IN19B007"
    assert mapping["IN19B007"]["role"] == "I2"
    assert mapping["IN19B007"]["type"] == "IN19B007"
    assert "IN19A007" not in mapping
    assert "dng100_body_id" not in mapping["DNg100"]
    left = mapping["DNg100"]["malecns"]["left"]
    right = mapping["DNg100"]["malecns"]["right"]
    assert "malecns_body_id" in left and "body_id" not in left
    assert "malecns_body_id" in right and "body_id" not in right
    assert mapping["DNg100"]["pugliese_reference"]["source_dataset"] == "MANC_T1"


def test_e1_annotation_covers_six_leg_neuropils():
    cells = load_annotation_neuropil(types=("IN17A001",))
    if not cells:
        return
    assert len(cells) == 6
    attach_roi(cells)
    assigned = {cell.assigned_slot for cell in cells.values() if cell.assigned_slot}
    assert assigned == set(NEURO_SLOTS)
    assert all(cell.assignment_source != "somaNeuromere_annotation" for cell in cells.values())
    dng = load_annotation_neuropil(types=("DNg100",))
    assert len(dng) == 2
    attach_roi(dng)
    sides = {cell.side for cell in dng.values()}
    assert sides == {"L", "R"}
    assert all(not cell.soma_neuromere for cell in dng.values())
    # Descending: all three neuropils on one side → not a single leg copy.
    assert all(cell.assigned_slot is None for cell in dng.values())
    assert all(cell.assignment_status == "AMBIGUOUS" for cell in dng.values())
