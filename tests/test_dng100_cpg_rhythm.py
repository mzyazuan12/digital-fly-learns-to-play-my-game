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
    E5_TYPE_PROVENANCE,
    WALKING_CIRCUIT_TYPES,
    WalkingCircuit,
)
from experiment.dng100_cpg_rhythm import run
from experiment import dng100_cpg_rhythm


def test_walking_circuit_types_use_published_e5():
    assert WALKING_CIRCUIT_TYPES["E1"] == "IN17A001"
    assert WALKING_CIRCUIT_TYPES["E2"] == "INXXX466"
    assert WALKING_CIRCUIT_TYPES["I1"] == "IN16B036"
    assert WALKING_CIRCUIT_TYPES["I2"] == "IN19A007"
    assert WALKING_CIRCUIT_TYPES["E3"] == "IN19B012"
    assert WALKING_CIRCUIT_TYPES["E4"] == "IN03A006"
    assert WALKING_CIRCUIT_TYPES["E5"] == "INXXX464"
    assert E5_TYPE_PROVENANCE["canonical"] == "INXXX464"
    assert E5_TYPE_PROVENANCE["rejected_alias"] == "INXXX466"
    e2 = next(p for p in WalkingCircuit.__init__.__globals__["CPG_INTERNEURONS"] if p.name == "E2")
    assert "E5" not in e2.aliases
    assert e2.types == ("INXXX466",)


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


def test_rhythmicity_detects_sine_and_tonic():
    dt = 1.0
    t = np.arange(400)
    sine = 0.5 + 0.4 * np.sin(2 * np.pi * 10.0 * t / 1000.0)
    scored = rhythmicity_score(sine, dt)
    assert scored["oscillatory"]
    assert 7.0 <= float(scored["peak_hz"]) <= 15.0
    tonic = np.full(400, 0.8)
    plateau = rhythmicity_score(tonic, dt)
    assert plateau["tonic_plateau"]
    assert plateau["oscillatory"] is False


def test_neural_cpg_mode_does_not_hand_joints_to_flygym():
    assert motor_fidelity_level(MotorMode.NEURAL_CPG) == 2
    report = command_for_mode(MotorMode.NEURAL_CPG, left=1.0, right=1.0, walk_mode="walk", mn_activity=[])
    assert report["legs_still_cpg"] is False
    assert report["neural_cpg_timing"] is True
    assert report["joints_from_neural_cpg"] is False


def test_dng100_rhythm_experiment_on_toy_does_not_call_walk():
    source = inspect.getsource(dng100_cpg_rhythm)
    assert "walk()" not in source
    assert "if DNg100" not in source
    result = run(connectome="toy", seed=1, current=40.0, steps=80, warmup=10, lesions=True)
    assert result["walk_api_called"] is False
    assert result["engineered_cpg_used"] is False
    assert result["graph_modified"] is False
    assert result["dynamics_retuned"] is False
    assert result["motor_mode"] == "MODE_NEURAL_CPG"
    assert result["walking_circuit_types"]["E5"] == "INXXX464"
    assert result["dng100_n"] == 2
    intact = result["conditions"]["intact"]["summary"]
    assert "legs" in intact
    assert "FL" in intact["legs"]
    assert "E1" in intact["legs"]["FL"]
    assert "lesion_E1" in result["conditions"]
    assert result["answer"] in {"yes", "no"}
    if not result["oscillation_reproduced"]:
        assert "Do not change the graph" in result["next_step"] or "do not change the graph" in result["next_step"].lower()
