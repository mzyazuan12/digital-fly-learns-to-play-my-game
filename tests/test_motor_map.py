from organism.motor_map import MotorNeuronMuscleMap
from organism.config import MotorMode
from organism.toy import miniature_connectome
from organism.fly import VirtualFly


def test_motor_map_resolves_toy_motor_neurons():
    graph = miniature_connectome(0)
    mmap = MotorNeuronMuscleMap(graph)
    assert mmap.vnc_motor_indices.size >= 1
    assert mmap.notes["not_complete_muscle_recovery"] is True
    assert mmap.notes["cross_sex_inference_present"] is True
    rows = mmap.activity(__import__("numpy").zeros(graph.n), 0.01)
    assert rows
    assert "mapping_kind" in rows[0]


def test_default_motor_mode_is_engineered_cpg():
    fly = VirtualFly.hatch(seed=0, connectome="synthetic")
    assert fly.motor_mode is MotorMode.ENGINEERED_CPG
    assert fly.motor_map.notes["n_vnc_motor_annotated"] >= 1
