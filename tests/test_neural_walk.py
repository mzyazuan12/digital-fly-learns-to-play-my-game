"""Walking start/stop must come from identified DNs, not a bout timer."""

from flybrain.network import LIFNetwork, LIFParams
from organism.bridge import MotorBridge
from organism.experiment_walk import probe_dnp09, run as run_walk_experiment
from organism.toy import miniature_connectome


def test_dnp09_current_is_necessary_and_sufficient_for_cpg_walk():
    result = probe_dnp09(seed=1, current=40.0)
    assert result["intact"]["mode"] == "walk"
    assert result["intact"]["scaffold_used"] is False
    assert result["intact"]["spikes"] > 0
    assert result["lesion"]["mode"] == "rest"
    assert result["lesion"]["spikes"] == 0
    assert result["restored"]["mode"] == "walk"
    assert result["neural_authority"] is True


def test_walking_drive_without_spikes_is_ignored():
    graph = miniature_connectome(2)
    bridge = MotorBridge(graph, legacy_scaffold=False)
    counts = __import__("numpy").zeros(graph.n, dtype=__import__("numpy").int32)
    cmd = bridge.read(counts, 0.04, walking_drive=1.0, walking_bout_s=5.0)
    assert cmd.mode == "rest"
    assert cmd.scaffold_used is False


def test_drive_sources_are_inspectable():
    graph = miniature_connectome(0)
    net = LIFNetwork(graph, params=LIFParams(dt=1.0), seed=0)
    bridge = MotorBridge(graph, legacy_scaffold=False)
    net.clear_drive()
    net.add_drive(bridge.walk_indices, 12.0, source="neuromod.octopamine.DNp09")
    net.step(2)
    assert "neuromod.octopamine.DNp09" in net.drive_sources
    assert "intrinsic.membrane_noise" in net.drive_sources


def test_walk_experiment_does_not_use_scaffold():
    result = run_walk_experiment()
    assert result["first_experiment_ok"]
    assert result["dnp09_lesion"]["neural_authority"]
    assert result["spontaneous"]["scaffold_used"] is False
    assert result["spontaneous"]["walk_implies_dn_activity"]
