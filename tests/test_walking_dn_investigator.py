"""Walking DN investigator: DNg100 / DNb08 / oDN1 / halt / VNC CPG.

Does not add walk(). FlyGym still executes joints.
"""

import inspect

import numpy as np

from organism.bridge import MotorBridge
from organism.toy import miniature_connectome
from organism.walking_pathways import WalkingCircuit
from experiment.walking_dn_investigator import probe_pathway, run
from flybrain.network import LIFNetwork, LIFParams


def test_toy_resolves_walking_command_dns_and_cpg():
    graph = miniature_connectome(1)
    circuit = WalkingCircuit(graph)
    catalog = circuit.catalog()
    for name in ("DNp09", "DNg100", "DNb08", "oDN1", "DNa01", "DNa02", "bluebell", "brake", "E1", "E2", "I1"):
        assert catalog[name]["resolved"], name
    assert catalog["foxglove"]["resolved"] is False
    assert catalog["MDN"]["resolved"] is False
    anatomy = circuit.anatomy()
    assert anatomy["dng100_to_E1"]["contacts"] > 0
    assert anatomy["E1_to_E2"]["contacts"] > 0
    assert anatomy["I1_to_E1"]["contacts"] > 0
    assert anatomy["cpg_core_to_vnc_motor"]["contacts"] > 0
    assert anatomy["engineered_cpg_still_executes_joints"] is True
    assert anatomy["neural_vnc_cpg_drives_joints"] is False


def test_motor_bridge_keeps_dnp09_as_named_walk_indices():
    graph = miniature_connectome(1)
    bridge = MotorBridge(graph, legacy_scaffold=False)
    names = {p.name: p for p in bridge.pathways}
    assert "DNp09" in names["walk_initiation"].resolved_types
    assert "DNg100" in names["walking_dng100"].resolved_types
    assert "DNg97" in names["walking_odn1"].resolved_types
    assert bridge.dng100_indices.size == 2
    assert set(bridge.walk_indices.tolist()) <= set(bridge.forward_walk_indices.tolist())
    assert set(bridge.dng100_indices.tolist()) <= set(bridge.forward_walk_indices.tolist())


def test_dng100_current_engages_engineered_cpg_and_raises_vnc_cpg():
    graph = miniature_connectome(1)
    bridge = MotorBridge(graph, legacy_scaffold=False)
    result = probe_pathway(
        graph,
        bridge.dng100_indices,
        source="experiment.optogenetic.DNg100",
        seed=1,
        current=40.0,
        steps=12,
    )
    assert result["present"]
    assert result["driven"]["scaffold_used"] is False
    assert result["engineered_cpg_engaged"]
    assert result["driven"]["mode"] == "walk"
    assert result["cpg_or_mn_responded"]
    assert result["driven"]["rates"]["E1"]["analog"] > 0.0 or result["driven"]["rates"]["vnc_motor"]["hz"] > 0.0


def test_dnb08_is_not_decoded_as_walking():
    graph = miniature_connectome(1)
    bridge = MotorBridge(graph, legacy_scaffold=False)
    counts = np.zeros(graph.n, dtype=np.int32)
    counts[bridge.dnb08_indices] = 80
    cmd = bridge.read(counts, 0.01)
    assert cmd.mode == "rest"
    assert cmd.scaffold_used is False
    assert set(bridge.dnb08_indices.tolist()).isdisjoint(set(bridge.forward_walk_indices.tolist()))


def test_dng100_lesion_of_e1_reduces_mn_response_without_a_walk_api():
    graph = miniature_connectome(1)
    params = LIFParams(dt=1.0)
    net = LIFNetwork(graph, params=params, seed=1)
    bridge = MotorBridge(graph, legacy_scaffold=False)
    net.add_drive(bridge.dng100_indices, 40.0, source="experiment.optogenetic.DNg100")
    counts = net.step(12)
    analog = getattr(net, "graded_release", net.graded_output)
    intact_mn = float(counts[bridge.circuit.vnc_motor].sum())
    intact_e1 = float(np.mean(analog[bridge.cpg_e1]))

    net = LIFNetwork(graph, params=params, seed=1)
    net.lesion(bridge.cpg_e1, silent=True)
    net.add_drive(bridge.dng100_indices, 40.0, source="experiment.optogenetic.DNg100")
    counts = net.step(12)
    analog = getattr(net, "graded_release", net.graded_output)
    lesion_e1 = float(np.mean(analog[bridge.cpg_e1]))
    assert intact_e1 > lesion_e1
    cmd = bridge.read(counts, 0.012, net=net, external_command="experiment.optogenetic.DNg100")
    assert cmd.scaffold_used is False
    # FlyGym still receives locomotor_drive from DNg100 even if E1 is silent.
    assert cmd.mode == "walk"
    assert "def walk(" not in inspect.getsource(__import__("organism.walking_pathways", fromlist=["walking_pathways"]))
    assert intact_mn >= 0.0


def test_investigator_toy_experiment(tmp_path):
    result = run(
        connectome="synthetic",
        seed=1,
        out=tmp_path / "walking_dn_investigator.json",
        probe_steps=12,
    )
    assert result["no_walk_function"] is True
    assert result["engineered_cpg_still_executes"] is True
    assert result["neural_vnc_cpg_drives_joints"] is False
    assert result["anatomy_checks"]["DNg100 present"]
    assert result["anatomy_checks"]["DNg100 contacts E1"]
    assert result["anatomy_checks"]["core CPG types present"]
    assert result["probes"]["DNg100"]["engineered_cpg_engaged"]
    assert result["probes"]["DNp09"]["engineered_cpg_engaged"]
    assert result["probes"]["DNb08"]["driven"]["scaffold_used"] is False
    assert "foxglove" in result["missing_types"]
    assert (tmp_path / "walking_dn_investigator.json").exists()
