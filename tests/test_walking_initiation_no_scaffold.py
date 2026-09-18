"""NO_SCAFFOLD walking initiation: no timer, no fallback, no named gait command."""

from organism.config import NO_SCAFFOLD, LEGACY_POLICY, motor_fidelity_level, MotorMode
from organism.bridge import MotorBridge, format_walk_trace
from organism.neuromodulation import MODULATORY_EFFECTS
from organism.toy import miniature_connectome
from experiment.walking_initiation_no_scaffold import run, probe_dnp09_on
from flybrain.network import LIFNetwork, LIFParams
from flybrain.neuron_model import NeuronKind
import numpy as np


def test_no_scaffold_disables_timers_and_fallbacks():
    assert NO_SCAFFOLD.allow_behavior_timers is False
    assert NO_SCAFFOLD.allow_motor_fallbacks is False
    assert NO_SCAFFOLD.allow_named_gait_commands is False
    assert NO_SCAFFOLD.allow_root_motion is False
    assert NO_SCAFFOLD.allow_privileged_world_state is False
    assert NO_SCAFFOLD.allow_pretrained_low_level_gait is True
    assert LEGACY_POLICY.allow_behavior_timers is True


def test_motor_fidelity_level_is_identified_dns_to_cpg():
    assert motor_fidelity_level(MotorMode.ENGINEERED_CPG) == 1


def test_graded_cell_delivers_without_spiking():
    graph = miniature_connectome(0)
    net = LIFNetwork(graph, params=LIFParams(dt=1.0), seed=0)
    net.intrinsic_noise_std = 0.0
    graded = np.flatnonzero(net.is_graded)
    assert graded.size >= 1
    i = int(graded[0])
    start = int(net.ptr[i])
    end = int(net.ptr[i + 1])
    assert end > start
    posts = net.post[start:end]
    net.drive.fill(0)
    net.v.fill(net.v_rest)
    net.v[i] = net.v_rest + 8.0
    net.step(1)
    assert bool(net.last_spikes[i]) is False
    assert net.models.kind[i] == NeuronKind.GRADED_RATE.value
    assert float(net.graded_output[i]) > 0.0
    assert float(np.abs(net.g[posts]).sum()) > 0.0
    assert net.last_n_graded_deliveries >= 1


def test_graded_cell_at_rest_does_not_leak():
    graph = miniature_connectome(0)
    net = LIFNetwork(graph, params=LIFParams(dt=1.0), seed=0)
    net.intrinsic_noise_std = 0.0
    graded = np.flatnonzero(net.is_graded)
    i = int(graded[0])
    net.drive.fill(0)
    net.v.fill(net.v_rest)
    net.g.fill(0)
    net.step(1)
    assert float(net.graded_output[i]) < 1e-3
    assert net.last_n_graded_deliveries == 0


def test_birth_membranes_are_not_identical():
    graph = miniature_connectome(0)
    net = LIFNetwork(graph, params=LIFParams(dt=1.0), seed=7)
    assert float(net.v.std()) > 0.2
    assert abs(float(net.v.mean()) - float(net.params.v_rest)) < 1.0


def test_modulatory_effects_have_provenance():
    assert MODULATORY_EFFECTS
    for item in MODULATORY_EFFECTS:
        assert item.confidence in {"MEASURED", "LITERATURE_DERIVED", "INFERRED", "ASSUMED"}
        assert item.source
        assert "walk()" not in item.effect


def test_walk_trace_lists_no_external_command_on_rest():
    graph = miniature_connectome(1)
    bridge = MotorBridge(graph, legacy_scaffold=False)
    counts = np.zeros(graph.n, dtype=np.int32)
    cmd = bridge.read(counts, 0.04, walking_drive=1.0, walking_bout_s=5.0)
    assert cmd.mode == "rest"
    assert cmd.scaffold_used is False
    text = format_walk_trace(bridge.last_trace)
    assert "External command:       NONE" in text
    assert "Behavior timer:         NONE" in text
    assert "REST" in text
    assert bridge.last_trace["motor_fidelity_level"] == 1


def test_no_scaffold_experiment_toy():
    result = run(connectome="synthetic", seed=1, spontaneous_steps=40)
    required = result["checklist_pass_required"]
    assert all(required.values()), required
    assert result["dnp09_probe"]["neural_authority"]
    assert result["conditions"]["normal"]["scaffold_used"] is False
    assert result["standing"]["n_rest"] == result["standing"]["n"]
    # A statue is allowed. Do not require n_walk > 0.
    assert result["statue_is_a_result"] is True
    assert result["policy"]["allow_behavior_timers"] is False
    assert "WALK INITIATION TRACE" in result["dnp09_probe"]["intact"]["trace_text"]


def test_optogenetic_dnp09_probe_still_causal():
    graph = miniature_connectome(1)
    result = probe_dnp09_on(graph, seed=1)
    assert result["neural_authority"]
    assert result["intact"]["scaffold_used"] is False
    assert result["lesion"]["mode"] == "rest"
    assert result["restored"]["mode"] == "walk"
