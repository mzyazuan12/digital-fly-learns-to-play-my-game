"""The organism is a fly, not a Shiritori solver."""

import inspect

import numpy as np
import pytest

from flybrain.network import LIFNetwork, LIFParams
from organism import VirtualFly
from organism.body import Pose
from organism.bridge import MotorBridge
from organism.fly import VirtualFly as VirtualFlyClass
from organism.loop import SensorimotorLoop
from organism.sensory import SensoryObservation, SensorySystem
from organism.toy import miniature_connectome
from worlds import empty_arena, living_room, spawn_on_rug, stimulus_arena, workstation
from worlds.base import World


FORBIDDEN_BRAIN_APIS = (
    "play_shiritori",
    "solve_word",
    "navigate_to_keyboard",
    "correct_answer",
    "keyboard_key",
    "target_position",
)


def test_brain_has_no_task_methods():
    for name in FORBIDDEN_BRAIN_APIS:
        assert not hasattr(VirtualFly, name)
    source = inspect.getsource(VirtualFlyClass)
    for name in ("play_shiritori", "solve_word", "navigate_to_keyboard"):
        assert name not in source
    loop_src = inspect.getsource(SensorimotorLoop)
    assert "KeyboardDecoder" not in loop_src
    assert "play_shiritori" not in loop_src
    sense_src = inspect.getsource(SensorySystem)
    assert "keyboard_key" not in sense_src
    assert "correct_answer" not in sense_src
    assert "target_position" not in sense_src


def test_world_has_no_global_reward():
    world = living_room()
    assert "reward" not in World.__dataclass_fields__
    assert world.describe()["reward"] is None


def test_sensory_observation_is_receptor_level():
    fields = set(SensoryObservation.__dataclass_fields__)
    assert "left_eye" in fields
    assert "joint_angles" in fields
    assert "prefix" not in fields
    assert "keyboard_key" not in fields
    assert "correct_answer" not in fields
    assert "target_position" not in fields


def test_no_toy_behavior_shortcuts():
    graph = miniature_connectome(0)
    assert graph.report.get("toy_phototaxis_wiring") is False
    toy_mod = inspect.getsource(__import__("organism.toy", fromlist=["toy"]))
    assert "CONNECT_L" not in toy_mod
    assert "CONNECT_R" not in toy_mod
    assert "np.arange(24" not in toy_mod
    fly = VirtualFly.hatch(seed=0, connectome="synthetic")
    assert fly.connectome.report.get("toy_phototaxis_wiring") is False


def test_motor_bridge_resolves_identified_types():
    graph = miniature_connectome(1)
    bridge = MotorBridge(graph)
    assert bridge.notes["fallback"] == "identified_types"
    names = {p.name: p for p in bridge.pathways}
    assert "DNp09" in names["walk_initiation"].resolved_types
    assert "DNa02" in names["steer_left"].resolved_types
    assert names["steer_left"].side == "L"
    assert names["steer_right"].side == "R"
    assert names["reverse"].indices.size == 0
    assert names["flight"].indices.size >= 1
    assert names["groom"].indices.size >= 1
    fly = VirtualFly.hatch(seed=1, connectome="synthetic")
    assert fly.bridge.walk_indices.size >= 1


def test_motor_bridge_silent_dns_do_not_walk_without_scaffold():
    graph = miniature_connectome(1)
    neural = MotorBridge(graph, legacy_scaffold=False)
    counts = np.zeros(graph.n, dtype=np.int32)
    rest = neural.read(counts, 0.04)
    assert rest.mode == "rest"
    assert rest.left + rest.right == 0.0
    assert rest.scaffold_used is False
    assert rest.neural_only is True
    from organism.bridge import NonNeuralMotorAuthority

    with pytest.raises(NonNeuralMotorAuthority):
        neural.read(
            counts, 0.04, walking_drive=0.9, walking_bout_s=2.0, grooming_drive=0.9, flight_drive=0.9
        )

    legacy = MotorBridge(graph, legacy_scaffold=True)
    walk = legacy.read(
        counts, 0.04, walking_drive=0.7, walking_bout_s=1.2, grooming_drive=0.2, flight_drive=0.0
    )
    assert walk.mode == "walk"
    assert walk.scaffold_used is True
    groom = legacy.read(
        counts, 0.04, walking_drive=0.1, walking_bout_s=0.0, grooming_drive=0.72, flight_drive=0.0
    )
    assert groom.mode == "groom"
    fly = legacy.read(
        counts, 0.04, walking_drive=0.1, walking_bout_s=0.0, grooming_drive=0.1, flight_drive=0.8
    )
    assert fly.mode == "fly"


def test_autonomous_loop_does_not_use_a_walk_timer():
    fly = VirtualFly.hatch(seed=0, connectome="synthetic", legacy_scaffold=False)
    x, y, z = spawn_on_rug()
    fly.inhabit(living_room(), spawn=Pose(x_mm=x, y_mm=y, z_mm=z))
    records = fly.run(120)
    assert fly.legacy_scaffold is False
    assert fly.world.name == "living_room"
    assert all(not r.scaffold_used for r in records)
    walked = [r for r in records if r.mode == "walk"]
    for rec in walked:
        assert rec.walk_hz > 0.0 or rec.walk_trace > 0.0
    sources = {s.value for r in records for s in r.sources}
    assert "developer_override" not in sources


def test_loop_has_no_task_branch():
    src = inspect.getsource(SensorimotorLoop.step)
    assert "if wall" not in src
    assert "find_food" not in src
    assert "world.reward" not in src
    assert "play_shiritori" not in src


def test_physiology_modulates_neurons_not_actions():
    import organism.physiology as phys_mod

    src = inspect.getsource(phys_mod.Physiology)
    assert "find_food" not in src
    assert "neuromodulation" in src or "add_drive" in src
    fly = VirtualFly.hatch(seed=2, connectome="synthetic")
    fly.physiology.step(0.05, walking=True, contact=0.0, odor=0.0, vision=0.2)
    assert 0.0 <= fly.physiology.state.hunger <= 1.0
    fly.physiology.salient_event(1.0)
    assert fly.physiology.state.novelty > 0.4
    stance = fly.physiology.__class__(seed=3)
    for _ in range(40):
        stance.step(0.05, walking=False, contact=1.0, odor=0.0, vision=0.0)
    assert stance.state.flight_bout_s == 0.0
    assert stance.state.flight_drive < 0.35
    assert "play_shiritori" not in src


def test_brain_survives_world_change():
    fly = VirtualFly.hatch(seed=2, connectome="synthetic")
    fly.inhabit(empty_arena())
    fly.run(8)
    v = fly.net.v.copy()
    g = fly.net.g.copy()
    sim_ms = fly.net.sim_ms
    phys = fly.physiology.state.snapshot()
    fly.detach()
    fly.inhabit(stimulus_arena(side="right"), spawn=Pose(x_mm=1.0))
    assert fly.net.sim_ms == sim_ms
    assert np.allclose(fly.net.v, v)
    assert np.allclose(fly.net.g, g)
    assert fly.physiology.state.walking_drive == pytest.approx(phys["walking_drive"])
    fly.run(5)
    assert fly.net.sim_ms > sim_ms


def test_save_reload_same_individual(tmp_path):
    fly = VirtualFly.hatch(seed=3, connectome="synthetic")
    fly.inhabit(empty_arena())
    fly.run(12)
    fly.net.functional_gain[0] = 1.7
    fly.net._rebuild_weights()
    hunger = fly.physiology.state.hunger
    path = fly.save(tmp_path / "unit.fly")
    loaded = VirtualFly.load(path)
    assert loaded.identity.fly_id == fly.identity.fly_id
    assert loaded.identity.seed == fly.identity.seed
    assert np.allclose(loaded.net.v, fly.net.v)
    assert np.allclose(loaded.net.g, fly.net.g)
    assert np.allclose(loaded.net.efficacy, fly.net.efficacy)
    assert loaded.net.sim_ms == fly.net.sim_ms
    assert loaded.plasticity.n_updates == fly.plasticity.n_updates
    assert loaded.physiology.state.hunger == pytest.approx(hunger)
    loaded.inhabit(workstation())
    loaded.run(3)
    assert loaded.world.name == "shiritori_workstation"


def test_developer_teleport_is_not_learned():
    fly = VirtualFly.hatch(seed=4, connectome="synthetic")
    fly.inhabit(empty_arena())
    fly.run(5)
    fly.developer.teleport(50.0, -20.0, heading_rad=1.2)
    assert fly.body.pose.x_mm == pytest.approx(50.0)
    last = fly.provenance.records[-1]
    assert last.developer
    assert last.sources[0].value == "developer_override"
    learned = fly.provenance.learned_behavior_records()
    assert all(not r.developer for r in learned)


def test_lif_params_are_explicit_modeling_choices():
    net = LIFNetwork(miniature_connectome(0), params=LIFParams(dt=1.0), seed=0)
    assert net.params.v_rest == -52.0
    assert net.connectome.report.get("dataset_id") == "miniature"
    assert net.connectome.report.get("toy_phototaxis_wiring") is False


def test_stimulus_is_seen_but_not_a_hardwired_taxis():
    fly = VirtualFly.hatch(seed=1, connectome="synthetic")
    fly.inhabit(stimulus_arena(side="left"), spawn=Pose(x_mm=40.0, y_mm=40.0))
    records = fly.run(40)
    assert np.mean([r.left_eye for r in records]) > np.mean([r.right_eye for r in records])
    toy_mod = inspect.getsource(__import__("organism.toy", fromlist=["toy"]))
    assert "toy_phototaxis_wiring\": True" not in toy_mod


def test_mock_body_walks_at_adult_drosophila_speed():
    from organism.body import MockBody
    from organism.bridge import MotorCommand

    body = MockBody()
    body.dt_s = 0.001
    body.apply_descending(MotorCommand(1.0, 1.0, "walk", 1.0, 0.0, 0.0, ()))
    for _ in range(250):
        body.step_physics()
    dist = float(np.hypot(body.pose.x_mm, body.pose.y_mm))
    # 18 mm/s × 0.25 s. Adult walking is ~10–25 mm/s.
    assert 3.5 < dist < 5.5
