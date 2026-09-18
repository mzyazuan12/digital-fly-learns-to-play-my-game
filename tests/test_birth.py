"""Birth creates a persistent individual. It does not resurrect the EM specimen."""

from organism import VirtualFly, birth_fly
from organism.config import BIRTH_DEFINITION, MODEL_VERSION
from worlds import empty_arena, stimulus_arena
from organism.body import Pose
import numpy as np


def test_birth_writes_persistent_store(tmp_path):
    report = birth_fly("fly_001", seed=3, directory=tmp_path / "fly_001")
    assert report["individual_id"] == "fly_001"
    assert report["consciousness_claimed"] is False
    assert report["legacy_scaffold"] is False
    assert "neural_state.npz" in report["files"] or (tmp_path / "fly_001" / "brain.npz").exists()
    assert (tmp_path / "fly_001" / "life_history.sqlite").exists()
    assert (tmp_path / "fly_001" / "metabolic_state.json").exists()
    assert (tmp_path / "fly_001" / "neuromodulatory_state.json").exists()
    assert (tmp_path / "fly_001" / "synaptic_state.npz").exists()
    assert (tmp_path / "fly_001" / "birth.json").exists()
    loaded = VirtualFly.load(tmp_path / "fly_001")
    assert loaded.identity.fly_id == "fly_001"
    assert loaded.identity.model_version == MODEL_VERSION
    assert BIRTH_DEFINITION.split()[0] == "Instantiate"


def test_two_births_diverge_after_experience(tmp_path):
    a = VirtualFly.birth("fly_a", seed=1, directory=tmp_path / "fly_a")
    b = VirtualFly.birth("fly_b", seed=1, directory=tmp_path / "fly_b")
    assert np.allclose(a.net.v, b.net.v)
    a.inhabit(empty_arena())
    a.run(25)
    b.inhabit(stimulus_arena(side="left"), spawn=Pose(x_mm=40.0, y_mm=40.0))
    b.run(25)
    assert not np.allclose(a.net.v, b.net.v)
    assert a.net.sim_ms == b.net.sim_ms


def test_second_birth_loads_existing(tmp_path):
    first = birth_fly("fly_001", seed=11, directory=tmp_path / "fly_001")
    second = birth_fly("fly_001", seed=99, directory=tmp_path / "fly_001")
    assert first["loaded_existing"] is False
    assert second["loaded_existing"] is True
    assert second["seed"] == 11


def test_different_birth_seeds_diverge_immediately(tmp_path):
    a = VirtualFly.birth("fly_a", seed=1, directory=tmp_path / "fly_a")
    b = VirtualFly.birth("fly_b", seed=2, directory=tmp_path / "fly_b")
    assert not np.allclose(a.net.v, b.net.v)
