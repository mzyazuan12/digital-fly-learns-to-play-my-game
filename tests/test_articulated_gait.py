"""Brain-disconnected NeuroMechFly gait. MaleCNS is not loaded."""

from pathlib import Path

import numpy as np
import pytest

from organism.body import flygym_available
from organism.gait import (
    ArticulatedFly,
    RootMotionForbidden,
    evaluate_battery,
    run_battery,
)

pytestmark = pytest.mark.physics

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not flygym_available(), reason="FlyGym / MuJoCo is not installed")
def test_gait_module_does_not_write_the_root():
    src = (ROOT / "organism/gait.py").read_text()
    assert "qpos[0:3]" not in src
    assert "qpos[0:7]" not in src
    assert "data.qpos[0:3]" not in src
    physics = (ROOT / "organism/physics.py").read_text()
    assert "data.qpos[0:3]" not in physics
    assert "RootMotionForbidden" in physics


@pytest.mark.skipif(not flygym_available(), reason="FlyGym / MuJoCo is not installed")
def test_neuromechfly_walks_by_articulating_legs():
    fly = ArticulatedFly()
    model = fly.inspect_model()
    assert model["kind"] == "neuromechfly"
    assert model["brain"] is None
    assert model["hinge_joints"] >= 42
    assert model["free_joints"] == 1
    assert model["nu"] >= 42
    assert any("_tarsus5" in name for name in model["bodies"])
    assert any(name.endswith("c_head") or name == "c_head" for name in model["bodies"])
    assert any(name.endswith("l_eye") or name == "l_eye" for name in model["bodies"])
    assert all(int(i) >= 0 for i in fly._internal_body_ids)
    geoms = {t.segment.rsplit("/", 1)[-1]: t.pos_mm for t in fly.geom_transforms()}
    assert "l_eye" in geoms and "r_eye" in geoms
    assert float(np.linalg.norm(geoms["l_eye"] - geoms["r_eye"])) > 0.15

    stand_angles = []
    for _ in range(400):
        snap = fly.step("stand")
        stand_angles.append(snap.joint_angles)
    stand = np.stack(stand_angles)

    walk_angles = []
    walk_xyz = []
    walk_contact = []
    for _ in range(2500):
        snap = fly.step("walk")
        walk_angles.append(snap.joint_angles)
        walk_xyz.append(snap.thorax_xyz_mm)
        walk_contact.append(snap.contact_found)
    walk_angles = np.stack(walk_angles)
    walk_xyz = np.stack(walk_xyz)
    walk_contact = np.stack(walk_contact)

    std = np.std(walk_angles, axis=0)
    names = fly.joint_names
    per_leg = {}
    for leg in ("lf", "lm", "lh", "rf", "rm", "rh"):
        mask = np.array([f"{leg}_" in n for n in names])
        per_leg[leg] = float(np.max(std[mask])) if mask.any() else 0.0
    assert all(v > 0.02 for v in per_leg.values()), per_leg
    disp = float(np.linalg.norm(walk_xyz[-1, :2] - walk_xyz[0, :2]))
    assert disp > 1.0, f"thorax did not translate from physics: {disp} mm"
    assert float(np.mean(walk_xyz[:, 2])) > 0.2
    assert float(np.mean(walk_contact > 0.5)) > 0.1
    assert float(np.max(np.std(stand, axis=0))) < float(np.max(std))
    with pytest.raises(RootMotionForbidden):
        fly.teleport()


@pytest.mark.skipif(not flygym_available(), reason="FlyGym / MuJoCo is not installed")
def test_gait_battery_short():
    report = run_battery(
        durations=(
            ("stand", 0.05),
            ("walk", 0.25),
            ("walk_slow", 0.12),
            ("walk_fast", 0.12),
            ("turn_left", 0.18),
            ("turn_right", 0.18),
            ("stop", 0.05),
        ),
        sample_every=20,
        report_path=ROOT / "outputs" / "gait_battery_test.json",
    )
    ok, errors = evaluate_battery(report)
    assert ok, errors
    assert report["brain"] is None
    assert report["body"]["kind"] == "neuromechfly"


def test_motor_bridge_mapping_file_is_explicit():
    payload = (ROOT / "data/motor_bridge_mapping.json").read_text()
    assert "malecns_body_id" in payload
    assert "decoded_signal" in payload
    assert "left_steering_drive" in payload
    assert "forward_locomotor_drive" in payload
    assert "HybridTurningController" in payload
    assert "world_x" in payload
