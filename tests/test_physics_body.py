import numpy as np
import pytest

from organism.body import Pose, flygym_available
from organism.fly import VirtualFly
from worlds import empty_arena

pytestmark = pytest.mark.physics


@pytest.mark.skipif(not flygym_available(), reason="FlyGym / MuJoCo is not installed")
def test_flygym_body_steps():
    fly = VirtualFly.hatch(seed=0, connectome="synthetic")
    fly.inhabit(empty_arena(), physics=True, spawn=Pose(z_mm=0.8), brain_ticks=1)
    assert fly.body.kind == "neuromechfly"
    fly.run(3)
    end = fly.body.pose
    assert fly.body.kind != "mock_unicycle"
    obs = fly.body.sense(fly.world)
    if obs.ommatidia is not None:
        assert obs.ommatidia.shape[0] == 2
        assert obs.ommatidia.shape[1] >= 100
    assert end.z_mm > 0.2
    assert all(
        np.isfinite(v)
        for v in (end.x_mm, end.y_mm, end.z_mm, end.heading_rad)
    )
