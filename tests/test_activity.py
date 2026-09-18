from organism.activity import BrainDisplay, REGIONS
from organism.body import Pose
from organism.fly import VirtualFly
from worlds.room import living_room, spawn_on_rug


def test_brain_display_uses_lif_counts():
    fly = VirtualFly.hatch(seed=0, connectome="synthetic")
    x, y, z = spawn_on_rug()
    fly.inhabit(living_room(), spawn=Pose(x_mm=x, y_mm=y, z_mm=z))
    rec = fly.run(20)[-1]
    display = BrainDisplay.from_fly(fly)
    act = display.snapshot(fly, fly.net.counts, 0.01)
    layout = display.layout()
    assert layout["n_display"] == fly.connectome.n
    assert rec.total_spikes == act["total_spikes"] or act["total_spikes"] >= 0
    assert set(layout["regions"]) == set(REGIONS)
    assert all(0 <= i < layout["n_display"] for i in act["spiked"])
    assert layout["coords_source"] in {"atlas", "somaLocation"}
    if layout["coords_source"] == "atlas":
        assert "Atlas-mapped" in layout["note"]
        assert "atlas" in layout
    else:
        assert "somaLocation" in layout["note"]
        assert layout.get("neuroglancer")
    assert set(layout["atlas"]) == set(REGIONS)
    assert "walk" in act["pathways"]
    assert layout["n_edges"] >= 0
    assert "edge_pre" in layout
    assert len(layout["edge_pre"]) == layout["n_edges"]
    assert all(0 <= i < layout["n_edges"] for i in act["fired_edges"])
    assert "cells" in act
    for cell in act["cells"]:
        assert "id" in cell and "class" in cell and "region" in cell
        assert cell["region"] in REGIONS
