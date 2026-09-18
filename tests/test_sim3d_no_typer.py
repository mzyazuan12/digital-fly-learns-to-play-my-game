"""The 3D fly is not a Shiritori word bot."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_desk_script_does_not_type_words():
    src = (ROOT / "sim3d/js/main.js").read_text()
    assert "liveSuggest" not in src
    assert "typeWord" not in src
    assert "pick_live_word" not in src
    assert "for (let turn" not in src
    assert "The fly does not play" in src
    assert "BodyController" not in src
    assert "liveKey" in src
    assert "liveMode" in src
    html = (ROOT / "sim3d/index.html").read_text()
    assert "btn-play" not in html
    assert "btn-manual" not in html
    assert "Manual override" not in html
    assert "btn-casual" in html
    assert "btn-featherine" in html
    assert "btn-lambdadelta" in html
    assert "You play" in html


def test_fly_visual_is_not_a_dark_hovering_blob():
    src = (ROOT / "sim3d/js/fly.js").read_text()
    assert "SphereGeometry" not in src
    assert "0x7cffd4" not in src
    assert "lookFor" not in src
    assert "MeshPhysicalMaterial" in src
    assert "MeshStandardMaterial" in src
    assert "0xcc0700" in src
    assert "0xac5924" in src
    assert "setFlying(v)" in src
    assert "applyBodies" in src
    assert "mujocoToThreeGeometry" in src
    assert "nodeFor" in src
    assert "setGrooming" in src
    assert "flybody.glb" in src
    assert "pose.position.y = restY +" not in src
    assert "drawGame" not in (ROOT / "sim3d/js/monitor.js").read_text()
    assert "buildPhysicalKeys" in (ROOT / "sim3d/js/keyboard.js").read_text()
    view = (ROOT / "sim3d/js/living_view.js").read_text()
    assert "startLivingRoom" in view
    assert "setFlying(false)" in view
    assert "NeutralToneMapping" in view
    assert "fly.root.rotation.set(0, -p.heading_rad" not in view
    assert "KeyW" in view
    assert "/api/browser/frame" in view
    room = (ROOT / "sim3d/js/living_room.js").read_text()
    assert "buildKeyboard" in room
    assert "buildMonitor" in room
    assert "buildPhysicalKeys" not in room
    assert "RugKeyboard" not in room
    assert "drawGame" not in room


def test_live_site_refuses_lexicon_suggestions():
    src = (ROOT / "sim3d/live_site.py").read_text()
    assert "pick_live_word" not in src
    assert "Lexicon autoplay is disabled" in src


def test_organism_loop_is_not_a_player():
    src = (ROOT / "organism/loop.py").read_text()
    assert "prefix" not in src
    assert "liveSuggest" not in src
    assert "KeyboardDecoder" not in src


def test_habitat_runtime_has_no_game_fields():
    from sim3d.organism_runtime import OrganismRuntime

    rt = OrganismRuntime(seed=0)
    snap = rt.snapshot()
    assert snap["world"] == "living_room"
    assert "prefix" not in snap
    assert "correct_answer" not in snap
    assert snap["ok"]
    assert 3400 < snap["x_mm"] < 4100
    assert 400 < snap["y_mm"] < 850
    layout = rt.room_layout()
    assert layout["width_mm"] >= 4000
    assert any(s["name"] == "desk" for s in layout["solids"])
    assert layout["keys"]
    assert any(k["label"] == "a" for k in layout["keys"])
    assert all(k["y_mm"] < 900 for k in layout["keys"])


def test_live_spikes_are_from_the_running_lif():
    import time

    from sim3d.organism_runtime import OrganismRuntime

    rt = OrganismRuntime(seed=0)
    time.sleep(0.12)
    snap = rt.snapshot()
    act = snap["activity"]
    layout = rt.brain_layout()
    assert act["n_full"] == snap["neurons"]
    assert layout["n_display"] == act["n_display"]
    assert layout["coords_source"] in {"atlas", "somaLocation"}
    assert "atlas" in layout
    assert "cells" in act
    assert all(0 <= i < layout["n_display"] for i in act["spiked"])
    assert set(act["readout"]) >= {"walk", "steer_l", "steer_r", "rest"}
    assert "prefix" not in act
    assert "fired_edges" in act
    assert layout["n_edges"] == len(layout["edge_pre"])
    # At least one window of real counts should have been published.
    assert act["window_ms"] > 0


def test_organism_page_has_real_brain_and_shiritori():
    html = (ROOT / "sim3d/organism.html").read_text()
    assert "data-cmd" not in html
    assert "Walk</button>" not in html
    assert "neuroglancer-demo.appspot.com" in html
    assert "Ask the fly to play" in html
    assert "FlyEM" in html
    assert "Living room" in html
    assert 'id="viewport"' in html
    assert 'id="brain-panel"' in html
    assert 'id="brain-panel" hidden' not in html
    assert "btn-featherine" in html
    assert "btn-lambdadelta" in html
    assert 'id="btn-brain"' in html
    assert 'id="game-panel" hidden' in html
    assert "fly-frame" not in html
    js = (ROOT / "sim3d/js/organism.js").read_text()
    assert "/api/invite" in js
    assert "startLivingRoom" in js
    assert "applyBodies" in js
    assert "/api/browser/mode" in js
    assert "setBrainVisible" in js
    assert "rebase" not in js
    assert "/api/live/frame" not in js
    serve = (ROOT / "sim3d/serve.py").read_text()
    assert '"/organism.html"' in serve
    assert "gait_runtime(start=True)" in serve
    assert "/api/invite" in serve
    assert "KeyboardDecoder" not in serve
    assert "ShiritoriEnv" not in serve
    assert "live = not bool(args.no_live)" in serve
    assert 'default="malecns"' in serve


def test_invite_is_sensory_not_a_word_solver():
    import inspect

    from organism.body import Pose
    from organism.loop import SensorimotorLoop
    from sim3d.organism_runtime import OrganismRuntime

    loop_src = inspect.getsource(SensorimotorLoop)
    runtime_src = (ROOT / "sim3d/organism_runtime.py").read_text()
    assert "play_shiritori" not in loop_src
    assert "pick_live_word" not in runtime_src
    assert "KeyboardDecoder" not in runtime_src
    assert "ShiritoriEnv" not in runtime_src
    rt = OrganismRuntime(seed=0)
    rt.set_paused(True)
    before = rt.fly.physiology.state.novelty
    snap = rt.invite()
    assert snap["invite"] is True
    assert rt.world.monitor_on
    assert rt.fly.physiology.state.novelty >= before
    assert snap["game"]["phase"] == "menu"
    key = next(k for k in rt.world.keys if k.label == "q")
    rt.fly.body.teleport(Pose(x_mm=key.x_mm, y_mm=key.y_mm, z_mm=0.5))
    pressed = rt._apply_key_contacts()
    assert pressed == "q"
    assert rt.key_buffer == "q"
    assert rt._apply_key_contacts() == ""
    rt.fly.body.teleport(Pose(x_mm=key.x_mm + 80, y_mm=key.y_mm, z_mm=0.5))
    assert rt._apply_key_contacts() == ""
    rt._last_key_t = 0.0
    rt.fly.body.teleport(Pose(x_mm=key.x_mm, y_mm=key.y_mm, z_mm=0.5))
    assert rt._apply_key_contacts() == "q"

    class FakeLive:
        def __init__(self):
            self.ops = []

        def call(self, op, timeout=12.0, **kw):
            self.ops.append((op, kw))
            return {"ok": True, "phase": "playing"}

        def snapshot(self):
            return {
                "ok": True,
                "phase": "playing",
                "prefix": "q",
                "buffer": "q",
                "lastWord": "",
                "url": "https://shiritori.lol/",
                "message": "live",
            }

    import time

    live = FakeLive()
    typed = OrganismRuntime(seed=0, live=live)
    typed.set_paused(True)
    typed.invite()
    q = next(k for k in typed.world.keys if k.label == "q")
    typed.fly.body.teleport(Pose(x_mm=q.x_mm, y_mm=q.y_mm, z_mm=0.5))
    assert typed._apply_key_contacts() == "q"
    deadline = time.time() + 1.0
    while time.time() < deadline and not any(op == "key" for op, _ in live.ops):
        time.sleep(0.02)
    assert any(op == "key" and kw.get("key") == "q" for op, kw in live.ops)
    assert typed.snapshot()["game"]["source"] == "shiritori.lol"


def test_fly_motion_matches_adult_walking_scale():
    fly_js = (ROOT / "sim3d/js/fly.js").read_text()
    view = (ROOT / "sim3d/js/living_view.js").read_text()
    runtime = (ROOT / "sim3d/organism_runtime.py").read_text()
    loop = (ROOT / "organism/loop.py").read_text()
    body = (ROOT / "organism/body.py").read_text()
    assert "physics_substeps=40" not in runtime
    assert "max_catchup_s" in runtime
    assert "elapsed / timestep" in runtime
    assert "visualScale" not in fly_js
    assert "perch" not in fly_js
    assert "lookFor" not in fly_js
    assert "originXmm" not in view
    assert "visualScale" not in view
    assert "display: 8" in view
    assert "plantY" in view
    assert "display = 1" in fly_js
    assert "levelStance" in fly_js
    assert "setFromUnitVectors" in fly_js
    assert "scene.add(fly.root)" in view
    assert "_shape_command" not in loop
    assert "0.9, 1.25" not in loop
    assert "walk_gain: float = 18.0" in body
    gait = (ROOT / "organism/gait.py").read_text()
    assert "geom_transforms" in gait
    assert "fusestatic = False" in gait
    assert "CONTROL_DECIMATION = 40" in gait
    assert "make_locomotion_fly" in gait
    assert "kind = \"neuromechfly\"" in gait or "kind = 'neuromechfly'" in gait
    assert "fly.add_vision()" in gait
    runtime_py = (ROOT / "sim3d/organism_runtime.py").read_text()
    assert "brain_ticks=40 if physics" in runtime_py
    assert "brain_period_s" in runtime_py
    assert "step_body" in runtime_py
    assert "body_transforms" in runtime_py
    assert "geom_transforms" in runtime_py
    assert "_walk_off" in runtime_py
    assert "walking_bout_s" in (ROOT / "organism/bridge.py").read_text()
    assert "mode = \"fly\"" in (ROOT / "organism/bridge.py").read_text() or "mode = 'fly'" in (ROOT / "organism/bridge.py").read_text()


def test_flybody_meshes_bind_to_leg_bodies():
    from sim3d.export_flybody import geom_to_body

    bodies = {
        "c_thorax",
        "c_head",
        "lf_tibia",
        "lf_tarsus5",
        "l_wing",
        "l_pedicel",
        "c_abdomen12",
    }
    assert geom_to_body("lf_tibia_body", bodies) == "lf_tibia"
    assert geom_to_body("lf_tarsus5_brown", bodies) == "lf_tarsus5"
    assert geom_to_body("l_wing_membrane", bodies) == "l_wing"
    assert geom_to_body("c_head_red", bodies) == "c_head"
    assert geom_to_body("l_antenna_black", bodies) == "l_pedicel"
    assert geom_to_body("c_abdomen1_body", bodies) == "c_abdomen12"
    view = (ROOT / "sim3d/js/brainview.js").read_text()
    assert "clippingPlanes" in view
    assert "brain-slice" in view
    assert "fired_edges" in view
    assert "firing-cells" in view
    html = (ROOT / "sim3d/habitat.html").read_text()
    assert "firing-cells" in html
    assert "data-slice-axis" in html
    assert "Drosophila CNS" in html
    css = (ROOT / "sim3d/styles.css").read_text()
    assert "Inter" in css
    assert "--void: #050505" in css
