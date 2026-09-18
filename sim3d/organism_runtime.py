"""Persistent VirtualFly stepped for the organism page.

MaleCNS drives identified DNs. Those rates go to the NeuroMechFly
HybridTurningController. Tarsus contact on the desk keyboard types on the live
shiritori.lol session — not a local word bot, not a dictionary.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from organism.activity import BrainDisplay
from organism.body import Pose
from organism.fly import VirtualFly
from worlds.room import living_room, spawn_on_rug

ROOT = Path(__file__).resolve().parents[1]


def _load_or_hatch(*, connectome: str, seed: int) -> VirtualFly:
    path = ROOT / "individuals" / "fly_001"
    identity_path = path / "identity.json"
    if identity_path.exists() and ((path / "neural_state.npz").exists() or (path / "brain.npz").exists()):
        identity = json.loads(identity_path.read_text())
        dataset = str(identity.get("connectome_dataset", ""))
        want_male = connectome in {"malecns", "malecns_v1", "full"}
        is_male = "malecns" in dataset
        if want_male == is_male:
            return VirtualFly.load(path)
    return VirtualFly.hatch(seed=seed, connectome=connectome, legacy_scaffold=False)


class OrganismRuntime:
    def __init__(self, *, connectome: str = "synthetic", seed: int = 1, physics: bool = False, live=None):
        self.fly = _load_or_hatch(connectome=connectome, seed=seed)
        room = living_room()
        x, y, z = spawn_on_rug()
        spawn_z = 0.6 if physics else z
        self.fly.inhabit(
            room,
            spawn=Pose(x_mm=x, y_mm=y, z_mm=spawn_z, heading_rad=0.15),
            physics=physics,
            brain_ticks=40 if physics else 10,
        )
        require_soma = self.fly.identity.connectome_dataset in {"malecns_v1", "malecns"}
        self.brain = BrainDisplay.from_fly(self.fly, require_soma=require_soma)
        self.world = room
        self.live = live
        self.lock = threading.Lock()
        self.paused = False
        self._running = True
        self._render_error: str | None = None
        self.invite_active = False
        self.key_buffer = ""
        self._last_key_t = 0.0
        self._last_key = ""
        self._last_letter = ""
        self._last_camp_t = 0.0
        self._last_move_teach = 0.0
        self._keys_held: set[str] = set()
        self._prev_xy = (x, y)
        self._speed_mm_s = 0.0
        self._last_rec = None
        self._activity_cache: dict[str, Any] = {}
        self._bodies_cache: list[dict[str, Any]] = []
        self._bodies_t = 0.0
        self.latest: dict[str, Any] = self._snapshot(None)
        if self.live is not None:
            self.world.present_monitor()
            self.latest = self._snapshot(None)
        self._thread = threading.Thread(target=self._loop, name="organism", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        """Advance MuJoCo at ~1× wall clock. Adult Drosophila walks ~10–25 mm/s.

        MaleCNS LIF windows are slower than the 0.1 ms physics tick and hold
        the GIL, so the body walks on the last descending command and the
        brain refreshes a few times per second instead of blocking every tick.
        """
        timestep = 0.0
        sim = getattr(self.fly.body, "sim", None)
        if sim is not None:
            timestep = float(getattr(sim, "timestep", 0.0) or 0.0)
        brain_dt = self.fly.loop.brain_ticks * self.fly.net.params.dt / 1000.0
        last = time.perf_counter()
        nxt = last
        last_brain = last
        max_catchup_s = 0.05
        brain_period_s = 0.45
        if timestep > 0.0:
            rec = self.fly.loop.infer_command()
            self._commit_brain(rec, phys_dt=0.0)
            last = last_brain = time.perf_counter()
        while self._running:
            if self.paused:
                last = time.perf_counter()
                nxt = last
                time.sleep(0.02)
                continue
            now = time.perf_counter()
            if timestep > 0.0:
                elapsed = min(max(now - last, timestep), max_catchup_s)
                n = max(1, int(round(elapsed / timestep)))
                last += n * timestep
                if last < now - max_catchup_s:
                    last = now - max_catchup_s
                self.fly.loop.step_body(n)
                self._note_speed(n * timestep)
                if now - last_brain >= brain_period_s:
                    wall_dt = min(now - last_brain, 0.25)
                    self._step_physiology(wall_dt)
                    rec = self.fly.loop.infer_command()
                    self._commit_brain(rec, phys_dt=0.0)
                    last_brain = time.perf_counter()
                    last = last_brain
                pressed = self._apply_key_contacts()
                with self.lock:
                    self.latest = self._snapshot(self._last_rec, pressed=pressed)
                if last > time.perf_counter():
                    time.sleep(last - time.perf_counter())
            else:
                rec = self.fly.step()
                self._last_rec = rec
                self._note_speed(brain_dt)
                pressed = self._apply_key_contacts()
                with self.lock:
                    self.latest = self._snapshot(rec, pressed=pressed)
                nxt += brain_dt
                delay = nxt - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    nxt = time.perf_counter()

    def _step_physiology(self, dt_s: float) -> None:
        observation = self.fly.loop._observation
        command = self.fly.bridge.last_command
        contact = 0.0
        odor = 0.0
        vision = 0.0
        if observation is not None:
            if observation.contact.size:
                contact = float(np.max(observation.contact))
            odor = observation.odor
            vision = 0.5 * (observation.left_eye + observation.right_eye)
        elif self.fly.body is not None and self.fly.world is not None:
            try:
                observation = self.fly.body.sense(self.fly.world)
                if observation.contact.size:
                    contact = float(np.max(observation.contact))
                odor = observation.odor
                vision = 0.5 * (observation.left_eye + observation.right_eye)
            except Exception:
                contact = 0.0
        self.fly.physiology.step(
            max(float(dt_s), 1e-4),
            walking=command.mode in {"walk", "reverse"},
            contact=contact,
            odor=odor,
            vision=vision,
        )

    def _commit_brain(self, rec, *, phys_dt: float | None = None) -> None:
        if phys_dt is None:
            phys_dt = self.fly.loop._duration_s
        if phys_dt > 0.0:
            self._step_physiology(phys_dt)
        self.fly.plasticity.observe_activity()
        self.fly.provenance.append(rec)
        self._last_rec = rec
        self._activity_cache = self._activity(rec)

    def _note_speed(self, dt_s: float) -> None:
        pose = self.fly.body.pose
        dt = max(float(dt_s), 1e-6)
        dx = pose.x_mm - self._prev_xy[0]
        dy = pose.y_mm - self._prev_xy[1]
        self._speed_mm_s = float(np.hypot(dx, dy) / dt)
        self._prev_xy = (pose.x_mm, pose.y_mm)
        if self.invite_active and self._speed_mm_s > 3.5:
            now = time.perf_counter()
            if now - self._last_move_teach > 0.5:
                self._last_move_teach = now
                self._teach(0.18)

    def _touch_points_mm(self) -> list[tuple[float, float, float]]:
        inner = getattr(self.fly.body, "inner", None)
        points: list[tuple[float, float, float]] = []
        if inner is not None and hasattr(inner, "body_transforms"):
            for t in inner.body_transforms():
                name = t.segment.lower()
                if name.endswith("tarsus5") or name.endswith("c_thorax") or name.endswith("_tibia"):
                    p = t.pos_mm
                    points.append((float(p[0]), float(p[1]), float(p[2])))
        pose = self.fly.body.pose
        points.append((pose.x_mm, pose.y_mm, pose.z_mm))
        return points

    def _live_phase(self) -> str:
        if self.live is None:
            return ""
        try:
            return str((self.live.snapshot() or {}).get("phase") or "")
        except Exception:
            return ""

    def _keyboard_uv(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        keys = self.world.keys
        if not keys:
            return 0.5, 0.5
        xs = [k.x_mm for k in keys]
        ys = [k.y_mm for k in keys]
        u = (x_mm - min(xs)) / max(1e-3, max(xs) - min(xs))
        v = 1.0 - (y_mm - min(ys)) / max(1e-3, max(ys) - min(ys))
        return float(np.clip(u, 0.04, 0.96)), float(np.clip(v, 0.08, 0.92))

    def _walk_off(self, strength: float = 0.7) -> None:
        """Sensory/neuromodulatory event after a key. Does not start a walk bout."""
        self.fly.physiology.salient_event(strength)

    def _teach(self, reward: float) -> None:
        if abs(float(reward)) < 1e-6:
            return
        self.fly.plasticity.observe_activity()
        self.fly.plasticity.apply_reward(float(reward))

    def _apply_key_contacts(self) -> str:
        if not self.invite_active:
            return ""
        now = time.perf_counter()
        hits: list[str] = []
        for key in self.world.keys:
            hx = key.half_mm
            for x, y, z in self._touch_points_mm():
                if abs(x - key.x_mm) <= hx and abs(y - key.y_mm) <= hx and abs(z - key.z_mm) <= 6.0:
                    hits.append(key.label)
                    break
        held = set(hits)
        rising = [lab for lab in hits if lab not in self._keys_held]
        left = self._keys_held - held
        if left:
            self._teach(0.4)
            self._walk_off(0.55)
        self._keys_held = held
        if not rising:
            if held and now - self._last_key_t > 0.28 and now - self._last_camp_t > 0.45:
                self._last_camp_t = now
                self._teach(-0.25)
                self._walk_off(0.95)
            return ""
        if now - self._last_key_t < 0.12:
            return ""
        label = rising[0]
        phase = self._live_phase()
        pose = self.fly.body.pose
        if phase in {"menu", "finished", "boot"}:
            u, v = self._keyboard_uv(pose.x_mm, pose.y_mm)
            self._send_live_click(u, v)
            self._last_key = label
            self._last_key_t = now
            self._teach(0.85)
            self._walk_off(0.7)
            return f"click:{label}"
        action = "ENTER" if label == "ENTER" else "BACKSPACE" if label == "BACKSPACE" else label.lower()
        self._note_key(action)
        self._send_live_key(action)
        self._last_key = label
        self._last_key_t = now
        self._teach(0.9 if action != self._last_letter else 0.12)
        self._last_letter = action
        self.fly.physiology.salient_event(0.3)
        self._walk_off(0.85)
        return label

    def _note_key(self, action: str) -> None:
        if action == "BACKSPACE":
            self.key_buffer = self.key_buffer[:-1]
        elif action == "ENTER":
            self.key_buffer = ""
        elif len(action) == 1:
            self.key_buffer += action

    def _send_live_key(self, action: str) -> None:
        live = self.live
        if live is None:
            return

        def _go() -> None:
            try:
                live.call("key", timeout=12.0, key=action)
            except Exception:
                return

        threading.Thread(target=_go, name="live-key", daemon=True).start()

    def _send_live_click(self, x: float, y: float) -> None:
        live = self.live
        if live is None:
            return

        def _go() -> None:
            try:
                live.call("click", timeout=12.0, x=float(x), y=float(y))
            except Exception:
                return

        threading.Thread(target=_go, name="live-click", daemon=True).start()

    def _show_live_menu(self) -> None:
        live = self.live
        if live is None:
            return
        try:
            live.call("menu", timeout=45.0)
        except Exception:
            return

    def invite(self) -> dict[str, Any]:
        with self.lock:
            self.invite_active = True
            self.key_buffer = ""
            self._last_key = ""
            self._last_letter = ""
            self._last_key_t = 0.0
            self._keys_held = set()
            self.world.present_monitor()
            self.fly.physiology.salient_event(1.0)
            self.latest = self._snapshot(None)
            snap = dict(self.latest)
        if self.live is not None:
            threading.Thread(target=self._show_live_menu, name="live-menu", daemon=True).start()
        return snap

    def _bodies(self) -> list[dict[str, Any]]:
        now = time.perf_counter()
        if self._bodies_cache and now - self._bodies_t < 0.04:
            return self._bodies_cache
        inner = getattr(self.fly.body, "inner", None)
        if inner is None:
            return []
        by_name: dict[str, Any] = {}
        bodies = getattr(inner, "body_transforms", None)
        if bodies is not None:
            for t in bodies():
                by_name[t.segment] = t
        geoms = getattr(inner, "geom_transforms", None)
        if geoms is not None and len(by_name) < 40:
            for t in geoms():
                if t.segment not in by_name:
                    by_name[t.segment] = t
        if len(by_name) < 12:
            return []
        out = []
        for t in by_name.values():
            p = t.pos_mm
            q = t.quat_wxyz
            out.append(
                {
                    "s": t.segment,
                    "p": [round(float(p[0]), 4), round(float(p[1]), 4), round(float(p[2]), 4)],
                    "q": [round(float(v), 5) for v in q],
                }
            )
        self._bodies_cache = out
        self._bodies_t = now
        return out

    def _game(self) -> dict[str, Any]:
        live_state = self.live.snapshot() if self.live is not None else None
        if live_state and live_state.get("ok"):
            phase = live_state.get("phase") or ("playing" if self.invite_active else "idle")
            prefix = live_state.get("prefix") or ""
            buffer = live_state.get("buffer") or ""
            return {
                "ok": True,
                "source": "shiritori.lol",
                "phase": phase,
                "prefix": prefix,
                "lastWord": live_state.get("lastWord") or "",
                "buffer": buffer,
                "score": live_state.get("score") or 0,
                "lives": live_state.get("lives"),
                "botLives": live_state.get("botLives"),
                "tries": live_state.get("tries"),
                "yourTurn": bool(live_state.get("yourTurn")),
                "signedIn": bool(live_state.get("signedIn")),
                "url": live_state.get("url") or "https://shiritori.lol/",
                "message": live_state.get("message")
                or (
                    f"shiritori.lol · need “{prefix}”. Buffer {buffer or '_'}"
                    if phase == "playing"
                    else "Mode select is on the monitor. Walk, tap once, walk off. Taps on the menu click the site."
                ),
            }
        if self.invite_active:
            return {
                "ok": True,
                "source": "keys",
                "phase": "menu",
                "prefix": "",
                "lastWord": "",
                "buffer": self.key_buffer,
                "score": 0,
                "lives": None,
                "botLives": None,
                "message": "Tap once per letter, then walk off. On the menu, a tap clicks the site.",
            }
        return {
            "ok": True,
            "source": "shiritori.lol" if self.live is not None else "idle",
            "phase": "idle",
            "prefix": "",
            "lastWord": "",
            "buffer": "",
            "score": 0,
            "lives": None,
            "botLives": None,
            "message": "Ask the fly to play. It taps keys; it does not camp on one letter.",
        }

    def _snapshot(self, rec, pressed: str = "") -> dict[str, Any]:
        pose = self.fly.body.pose if self.fly.body is not None else Pose()
        sources = []
        mode = "rest"
        drive = 0.0
        spikes = 0
        descending = [0.0, 0.0]
        eyes = [0.0, 0.0]
        if rec is not None:
            sources = [s.value for s in rec.sources]
            mode = rec.mode
            drive = rec.walking_drive
            spikes = rec.total_spikes
            descending = list(rec.descending)
            eyes = [rec.left_eye, rec.right_eye]
        if pressed:
            sources.append("physical_key_contact")
        return {
            "ok": True,
            "fly_id": self.fly.identity.fly_id,
            "dataset": self.fly.identity.connectome_dataset,
            "neurons": self.fly.connectome.n,
            "body": self.fly.identity.body_kind,
            "world": self.world.name,
            "x_mm": pose.x_mm,
            "y_mm": pose.y_mm,
            "z_mm": pose.z_mm,
            "heading_rad": pose.heading_rad,
            "speed_mm_s": round(self._speed_mm_s, 2),
            "mode": mode,
            "walking_drive": drive,
            "descending": descending,
            "left_eye": eyes[0],
            "right_eye": eyes[1],
            "spikes": spikes,
            "sources": sources,
            "paused": self.paused,
            "physiology": self.fly.physiology.state.snapshot(),
            "neuromodulation": self.fly.physiology.neuromodulation.state.snapshot(),
            "walk_hz": float(getattr(rec, "walk_hz", 0.0) or 0.0) if rec is not None else 0.0,
            "walk_trace": float(getattr(rec, "walk_trace", 0.0) or 0.0) if rec is not None else 0.0,
            "scaffold_used": bool(getattr(rec, "scaffold_used", False)) if rec is not None else False,
            "legacy_scaffold": self.fly.legacy_scaffold,
            "policy": self.fly.policy.as_dict(),
            "motor_mode": self.fly.motor_mode.value,
            "motor_fidelity_level": self.fly.motor_fidelity_level,
            "walk_initiation_trace": dict(self.fly.bridge.last_trace or {}),
            "walk_trace_text": (self.fly.bridge.last_trace or {}).get("text", ""),
            "consciousness_claimed": False,
            "gait_phase": float(getattr(self.fly.body, "gait_phase", 0.0)),
            "activity": self._activity_cache or self._activity(rec),
            "frame_ready": False,
            "coords_note": self.brain.coords_source,
            "render_error": self._render_error,
            "bodies": self._bodies(),
            "invite": self.invite_active,
            "pressed": pressed,
            "game": self._game(),
            "monitor_on": bool(self.world.monitor_on),
            "wing_phase": time.perf_counter() * 80.0 if mode == "fly" else 0.0,
        }

    def _activity(self, rec) -> dict:
        duration_s = self.fly.loop.brain_ticks * self.fly.net.params.dt / 1000.0 if self.fly.loop else 0.01
        counts = self.fly.net.counts if rec is not None else np.zeros(self.fly.connectome.n, dtype=np.int32)
        return self.brain.snapshot(self.fly, counts, duration_s)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.latest)

    def jpeg(self) -> bytes:
        return b""

    def room_layout(self) -> dict[str, Any]:
        w = self.world
        x, y, z = spawn_on_rug()
        return {
            "width_mm": w.width_mm,
            "depth_mm": w.depth_mm,
            "height_mm": w.height_mm,
            "spawn_mm": [x, y, z],
            "solids": [
                {
                    "name": s.name,
                    "x0": s.x0,
                    "y0": s.y0,
                    "x1": s.x1,
                    "y1": s.y1,
                    "z0": s.z0,
                    "z1": s.z1,
                }
                for s in w.solids
            ],
            "lights": [
                {"x_mm": lamp.x_mm, "y_mm": lamp.y_mm, "z_mm": lamp.z_mm, "intensity": lamp.intensity}
                for lamp in w.lights
            ],
            "keys": [
                {
                    "label": k.label,
                    "x_mm": k.x_mm,
                    "y_mm": k.y_mm,
                    "z_mm": k.z_mm,
                    "half_mm": k.half_mm,
                }
                for k in w.keys
            ],
            "monitor": w.monitor,
        }

    def brain_layout(self) -> dict[str, Any]:
        payload = self.brain.layout()
        payload["dataset"] = self.fly.identity.connectome_dataset
        payload["edges_full"] = self.fly.connectome.n_edges
        return payload

    def set_paused(self, paused: bool) -> dict[str, Any]:
        self.paused = bool(paused)
        self.fly.developer.pause(self.paused)
        return self.snapshot()
