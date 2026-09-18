"""Live MuJoCo NeuroMechFly gait. MaleCNS is not loaded.

Rendering is the FlyGym/MuJoCo renderer. The browser displays those frames.
It does not move a decorative mesh.

MuJoCo's GL renderer is created and used on the same worker thread.
"""

from __future__ import annotations

import io
import threading
import time
import traceback
from typing import Any

import numpy as np

from organism.gait import ArticulatedFly, GAIT_COMMANDS


class GaitRuntime:
    def __init__(self):
        self.lock = threading.Lock()
        self.fly: ArticulatedFly | None = None
        self.command = "stand"
        self.paused = False
        self._running = True
        self._jpeg = b""
        self._error: str | None = None
        self._joint_trace: list[list[float]] = []
        self.latest: dict[str, Any] = {
            "ok": False,
            "brain": None,
            "error": "Building NeuroMechFly…",
        }
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="gait", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        try:
            fly = ArticulatedFly(render=False)
            fly.enable_renderer(
                camera_res=(1080, 1920),
                buffer_frames=False,
                playback_speed=1.0,
            )
            fly.set_command(self.command)
            self.fly = fly
            self._ready.set()
            steps_per_frame = 40
            while self._running:
                if self.paused:
                    time.sleep(0.02)
                    continue
                for _ in range(steps_per_frame):
                    fly.step(self.command)
                frame = fly.render_frame()
                jpeg = _jpeg(frame)
                snap = self._snapshot(fly)
                with self.lock:
                    self._jpeg = jpeg
                    self.latest = snap
        except Exception:
            self._error = traceback.format_exc()
            with self.lock:
                self.latest = {
                    "ok": False,
                    "brain": None,
                    "error": self._error.splitlines()[-1],
                    "traceback": self._error,
                }

    def _snapshot(self, fly: ArticulatedFly) -> dict[str, Any]:
        snap = fly.snapshot()
        prop = fly.proprioception()
        std = float(np.std(snap.joint_angles)) if snap.joint_angles.size else 0.0
        self._joint_trace.append(snap.joint_angles.tolist())
        self._joint_trace = self._joint_trace[-240:]
        return {
            "ok": True,
            "brain": None,
            "body": fly.kind,
            "controller": fly.controller_kind,
            "controller_role": fly.controller_role,
            "command": self.command,
            "mode": snap.mode,
            "left": snap.left,
            "right": snap.right,
            "t_s": snap.t_s,
            "thorax_mm": snap.thorax_xyz_mm.tolist(),
            "heading_rad": snap.heading_rad,
            "joint_std": std,
            "joint_names": snap.joint_names,
            "joint_angles": snap.joint_angles.tolist(),
            "joint_trace": self._joint_trace[-80:],
            "contact": snap.contact_found.tolist(),
            "contact_force": snap.contact_force_mag.tolist(),
            "actuator_force": prop.actuator_forces.tolist(),
            "angular_velocity": prop.angular_velocity.tolist(),
            "thorax_quat": prop.thorax_quat_wxyz.tolist(),
            "tarsus5_z_mm": prop.tarsus5_z_mm.tolist(),
            "root_written": snap.root_written,
            "paused": self.paused,
            "commands": list(GAIT_COMMANDS),
            "render_map": {
                "bodies": [
                    {
                        "segment": t.segment,
                        "mujoco_body": t.mujoco_name,
                        "render_node": t.segment,
                    }
                    for t in fly.body_transforms()
                ]
            },
        }

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.latest)

    def jpeg(self) -> bytes:
        with self.lock:
            return self._jpeg

    def set_command(self, command: str) -> dict[str, Any]:
        if command not in GAIT_COMMANDS:
            raise KeyError(command)
        self.command = command
        fly = self.fly
        if fly is not None:
            fly.set_command(command)
        with self.lock:
            latest = dict(self.latest)
            latest["command"] = command
            spec = GAIT_COMMANDS[command]
            latest["left"] = float(spec["left"])
            latest["right"] = float(spec["right"])
            latest["mode"] = str(spec["mode"])
            latest["ok"] = True
            latest.pop("error", None)
            self.latest = latest
            return dict(latest)

    def set_paused(self, paused: bool) -> dict[str, Any]:
        self.paused = bool(paused)
        return self.snapshot()


def _jpeg(frame: np.ndarray) -> bytes:
    from PIL import Image

    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)[..., :3])
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return buf.getvalue()
