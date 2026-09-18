"""Serve the living-room fly.

  python -m sim3d.serve              # MaleCNS → NeuroMechFly; shiritori.lol on the desk
  python -m sim3d.serve --no-live    # body + brain only (no Playwright)
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent


class Runtime:
    def __init__(self, live: bool, connectome: str = "malecns", physics: bool = True):
        self.lock = threading.Lock()
        self.live = None
        self.connectome = connectome
        self.physics = physics
        self.organism = None
        self.organism_loading = True
        self._organism_error = None
        self.gait = None
        self._gait_error = None
        self._gait_lock = threading.Lock()
        if live:
            from sim3d.live_site import LiveSite

            self.live = LiveSite()
            self.live.start()
        threading.Thread(target=self._load_organism, name="organism", daemon=True).start()

    def _load_organism(self) -> None:
        try:
            from sim3d.organism_runtime import OrganismRuntime

            print(f"[sim3d] hatching organism ({self.connectome}, physics={self.physics})…")
            self.organism = OrganismRuntime(
                connectome=self.connectome, physics=self.physics, live=self.live
            )
            print(f"[sim3d] organism ready: {self.organism.fly.connectome.n} cells")
        except Exception as exc:
            self._organism_error = str(exc)
            print(f"[sim3d] organism failed: {exc}")
        finally:
            self.organism_loading = False

    def organism_runtime(self):
        return self.organism

    def organism_payload(self) -> dict:
        if self.organism is not None:
            return self.organism.snapshot()
        return {
            "ok": False,
            "loading": True,
            "dataset": self.connectome,
            "error": self._organism_error or "connectome still loading",
        }

    def brain_layout_payload(self) -> dict:
        if self.organism is not None:
            return self.organism.brain_layout()
        return {
            "ok": False,
            "loading": True,
            "n_display": 0,
            "n_full": 0,
            "error": self._organism_error or "connectome still loading",
        }

    def gait_runtime(self, *, start: bool = False):
        with self._gait_lock:
            if start and self.gait is None and self._gait_error is None:
                try:
                    from sim3d.gait_runtime import GaitRuntime

                    print("[sim3d] building NeuroMechFly gait lab (MaleCNS disconnected)…")
                    self.gait = GaitRuntime()
                except Exception as exc:
                    self._gait_error = str(exc)
                    print(f"[sim3d] NeuroMechFly gait failed: {exc}")
            return self.gait

    def status(self) -> dict:
        live = None
        if self.live is not None:
            live = self.live.snapshot()
        org = self.organism
        if org is not None:
            return {
                "ok": True,
                "brain": True,
                "brainLoading": False,
                "name": org.fly.identity.connectome_dataset,
                "neurons": int(org.fly.connectome.n),
                "edges": int(org.fly.connectome.n_edges),
                "coords": org.brain.coords_source,
                "live": bool(self.live),
                "liveState": live,
                "disclaimer": "LIF on MaleCNS topology. Tarsus keys type on shiritori.lol.",
            }
        return {
            "ok": True,
            "brain": False,
            "brainLoading": self.organism_loading,
            "live": bool(self.live),
            "liveState": live,
            "error": self._organism_error or "MaleCNS still loading",
        }

    def start_game(self) -> dict:
        if self.live is None:
            return {"ok": False, "error": "live shiritori.lol is not attached"}
        return self.live.call("mode", timeout=60.0, mode="casual")

    def game_act(self, action: str) -> dict:
        if self.live is None:
            return {"ok": False, "error": "live shiritori.lol is not attached"}
        if action in {"ESC", "ESCAPE"}:
            return self.live.call("key", key="ESC")
        return self.live.call("key", key=action)

    def think(self, observation: dict) -> dict:
        return {
            "ok": False,
            "error": (
                "No keyboard BCI. Prefix and whose-turn are not fly senses. "
                "The fly types by standing on keys on shiritori.lol."
            ),
        }


def make_handler(runtime: Runtime):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(ROOT), **kwargs)

        def log_message(self, fmt, *args):
            print("[sim3d]", args[0] if args else fmt)

        def end_headers(self):
            if any(self.path.split("?", 1)[0].endswith(ext) for ext in (".js", ".html", ".css", ".glb")):
                self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def do_GET(self):
            path = urlparse(self.path).path
            if path in {"/", "/live", "/organism", "/habitat"}:
                self.path = "/organism.html"
                return super().do_GET()
            if path in {"/gait"}:
                runtime.gait_runtime(start=True)
                self.path = "/gait.html"
                return super().do_GET()
            if path == "/desk":
                self.path = "/index.html"
                return super().do_GET()
            if path == "/api/gait":
                gait = runtime.gait_runtime()
                if gait is None:
                    self._json(200, {"ok": False, "error": "gait lab is off. Open /gait only if you need the brain-off bench."})
                    return
                self._json(200, gait.snapshot())
                return
            if path == "/api/gait/frame":
                gait = runtime.gait_runtime()
                if gait is None:
                    self.send_error(503, "gait lab is off")
                    return
                jpeg = gait.jpeg()
                if not jpeg:
                    self.send_error(503, "no MuJoCo frame yet")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(jpeg)
                return
            if path == "/api/live":
                self._json(200, runtime.organism_payload())
                return
            if path == "/api/live/frame":
                org = runtime.organism_runtime()
                if org is None:
                    self.send_error(503, runtime._organism_error or "organism not ready")
                    return
                jpeg = org.jpeg()
                if not jpeg:
                    self.send_error(503, "no MuJoCo frame yet")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    self.wfile.write(jpeg)
                except BrokenPipeError:
                    return
                return
            if path == "/api/room":
                org = runtime.organism_runtime()
                if org is None:
                    self._json(503, {"ok": False, "loading": True, "error": "connectome still loading"})
                    return
                self._json(200, org.room_layout())
                return
            if path == "/api/brain-layout":
                self._json(200, runtime.brain_layout_payload())
                return
            if path == "/api/status":
                self._json(200, runtime.status())
                return
            if path == "/api/game":
                org = runtime.organism_runtime()
                if org is not None:
                    self._json(200, org.snapshot().get("game") or {"ok": True, "phase": "idle"})
                    return
                if runtime.live is not None:
                    self._json(200, runtime.live.snapshot())
                    return
                self._json(200, {"ok": True, "phase": "idle", "message": "live shiritori.lol is not attached"})
                return
            if path == "/api/browser/state":
                if runtime.live is None:
                    self._json(503, {"ok": False, "error": "live browser disabled"})
                    return
                self._json(200, runtime.live.snapshot())
                return
            if path == "/api/browser/frame":
                if runtime.live is None:
                    self.send_error(503, "live browser disabled")
                    return
                jpeg = runtime.live.frame()
                if not jpeg:
                    self.send_error(503, "no frame yet")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    self.wfile.write(jpeg)
                except BrokenPipeError:
                    return
                return
            super().do_GET()

        def do_POST(self):
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            live_ops = {
                "/api/browser/goto": "goto",
                "/api/browser/click": "click",
                "/api/browser/key": "key",
                "/api/browser/login": "login",
                "/api/browser/mode": "mode",
                "/api/browser/menu": "menu",
                "/api/browser/suggest": "suggest",
                "/api/browser/giveup": "giveup",
            }
            if path in live_ops:
                if runtime.live is None:
                    self._json(503, {"ok": False, "error": "live browser disabled"})
                    return
                payload = runtime.live.call(live_ops[path], **body)
                self._json(200 if payload.get("ok") else 400, payload)
                return
            if path == "/api/invite":
                org = runtime.organism_runtime()
                if org is None:
                    self._json(503, {"ok": False, "error": "connectome still loading"})
                    return
                self._json(200, org.invite())
                return
            with runtime.lock:
                if path == "/api/gait/command":
                    gait = runtime.gait_runtime()
                    if gait is None:
                        self._json(400, {"ok": False, "error": "gait lab is off"})
                        return
                    try:
                        self._json(200, gait.set_command(str(body.get("command") or "stand")))
                    except KeyError as exc:
                        self._json(400, {"ok": False, "error": str(exc)})
                    return
                if path == "/api/gait/pause":
                    gait = runtime.gait_runtime()
                    if gait is None:
                        self._json(503, {"ok": False, "error": runtime._gait_error or "gait not ready"})
                        return
                    self._json(200, gait.set_paused(bool(body.get("paused"))))
                    return
                if path == "/api/pause":
                    org = runtime.organism_runtime()
                    if org is None:
                        self._json(503, {"ok": False, "error": "connectome still loading"})
                        return
                    self._json(200, org.set_paused(bool(body.get("paused"))))
                    return
                if path == "/api/brain":
                    payload = runtime.think(body)
                    self._json(200 if payload.get("ok") else 400, payload)
                    return
                if path == "/api/game/start":
                    self._json(200, runtime.start_game())
                    return
                if path == "/api/game/act":
                    action = str(body.get("action") or body.get("key") or "NOOP")
                    self._json(200, runtime.game_act(action))
                    return
            self.send_error(404)

        def _json(self, code, payload):
            raw = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    return Handler


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--live", action="store_true", help="(default) attach Chromium to shiritori.lol")
    parser.add_argument("--no-live", action="store_true", help="do not open the real shiritori.lol browser")
    parser.add_argument("--brain", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-brain", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--connectome", default="malecns", choices=("synthetic", "malecns"))
    args = parser.parse_args(argv)
    live = not bool(args.no_live)
    runtime = Runtime(
        live=live,
        connectome=args.connectome,
        physics=True,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(runtime))
    print(f"Living room (MaleCNS → NeuroMechFly; tarsus → shiritori.lol) at http://{args.host}:{args.port}/")
    print(f"Gait lab (brain off, optional) at http://{args.host}:{args.port}/gait")
    print(f"Desk at http://{args.host}:{args.port}/desk")
    if runtime.live is None:
        print("Live shiritori.lol browser off (--no-live).")
    else:
        print("Monitor is a live Chromium session of https://shiritori.lol/")
    print(f"Organism connectome: {args.connectome}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
