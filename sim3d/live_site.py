"""Playwright session that paints shiritori.lol onto the 3D monitor.

All Playwright calls run on one worker thread. The HTTP handlers only enqueue
commands. Credentials come from SHIRITORI_EMAIL / SHIRITORI_PASSWORD.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from sim3d.live_policy import MODE_BUTTON_IDS, normalize_mode

ROOT = Path(__file__).resolve().parents[1]
HOME = "https://shiritori.lol/"
VIEW_W = 1280
VIEW_H = 720
SCRAPE_JS = """() => {
  const el = (id) => document.getElementById(id);
  const text = (id) => (el(id)?.innerText || "").replace(/\\s+/g, " ").trim();
  const isHidden = (id) => {
    const n = el(id);
    if (!n) return true;
    if (n.classList.contains("hidden")) return true;
    const s = getComputedStyle(n);
    return s.display === "none" || s.visibility === "hidden";
  };
  const lives = (id) => {
    const n = el(id);
    if (!n) return 0;
    const t = n.innerText || "";
    const hearts = (t.match(/[♥❤♡♥︎]/g) || []).length;
    if (hearts) return hearts;
    const filled = n.querySelectorAll("[data-filled='true'], .filled, .alive, .on").length;
    if (filled) return filled;
    const kids = [...n.children].filter((c) => getComputedStyle(c).display !== "none");
    if (kids.length) return kids.length;
    const num = parseInt(t.replace(/\\D/g, ""), 10);
    return Number.isFinite(num) ? num : 0;
  };
  const triesTxt = text("tries");
  const triesMatch = triesTxt.match(/(\\d+)/);
  const dash = (s) => /^[-–—]+$/.test(s) ? "" : s;
  const prefix = dash(text("prefix-need") || text("word-dock-prefix"));
  const lastWord = dash(text("wpm-word"));
  const finished = !isHidden("result-overlay");
  const menu = !isHidden("mode-select") && isHidden("hud");
  const playing = !finished && !menu && !isHidden("hud");
  const signedIn = !isHidden("auth-account-block") || Boolean(text("auth-display-name"));
  const input = el("word-input");
  const inputLocked = Boolean(input && (input.disabled || input.readOnly));
  const botBusy = !isHidden("bot-thinking") || !isHidden("opponent-wait") || Boolean(document.querySelector(".bot-turn, .opponent-turn, #bot-turn"));
  const yourTurn = playing && !inputLocked && !botBusy;
  return {
    url: location.href,
    title: document.title,
    prefix: prefix.toLowerCase(),
    lastWord: lastWord.toLowerCase(),
    buffer: (el("word-input")?.value || "").toLowerCase(),
    score: parseInt(text("score-num").replace(/\\D/g, ""), 10) || 0,
    tries: triesMatch ? parseInt(triesMatch[1], 10) : 5,
    lives: lives("you-lives"),
    botLives: lives("bot-lives"),
    chainLen: parseInt(text("chain-len"), 10) || 1,
    round: parseInt(text("round-num"), 10) || 1,
    phase: finished ? "finished" : playing ? "playing" : "menu",
    yourTurn,
    opponentTurn: playing && !yourTurn,
    signedIn,
    displayName: text("auth-display-name"),
    resultTitle: text("result-title"),
    resultMsg: text("result-msg"),
    authStatus: text("auth-status"),
  };
}"""


def load_dotenv(path: Path | None = None) -> None:
    env_path = path or ROOT / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _safe_url(raw: str) -> str:
    url = str(raw or "").strip() or HOME
    if url.startswith("/") and not url.startswith("//"):
        url = HOME.rstrip("/") + url
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"blocked url {raw!r}")
    return url


class LiveSite:
    def __init__(self) -> None:
        load_dotenv()
        self.commands: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.frame_jpeg = b""
        self.last_state: dict[str, Any] = {
            "ok": False,
            "phase": "boot",
            "url": HOME,
            "message": "Starting the live browser…",
        }
        self.used: set[str] = set()
        self.mode = "casual"
        self.running = False
        self.ready = threading.Event()
        self.worker: threading.Thread | None = None

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.worker = threading.Thread(target=self._run, name="live-shiritori", daemon=True)
        self.worker.start()

    def stop(self) -> None:
        self.running = False
        self.commands.put({"op": "stop"})

    def call(self, op: str, timeout: float = 45.0, **payload: Any) -> dict[str, Any]:
        if not self.running:
            return {"ok": False, "error": "live browser is not running"}
        box: queue.Queue = queue.Queue(maxsize=1)
        self.commands.put({"op": op, "reply": box, **payload})
        try:
            return box.get(timeout=timeout)
        except queue.Empty:
            return {"ok": False, "error": f"live browser timed out on {op}"}

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            state = dict(self.last_state)
        state["used"] = len(self.used)
        state["mode"] = self.mode
        return state

    def frame(self) -> bytes:
        with self.lock:
            return self.frame_jpeg

    def _run(self) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            self._set_state(
                {
                    "ok": False,
                    "phase": "error",
                    "message": "Playwright is not installed. pip install playwright && python -m playwright install chromium",
                }
            )
            self.ready.set()
            return
        try:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(
                    headless=True,
                    args=[
                        "--disable-dev-shm-usage",
                        "--ignore-gpu-blocklist",
                        "--disable-blink-features=AutomationControlled",
                    ],
                )
                page = browser.new_page(viewport={"width": VIEW_W, "height": VIEW_H})
                page.set_default_timeout(20000)
                self._boot(page)
                self.ready.set()
                last_shot = 0.0
                while self.running:
                    try:
                        cmd = self.commands.get(timeout=0.08)
                    except queue.Empty:
                        cmd = None
                    if cmd:
                        if cmd.get("op") == "stop":
                            break
                        self._handle(page, cmd)
                    now = time.time()
                    if now - last_shot >= 0.12:
                        self._grab(page)
                        last_shot = now
                browser.close()
        except Exception as exc:
            self._set_state({"ok": False, "phase": "error", "message": f"Live browser failed: {exc}"})
            self.ready.set()

    def _boot(self, page) -> None:
        page.goto(HOME, wait_until="domcontentloaded")
        page.wait_for_selector("#auth-email-input, #btn-casual, #word-input", timeout=20000)
        self._note_used(self._scrape(page))
        email = os.environ.get("SHIRITORI_EMAIL", "").strip()
        password = os.environ.get("SHIRITORI_PASSWORD", "")
        if email and password:
            login = self._login(page, email, password)
            message = login.get("message") or "Live shiritori.lol is on the monitor."
        else:
            message = "Live shiritori.lol is on the monitor. Set SHIRITORI_EMAIL / SHIRITORI_PASSWORD to sign in."
            login = {"ok": False, "signedIn": False}
        state = self._scrape(page)
        state.update(
            {
                "ok": True,
                "message": message,
                "signedIn": bool(login.get("signedIn") or state.get("signedIn")),
            }
        )
        self._set_state(state)

    def _handle(self, page, cmd: dict[str, Any]) -> None:
        reply = cmd.get("reply")
        op = cmd.get("op")
        try:
            if op == "goto":
                payload = self._goto(page, cmd.get("url") or HOME)
            elif op == "click":
                payload = self._click(page, float(cmd.get("x", 0)), float(cmd.get("y", 0)))
            elif op == "key":
                payload = self._key(page, str(cmd.get("key") or ""))
            elif op == "login":
                payload = self._login(
                    page,
                    str(cmd.get("email") or os.environ.get("SHIRITORI_EMAIL", "")),
                    str(cmd.get("password") or os.environ.get("SHIRITORI_PASSWORD", "")),
                )
            elif op == "mode":
                payload = self._start_mode(page, cmd.get("mode") or "casual")
            elif op == "menu":
                payload = self._show_menu(page)
            elif op == "suggest":
                payload = self._suggest(page)
            elif op == "giveup":
                payload = self._give_up(page, bool(cmd.get("model_failed")))
            elif op == "state":
                payload = self._refresh(page)
            else:
                payload = {"ok": False, "error": f"unknown op {op}"}
        except Exception as exc:
            payload = {"ok": False, "error": str(exc)}
        if reply is not None:
            reply.put(payload)

    def _goto(self, page, url: str) -> dict[str, Any]:
        page.goto(_safe_url(url), wait_until="domcontentloaded")
        return self._refresh(page, message=f"Opened {page.url}")

    def _click(self, page, x: float, y: float) -> dict[str, Any]:
        px = max(0.0, min(1.0, x)) * VIEW_W
        py = max(0.0, min(1.0, y)) * VIEW_H
        page.mouse.click(px, py)
        time.sleep(0.05)
        return self._refresh(page)

    def _key(self, page, raw: str) -> dict[str, Any]:
        key = str(raw or "").upper()
        mapping = {
            "ENTER": "Enter",
            "RETURN": "Enter",
            "BACKSPACE": "Backspace",
            "ESC": "Escape",
            "ESCAPE": "Escape",
            "TAB": "Tab",
            "SPACE": " ",
        }
        if key in mapping:
            if key in {"ENTER", "RETURN"}:
                page.evaluate(
                    """() => {
                      const input = document.getElementById("word-input");
                      if (input) {
                        input.focus();
                        for (const type of ["keydown", "keypress", "keyup"]) {
                          input.dispatchEvent(new KeyboardEvent(type, {
                            key: "Enter", code: "Enter", keyCode: 13, which: 13, bubbles: true,
                          }));
                        }
                      }
                      document.querySelector(".game-osk-go")?.click();
                    }"""
                )
            page.keyboard.press(mapping[key])
        elif len(key) == 1:
            page.keyboard.type(key.lower(), delay=15)
        else:
            return {"ok": False, "error": f"unsupported key {raw!r}"}
        return self._refresh(page)

    def _login(self, page, email: str, password: str) -> dict[str, Any]:
        email = (email or "").strip()
        if not email or not password:
            return {"ok": False, "error": "missing SHIRITORI_EMAIL / SHIRITORI_PASSWORD"}
        if "shiritori.lol" not in (page.url or ""):
            page.goto(HOME, wait_until="domcontentloaded")
        state = self._scrape(page)
        if state.get("signedIn"):
            state.update({"ok": True, "message": f"Already signed in as {state.get('displayName') or email}."})
            self._set_state(state)
            return state
        tab = page.locator("#auth-tab-signin")
        if tab.count():
            tab.first.click()
        page.fill("#auth-email-input", email)
        page.fill("#auth-password-input", password)
        page.click("#auth-submit")
        page.wait_for_timeout(800)
        deadline = time.time() + 15
        last = self._scrape(page)
        while time.time() < deadline:
            last = self._scrape(page)
            if last.get("signedIn"):
                last.update({"ok": True, "message": f"Signed in as {last.get('displayName') or email}."})
                self._set_state(last)
                return last
            status = (last.get("authStatus") or "").lower()
            if status and any(w in status for w in ("invalid", "incorrect", "fail", "error", "wrong")):
                last.update({"ok": False, "error": last.get("authStatus")})
                self._set_state(last)
                return last
            page.wait_for_timeout(250)
        last.update({"ok": False, "error": last.get("authStatus") or "sign-in did not complete"})
        self._set_state(last)
        return last

    def _start_mode(self, page, raw_mode: str) -> dict[str, Any]:
        mode = normalize_mode(raw_mode)
        self.mode = mode
        self.used = set()
        if "shiritori.lol" not in (page.url or ""):
            page.goto(HOME, wait_until="domcontentloaded")
        email = os.environ.get("SHIRITORI_EMAIL", "").strip()
        password = os.environ.get("SHIRITORI_PASSWORD", "")
        if email and password:
            self._login(page, email, password)
        button_id = MODE_BUTTON_IDS[mode]
        page.keyboard.press("Escape")
        page.wait_for_timeout(120)
        page.evaluate(
            """({buttonId}) => {
              document.getElementById("result-overlay")?.classList.add("hidden");
              document.getElementById("hud")?.classList.add("hidden");
              document.getElementById("mode-select")?.classList.remove("hidden");
              document.getElementById(buttonId)?.click();
            }""",
            {"buttonId": button_id},
        )
        page.wait_for_timeout(200)
        page.evaluate(
            """() => {
              const enter = document.getElementById("btn-enter");
              const guest = document.getElementById("btn-enter-guest");
              if (enter) enter.click();
              else if (guest) guest.click();
            }"""
        )
        deadline = time.time() + 12
        state = self._scrape(page)
        while time.time() < deadline and state.get("phase") == "menu":
            page.wait_for_timeout(200)
            state = self._scrape(page)
        self.used = set()
        self._note_used(state)
        if state.get("phase") == "playing":
            try:
                page.locator("#word-input").click(timeout=800)
            except Exception:
                page.mouse.click(VIEW_W * 0.55, VIEW_H * 0.45)
            state.update({"ok": True, "message": f"Live {mode} match. Need {state.get('prefix') or 'none'}."})
        else:
            state.update(
                {
                    "ok": state.get("phase") != "error",
                    "message": state.get("resultMsg") or f"Opened {mode} on shiritori.lol.",
                }
            )
        self._set_state(state)
        return state

    def _show_menu(self, page) -> dict[str, Any]:
        if "shiritori.lol" not in (page.url or ""):
            page.goto(HOME, wait_until="domcontentloaded")
        page.keyboard.press("Escape")
        page.wait_for_timeout(120)
        page.evaluate(
            """() => {
              document.getElementById("result-overlay")?.classList.add("hidden");
              document.getElementById("hud")?.classList.add("hidden");
              document.getElementById("mode-select")?.classList.remove("hidden");
            }"""
        )
        state = self._scrape(page)
        state.update({"ok": True, "message": "Mode select is on the monitor. Casual / Pro / Featherine / Lambdadelta."})
        self._set_state(state)
        return state

    def _suggest(self, page) -> dict[str, Any]:
        state = self._refresh(page)
        state.update(
            {
                "ok": False,
                "give_up": False,
                "word": None,
                "type": None,
                "error": (
                    "Lexicon autoplay is disabled. The fly does not receive the "
                    "dictionary, the prefix, or whose turn it is."
                ),
            }
        )
        return state

    def _give_up(self, page, model_failed: bool) -> dict[str, Any]:
        if not model_failed:
            return {
                "ok": False,
                "error": "give-up blocked until the model has no remaining word",
                **self._refresh(page),
            }
        # Do not click Skip or multiplayer forfeit. Stop typing and let the
        # live timer / lives resolve the failed turn.
        state = self._refresh(page, message="Model failed. Not skipping; the live match will time out.")
        state["give_up"] = True
        state["reason"] = "model_failed"
        return state

    def _refresh(self, page, message: str | None = None) -> dict[str, Any]:
        state = self._scrape(page)
        self._note_used(state)
        state["ok"] = True
        if message:
            state["message"] = message
        self._set_state(state)
        return state

    def _scrape(self, page) -> dict[str, Any]:
        try:
            data = page.evaluate(SCRAPE_JS)
        except Exception as exc:
            data = {"phase": "error", "message": str(exc), "url": page.url}
        data["ok"] = True
        data["mode"] = self.mode
        return data

    def _note_used(self, state: dict[str, Any]) -> None:
        last = str(state.get("lastWord") or "").lower()
        if last:
            self.used.add(last)

    def _grab(self, page) -> None:
        try:
            jpeg = page.screenshot(type="jpeg", quality=52)
        except Exception:
            return
        state = self._scrape(page)
        with self.lock:
            self.frame_jpeg = jpeg
            self.last_state = {**self.last_state, **state, "ok": True}

    def _set_state(self, state: dict[str, Any]) -> None:
        with self.lock:
            self.last_state = {**self.last_state, **state}
