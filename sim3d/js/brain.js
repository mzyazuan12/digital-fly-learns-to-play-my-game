/** MaleCNS + live shiritori.lol browser APIs. */

export async function brainStatus() {
  try {
    const res = await fetch("/api/status");
    if (!res.ok) return { ok: false, brain: false };
    return res.json();
  } catch {
    return { ok: false, brain: false };
  }
}

export async function startCasual() {
  const res = await fetch("/api/game/start", { method: "POST" });
  return res.json();
}

export async function gameAct(action) {
  const res = await fetch("/api/game/act", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action }),
  });
  return res.json();
}

export async function connectomeThink(observation) {
  const res = await fetch("/api/brain", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(observation),
  });
  if (!res.ok) return null;
  return res.json();
}

export async function liveState() {
  const res = await fetch("/api/browser/state");
  if (!res.ok) return { ok: false, phase: "error", message: "live browser is down" };
  return res.json();
}

export async function liveGoto(url) {
  return post("/api/browser/goto", { url });
}

export async function liveClick(x, y) {
  return post("/api/browser/click", { x, y });
}

export async function liveKey(key) {
  return post("/api/browser/key", { key });
}

export async function liveLogin() {
  return post("/api/browser/login", {});
}

export async function liveMode(mode) {
  return post("/api/browser/mode", { mode });
}

export async function liveSuggest() {
  return post("/api/browser/suggest", {});
}

export async function liveGiveUp() {
  return post("/api/browser/giveup", { model_failed: true });
}

async function post(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  try {
    return await res.json();
  } catch {
    return { ok: false, error: `${path} failed` };
  }
}

export function normalizeKey(raw) {
  if (!raw) return null;
  const k = String(raw).toUpperCase();
  if (k === "ENTER" || k === "RETURN") return "ENTER";
  if (k === "BACKSPACE" || k === "⌫") return "BACKSPACE";
  if (k === "ESCAPE" || k === "ESC") return "ESCAPE";
  if (k === "NOOP" || k === "NO-OP") return "NOOP";
  if (k === " ") return "SPACE";
  if (/^[A-Z]$/.test(k)) return k;
  return null;
}
