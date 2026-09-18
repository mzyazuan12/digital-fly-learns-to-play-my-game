import { createBrainView } from "./brainview.js";
import { startLivingRoom } from "./living_view.js?v=nmf30";

const hud = {
  status: document.getElementById("status"),
  backend: document.getElementById("backend-label"),
  dataset: document.getElementById("dataset"),
  neurons: document.getElementById("neurons"),
  body: document.getElementById("body"),
  drive: document.getElementById("drive"),
  walkDrive: document.getElementById("walk-drive"),
  groomDrive: document.getElementById("groom-drive"),
  thorax: document.getElementById("thorax"),
  spikes: document.getElementById("spikes"),
  pathways: document.getElementById("pathways"),
  source: document.getElementById("source"),
};
const game = {
  need: document.getElementById("need"),
  last: document.getElementById("last-word"),
  buffer: document.getElementById("buffer"),
  score: document.getElementById("score"),
  lives: document.getElementById("lives"),
  msg: document.getElementById("game-msg"),
};

let paused = false;
let brainView = null;
let room = null;

try {
  room = await startLivingRoom({ follow: true });
  const followBtn = document.getElementById("btn-follow");
  if (followBtn) followBtn.textContent = "Free camera";
} catch (err) {
  hud.status.textContent = `3D living room failed: ${err?.message || err}`;
}

function fmtHz(node) {
  if (!node) return "0";
  return Number(node.hz || 0).toFixed(1);
}

function fillSuperclass(map) {
  const list = document.getElementById("superclass-list");
  if (!list) return;
  const rows = Object.entries(map || {}).sort((a, b) => b[1] - a[1]).slice(0, 12);
  list.innerHTML = rows.length
    ? rows.map(([name, n]) => `<li><span>${name}</span><span>${n}</span></li>`).join("")
    : "<li class='empty'>No spikes this window.</li>";
}

function syncGame(g) {
  if (!g) return;
  game.need.textContent = g.prefix || "—";
  game.last.textContent = g.lastWord || "—";
  game.buffer.textContent = g.buffer || "—";
  game.score.textContent = String(g.score ?? 0);
  game.lives.textContent = `${g.lives ?? "—"} / bot ${g.botLives ?? "—"}`;
  game.msg.textContent = g.message || "Ask the fly to play.";
}

function syncFly(s) {
  if (!s?.ok) {
    hud.status.textContent = s?.error || "Loading the organism…";
    return;
  }
  const phys = s.physiology || {};
  hud.dataset.textContent = `${s.dataset} · ${s.coords_note || s.activity?.coords_source || ""}`;
  hud.neurons.textContent = Number(s.neurons || 0).toLocaleString();
  hud.body.textContent = s.body || "—";
  const d = s.descending || [0, 0];
  hud.drive.textContent = `${s.mode || "rest"}  L ${Number(d[0]).toFixed(2)}  R ${Number(d[1]).toFixed(2)}`;
  hud.walkDrive.textContent = Number(phys.walking_drive ?? s.walking_drive ?? 0).toFixed(2);
  hud.groomDrive.textContent = Number(phys.grooming_drive ?? 0).toFixed(2);
  hud.thorax.textContent = `${Number(s.x_mm || 0).toFixed(2)}, ${Number(s.y_mm || 0).toFixed(2)}, ${Number(s.z_mm || 0).toFixed(2)} mm · ${Number(s.speed_mm_s || 0).toFixed(1)} mm/s`;
  hud.spikes.textContent = String(s.spikes ?? 0);
  const pw = s.activity?.pathways || {};
  hud.pathways.textContent = `walk ${fmtHz(pw.walk)}  L ${fmtHz(pw.steer_l)}  R ${fmtHz(pw.steer_r)}`;
  hud.source.textContent = (s.sources || []).join(", ") || "—";
  hud.backend.textContent = `${s.dataset} · ${Number(s.neurons || 0).toLocaleString()} cells · ${s.body || "body"}`;
  if (room?.fly) {
    if (s.bodies?.length) room.fly.applyBodies(s.bodies, { mode: s.mode, wingPhase: s.wing_phase });
    else room.fly.applyRoot(s.x_mm, s.y_mm, (room.plantY || 0.811) * 1000, s.heading_rad || 0);
    room.fly.setWalking(s.mode === "walk" || s.mode === "reverse");
    room.fly.setGrooming(s.mode === "groom");
    room.fly.setFlying(s.mode === "fly");
  }
  if (brainView) brainView.apply(s.activity);
  fillSuperclass(s.activity?.by_superclass || s.activity?.by_class);
  syncGame(s.game);
  if (paused) hud.status.textContent = "Paused.";
  else if (s.invite) hud.status.textContent = s.pressed
    ? `The fly tapped ${s.pressed}.`
    : "Invited. Tap a key once, then walk off. On the menu, a tap clicks the site.";
  else hud.status.textContent = room?.mode.follow
    ? "Following NeuroMechFly. Free camera to look around."
    : "In the living room. WASD to look around. Metrics are live.";
}

async function pump() {
  try {
    const s = await fetch("/api/live", { cache: "no-store" }).then((r) => r.json());
    syncFly(s);
  } catch {
    hud.status.textContent = "Waiting for the organism server…";
  }
}

try {
  brainView = await createBrainView(document.getElementById("brain-stage"));
} catch {
  document.getElementById("brain-count").textContent = "Brain view waiting for MaleCNS…";
}

document.getElementById("btn-pause").onclick = async () => {
  paused = !paused;
  document.getElementById("btn-pause").textContent = paused ? "Resume" : "Pause";
  await fetch("/api/pause", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ paused }),
  });
};

document.getElementById("btn-follow").onclick = () => {
  const next = !room?.mode.follow;
  room?.setFollow(next);
  document.getElementById("btn-follow").textContent = next ? "Free camera" : "Follow";
};

function setBrainVisible(on) {
  const panel = document.getElementById("brain-panel");
  panel.hidden = !on;
  panel.style.display = on ? "" : "none";
  const btn = document.getElementById("btn-brain");
  if (btn) btn.textContent = on ? "Hide brain" : "Show brain";
  if (on) brainView?.resize();
}

setBrainVisible(true);
const brainBtn = document.getElementById("btn-brain");
if (brainBtn) brainBtn.onclick = () => setBrainVisible(document.getElementById("brain-panel").hidden);

window.addEventListener("keydown", (ev) => {
  if (ev.target instanceof HTMLInputElement || ev.metaKey || ev.ctrlKey) return;
  if (ev.code === "KeyH") {
    const hudEl = document.getElementById("hud");
    hudEl.hidden = !hudEl.hidden;
  }
  if (ev.code === "KeyB") {
    setBrainVisible(document.getElementById("brain-panel").hidden);
  }
  if (ev.code === "KeyG") {
    const panel = document.getElementById("game-panel");
    panel.hidden = !panel.hidden;
  }
});

async function startMode(mode) {
  const s = await fetch("/api/browser/mode", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode }),
  }).then((r) => r.json()).catch(() => ({ ok: false }));
  document.querySelectorAll(".modes button").forEach((btn) => {
    btn.classList.toggle("ghost", btn.id !== `btn-${mode}`);
  });
  if (s?.message) hud.status.textContent = s.message;
}

for (const mode of ["casual", "pro", "featherine", "lambdadelta"]) {
  const btn = document.getElementById(`btn-${mode}`);
  if (btn) btn.onclick = () => startMode(mode);
}

document.getElementById("btn-invite").onclick = async () => {
  const s = await fetch("/api/invite", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
  }).then((r) => r.json());
  const game = document.getElementById("game-panel");
  if (game) game.hidden = false;
  syncFly(s);
};

document.getElementById("btn-em").onclick = () => {
  const ng = document.getElementById("ng-frame");
  ng.hidden = !ng.hidden;
  if (!ng.hidden && !ng.src) {
    ng.src = "https://neuroglancer-demo.appspot.com/#!gs://flyem-male-cns/v1.0/male-cns-v1.0.json";
  }
  document.getElementById("btn-em").textContent = ng.hidden ? "Embed EM" : "Hide EM";
};

setInterval(pump, 40);
pump();
