import { createBrainView } from "./brainview.js";
import { startLivingRoom } from "./living_view.js?v=nmf30";

const hud = {
  status: document.getElementById("status"),
  backend: document.getElementById("backend-label"),
};

const room = await startLivingRoom({ follow: true });
if (hud.backend) hud.backend.textContent = "Living room · NeuroMechFly on the desk keyboard";
if (hud.status) hud.status.textContent = "Following the fly. Free camera to look around.";
const followBtnInit = document.getElementById("btn-follow");
if (followBtnInit) followBtnInit.textContent = "Free camera";

const followBtn = document.getElementById("btn-follow");
if (followBtn) {
  followBtn.onclick = () => {
    const next = !room.mode.follow;
    room.setFollow(next);
    followBtn.textContent = next ? "Free camera" : "Follow";
  };
}
const pauseBtn = document.getElementById("btn-pause");
if (pauseBtn) {
  pauseBtn.onclick = async () => {
    const label = pauseBtn.textContent === "Pause";
    pauseBtn.textContent = label ? "Resume" : "Pause";
    await fetch("/api/pause", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ paused: label }),
    });
  };
}

let brainView = null;
const stage = document.getElementById("brain-stage");
if (stage) {
  try {
    brainView = await createBrainView(stage);
  } catch {
    const note = document.getElementById("brain-count");
    if (note) note.textContent = "Brain view waiting for MaleCNS…";
  }
}

async function pump() {
  try {
    const s = await fetch("/api/live", { cache: "no-store" }).then((r) => r.json());
    if (!s?.ok) return;
    if (s.bodies?.length) room.fly.applyBodies(s.bodies, { mode: s.mode, wingPhase: s.wing_phase });
    else room.fly.applyRoot(s.x_mm, s.y_mm, (room.plantY || 0.811) * 1000, s.heading_rad || 0);
    room.fly.setFlying(s.mode === "fly");
    room.fly.setGrooming(s.mode === "groom");
    room.fly.setWalking(s.mode === "walk" || s.mode === "reverse");
    if (brainView && s.activity) brainView.apply(s.activity);
    if (s.paused && hud.status) hud.status.textContent = "Paused.";
  } catch {
    if (hud.status) hud.status.textContent = "Waiting for the habitat server…";
  }
}
setInterval(pump, 90);
pump();
