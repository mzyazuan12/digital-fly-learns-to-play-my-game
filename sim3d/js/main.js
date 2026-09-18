import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { buildRoom } from "./room.js";
import { buildKeyboard } from "./keyboard.js";
import { buildMonitor } from "./monitor.js";
import { emptySnap } from "./game.js";
import { brainStatus, liveClick, liveGoto, liveKey, liveMode, liveState, normalizeKey } from "./brain.js";

const hud = {
  status: document.getElementById("status"),
  command: document.getElementById("command"),
  need: document.getElementById("need"),
  last: document.getElementById("last-word"),
  buffer: document.getElementById("buffer"),
  score: document.getElementById("score"),
  lives: document.getElementById("lives"),
  tries: document.getElementById("tries"),
  backend: document.getElementById("backend-label"),
  url: document.getElementById("url-input"),
};

const mode = { follow: false };
let lastSnap = emptySnap();
let typing = false;

const viewport = document.getElementById("viewport");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.35;
renderer.outputColorSpace = THREE.SRGBColorSpace;
viewport.appendChild(renderer.domElement);

const pmrem = new THREE.PMREMGenerator(renderer);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x161616);
scene.fog = new THREE.Fog(0x161616, 8, 18);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.02, 40);
camera.position.set(0.08, 1.38, 0.38);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0.0, 1.14, -1.24);

scene.add(new THREE.HemisphereLight(0xffffff, 0x333333, 1.05));
const moon = new THREE.DirectionalLight(0xffffff, 1.2);
moon.position.set(-2, 4, 2);
scene.add(moon);
const lamp = new THREE.SpotLight(0xffffff, 28, 7, 0.7, 0.35, 1.0);
lamp.position.set(0.35, 1.7, -1.0);
lamp.target.position.set(0, 0.8, -1.0);
lamp.castShadow = true;
lamp.shadow.mapSize.set(1024, 1024);
scene.add(lamp);
scene.add(lamp.target);
const fill = new THREE.PointLight(0xffffff, 5.2, 4.2);
fill.position.set(0.2, 1.35, -0.7);
scene.add(fill);

const { desk } = buildRoom(scene);

const clock = new THREE.Clock();
function frame() {
  clock.getDelta();
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
frame();

await Promise.race([
  document.fonts.ready,
  new Promise((r) => setTimeout(r, 1200)),
]);
const keyboard = buildKeyboard(desk);
const monitor = buildMonitor(desk);

function setStatus(text) {
  hud.status.textContent = text;
}

function syncHud(s) {
  if (!s) return;
  lastSnap = { ...lastSnap, ...s };
  hud.need.textContent = lastSnap.prefix ? lastSnap.prefix.toUpperCase() : "none";
  hud.buffer.textContent = lastSnap.buffer ? lastSnap.buffer.toUpperCase() : "_";
  hud.score.textContent = String(lastSnap.score ?? 0);
  if (hud.last) hud.last.textContent = lastSnap.lastWord ? lastSnap.lastWord.toUpperCase() : "—";
  if (hud.lives) {
    const you = lastSnap.lives;
    const bot = lastSnap.botLives;
    hud.lives.textContent = you == null ? "—" : `you ${you} · bot ${bot ?? "—"}`;
  }
  if (hud.tries) hud.tries.textContent = lastSnap.tries == null ? "—" : String(lastSnap.tries);
  if (lastSnap.yourTurn) hud.command.textContent = "your turn";
  else if (lastSnap.opponentTurn) hud.command.textContent = "bot turn";
  else hud.command.textContent = lastSnap.phase || "—";
  if (s?.url) hud.url.value = s.url;
}

function showBrain(info) {
  const live = info?.liveState || {};
  const signed = live.signedIn ? live.displayName || "signed in" : "guest / signing in";
  hud.backend.className = "";
  hud.backend.textContent = live.ok === false
    ? (live.message || "No live browser. Start the server with --live.")
    : `You play on shiritori.lol (${signed}). The fly does not play.`;
}

async function sendKey(raw) {
  const key = normalizeKey(raw);
  if (!key || typing) return;
  typing = true;
  try {
    const mapped = key === "ESCAPE" ? "ESC" : key;
    keyboard.press(mapped).catch(() => {});
    const snap = await liveKey(key);
    syncHud(snap);
    if (snap?.message) setStatus(snap.message);
    else if (snap?.error) setStatus(snap.error);
  } finally {
    typing = false;
  }
}

async function pumpMonitor() {
  try {
    const res = await fetch("/api/browser/frame", { cache: "no-store" });
    if (!res.ok) return;
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      monitor.drawImage(img);
      URL.revokeObjectURL(url);
    };
    img.src = url;
  } catch {
    /* browser still booting */
  }
}

async function pumpState() {
  const snap = await liveState();
  if (snap?.ok) {
    syncHud(snap);
    if (snap.message) setStatus(snap.message);
  }
}

document.getElementById("btn-follow").onclick = () => {
  mode.follow = !mode.follow;
  document.getElementById("btn-follow").textContent = mode.follow ? "Free camera" : "Follow";
};
document.getElementById("url-form").onsubmit = async (ev) => {
  ev.preventDefault();
  const snap = await liveGoto(hud.url.value);
  syncHud(snap);
  setStatus(snap.message || snap.error || snap.url);
};
document.getElementById("btn-casual").onclick = async () => {
  const snap = await liveMode("casual");
  syncHud(snap);
  setStatus(snap.message || snap.error || "Casual match.");
  markMode("casual");
};
document.getElementById("btn-pro").onclick = async () => {
  const snap = await liveMode("pro");
  syncHud(snap);
  setStatus(snap.message || snap.error || "Pro match.");
  markMode("pro");
};
document.getElementById("btn-featherine").onclick = async () => {
  const snap = await liveMode("featherine");
  syncHud(snap);
  setStatus(snap.message || snap.error || "Featherine match.");
  markMode("featherine");
};
document.getElementById("btn-lambdadelta").onclick = async () => {
  const snap = await liveMode("lambdadelta");
  syncHud(snap);
  setStatus(snap.message || snap.error || "Lambdadelta match.");
  markMode("lambdadelta");
};

function markMode(mode) {
  for (const id of ["casual", "pro", "featherine", "lambdadelta"]) {
    const btn = document.getElementById(`btn-${id}`);
    if (btn) btn.classList.toggle("ghost", id !== mode);
  }
}

window.addEventListener("keydown", (ev) => {
  if (ev.target === hud.url || ev.target?.closest?.("input, textarea")) return;
  const key = normalizeKey(ev.key);
  if (!key) return;
  ev.preventDefault();
  sendKey(key);
});

renderer.domElement.addEventListener("pointerdown", (ev) => {
  const ray = new THREE.Raycaster();
  const mouse = new THREE.Vector2(
    (ev.clientX / innerWidth) * 2 - 1,
    -(ev.clientY / innerHeight) * 2 + 1
  );
  ray.setFromCamera(mouse, camera);
  const hits = ray.intersectObjects(scene.children, true);
  for (const hit of hits) {
    if (hit.object.name === "Screen" && hit.uv) {
      liveClick(hit.uv.x, 1 - hit.uv.y);
      return;
    }
    const label = hit.object.parent?.userData?.label || hit.object.userData.label;
    if (label) {
      sendKey(label);
      return;
    }
  }
});

window.addEventListener("resize", () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

setInterval(pumpMonitor, 140);
setInterval(pumpState, 800);
pumpMonitor();
pumpState();

const info = await brainStatus();
showBrain(info);
const live = await liveState();
syncHud(live);
setStatus(live.message || "You play. The fly does not play. Type here or start Casual.");
