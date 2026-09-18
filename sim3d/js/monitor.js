import * as THREE from "three";
import { theme } from "./theme.js";

const FONT = 'Inter, "Helvetica Neue", Helvetica, Arial, sans-serif';

export function buildMonitor(desk) {
  const monitor = new THREE.Group();
  monitor.name = "Monitor";

  const stand = new THREE.Mesh(
    new THREE.CylinderGeometry(0.05, 0.1, 0.2, 12),
    new THREE.MeshStandardMaterial({ color: 0x141414 })
  );
  stand.position.set(0, 0.88, -0.28);
  monitor.add(stand);

  const bezel = new THREE.Mesh(
    new THREE.BoxGeometry(0.98, 0.58, 0.04),
    new THREE.MeshStandardMaterial({ color: 0x050505, roughness: 0.35 })
  );
  bezel.position.set(0, 1.28, -0.22);
  bezel.castShadow = true;
  monitor.add(bezel);

  const canvas = document.createElement("canvas");
  canvas.width = 1280;
  canvas.height = 720;
  const ctx = canvas.getContext("2d");
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const screen = new THREE.Mesh(
    new THREE.PlaneGeometry(0.92, 0.518),
    new THREE.MeshBasicMaterial({ map: texture })
  );
  screen.position.set(0, 1.28, -0.198);
  screen.name = "Screen";
  monitor.add(screen);

  desk.add(monitor);

  function draw(state) {
    const w = canvas.width;
    const h = canvas.height;
    ctx.fillStyle = state.flash === "bad" ? "#1a1a1a" : theme.void;
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = theme.gold;
    ctx.font = `600 36px ${FONT}`;
    ctx.fillText("shiritori.lol", 48, 72);
    ctx.fillStyle = theme.muted;
    ctx.font = `22px ${FONT}`;
    ctx.fillText(state.message || "Opening the live browser…", 48, 120);
    texture.needsUpdate = true;
  }

  function drawImage(img) {
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    texture.needsUpdate = true;
  }

  draw({ message: "Live shiritori.lol is starting on this monitor." });

  return {
    monitor,
    screen,
    draw,
    drawImage,
    screenWorldPos(target = new THREE.Vector3()) {
      return screen.getWorldPosition(target);
    },
  };
}
