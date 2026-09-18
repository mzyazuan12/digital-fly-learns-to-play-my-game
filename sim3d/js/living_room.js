import * as THREE from "three";
import { buildDesk } from "./room.js";
import { buildKeyboard, keycapTopY } from "./keyboard.js?v=nmf18";
import { buildMonitor } from "./monitor.js";

function mm(v) {
  return v / 1000;
}

export const DESK_X = 3.75;
export const DESK_Z = 0.48;

export function buildLivingRoom(scene, layout) {
  const W = mm(layout.width_mm || 5000);
  const D = mm(layout.depth_mm || 4000);
  const H = mm(layout.height_mm || 2800);
  const room = new THREE.Group();
  room.name = "LivingRoom";

  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(W, D),
    new THREE.MeshStandardMaterial({ color: 0xc8c4bc, roughness: 0.92 })
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.set(W / 2, 0, D / 2);
  floor.receiveShadow = true;
  room.add(floor);

  const rug = new THREE.Mesh(
    new THREE.PlaneGeometry(1.8, 1.4),
    new THREE.MeshStandardMaterial({ color: 0x9a8f84, roughness: 0.94 })
  );
  rug.rotation.x = -Math.PI / 2;
  rug.position.set(DESK_X, 0.0006, 1.2);
  rug.receiveShadow = true;
  room.add(rug);

  const wallMat = new THREE.MeshStandardMaterial({ color: 0xe7e2d8, roughness: 0.9 });
  const back = new THREE.Mesh(new THREE.PlaneGeometry(W, H), wallMat);
  back.position.set(W / 2, H / 2, 0);
  room.add(back);
  const left = new THREE.Mesh(new THREE.PlaneGeometry(D, H), wallMat);
  left.rotation.y = Math.PI / 2;
  left.position.set(0, H / 2, D / 2);
  room.add(left);
  const right = left.clone();
  right.position.x = W;
  right.rotation.y = -Math.PI / 2;
  room.add(right);

  const wood = new THREE.MeshStandardMaterial({ color: 0x5a4638, roughness: 0.62 });
  const fabric = new THREE.MeshStandardMaterial({ color: 0x2c2a28, roughness: 0.88 });
  const plant = new THREE.MeshStandardMaterial({ color: 0x3d5a3a, roughness: 0.7 });
  for (const s of layout.solids || []) {
    if (s.name === "desk") continue;
    const sx = mm(s.x1 - s.x0);
    const sy = mm(s.z1 - s.z0);
    const sz = mm(s.y1 - s.y0);
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(Math.max(sx, 0.02), Math.max(sy, 0.02), Math.max(sz, 0.02)),
      s.name.includes("plant") ? plant : s.name === "couch" ? fabric : wood
    );
    mesh.position.set(mm((s.x0 + s.x1) / 2), mm((s.z0 + s.z1) / 2), mm((s.y0 + s.y1) / 2));
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.name = s.name;
    room.add(mesh);
  }

  const desk = buildDesk({ woodColor: 0x5a4638 });
  desk.position.set(DESK_X, 0, DESK_Z);
  room.add(desk);
  const monitor = buildMonitor(desk);
  const keyboard = buildKeyboard(desk);
  room.updateMatrixWorld(true);
  const keyPlantY = keycapTopY(keyboard);

  for (const lamp of layout.lights || []) {
    const light = new THREE.PointLight(0xfff6ea, Math.max(4, (lamp.intensity || 40) / 18), 8);
    light.position.set(mm(lamp.x_mm), mm(lamp.z_mm), mm(lamp.y_mm));
    room.add(light);
  }

  scene.add(room);
  return {
    room,
    desk,
    keyboard,
    keyPlantY,
    width: W,
    depth: D,
    height: H,
    drawScreen: (img) => monitor.drawImage(img),
    screen: monitor.screen,
  };
}
