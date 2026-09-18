import * as THREE from "three";
import { theme } from "./theme.js";

/** Cherry MX pitch, meters. */
const U = 0.01905;
const GAP = 0.001;
const CAP_H = 0.008;
const TOP_H = 0.0022;
const PLATE_Y = 0.011;

/**
 * ANSI 60% in key units. Origin is the inner-left / far edge of the key well.
 * Esc lives on the number row, far from P. Enter is on the home row, after quotes.
 * Backspace is 2u on the number row, after "=".
 */
const LAYOUT = [
  // number row
  ["ESC", "ESC", 0, 0, 1],
  ["1", "1", 1, 0, 1],
  ["2", "2", 2, 0, 1],
  ["3", "3", 3, 0, 1],
  ["4", "4", 4, 0, 1],
  ["5", "5", 5, 0, 1],
  ["6", "6", 6, 0, 1],
  ["7", "7", 7, 0, 1],
  ["8", "8", 8, 0, 1],
  ["9", "9", 9, 0, 1],
  ["0", "0", 10, 0, 1],
  ["-", "-", 11, 0, 1],
  ["=", "=", 12, 0, 1],
  ["BACKSPACE", "BKSP", 13, 0, 2],
  // Q row
  ["TAB", "TAB", 0, 1, 1.5],
  ["Q", "Q", 1.5, 1, 1],
  ["W", "W", 2.5, 1, 1],
  ["E", "E", 3.5, 1, 1],
  ["R", "R", 4.5, 1, 1],
  ["T", "T", 5.5, 1, 1],
  ["Y", "Y", 6.5, 1, 1],
  ["U", "U", 7.5, 1, 1],
  ["I", "I", 8.5, 1, 1],
  ["O", "O", 9.5, 1, 1],
  ["P", "P", 10.5, 1, 1],
  ["[", "[", 11.5, 1, 1],
  ["]", "]", 12.5, 1, 1],
  ["\\", "\\", 13.5, 1, 1.5],
  // home row
  ["CAPS", "CAPS", 0, 2, 1.75],
  ["A", "A", 1.75, 2, 1],
  ["S", "S", 2.75, 2, 1],
  ["D", "D", 3.75, 2, 1],
  ["F", "F", 4.75, 2, 1],
  ["G", "G", 5.75, 2, 1],
  ["H", "H", 6.75, 2, 1],
  ["J", "J", 7.75, 2, 1],
  ["K", "K", 8.75, 2, 1],
  ["L", "L", 9.75, 2, 1],
  [";", ";", 10.75, 2, 1],
  ["'", "'", 11.75, 2, 1],
  ["ENTER", "ENTER", 12.75, 2, 2.25],
  // shift row
  ["LSHIFT", "SHIFT", 0, 3, 2.25],
  ["Z", "Z", 2.25, 3, 1],
  ["X", "X", 3.25, 3, 1],
  ["C", "C", 4.25, 3, 1],
  ["V", "V", 5.25, 3, 1],
  ["B", "B", 6.25, 3, 1],
  ["N", "N", 7.25, 3, 1],
  ["M", "M", 8.25, 3, 1],
  [",", ",", 9.25, 3, 1],
  [".", ".", 10.25, 3, 1],
  ["/", "/", 11.25, 3, 1],
  ["RSHIFT", "SHIFT", 12.25, 3, 2.75],
  // bottom
  ["LCTRL", "CTRL", 0, 4, 1.25],
  ["LWIN", "WIN", 1.25, 4, 1.25],
  ["LALT", "ALT", 2.5, 4, 1.25],
  ["SPACE", "SPACE", 3.75, 4, 6.25],
  ["RALT", "ALT", 10, 4, 1.25],
  ["FN", "FN", 11.25, 4, 1.25],
  ["MENU", "MENU", 12.5, 4, 1.25],
  ["RCTRL", "CTRL", 13.75, 4, 1.25],
];

const ROWS = 5;
const COLS = 15;
const CASE_PAD = 0.008;

function legendTexture(legend, wu, du) {
  const pxW = Math.max(128, Math.round(128 * wu));
  const pxH = Math.max(128, Math.round(128 * du));
  const canvas = document.createElement("canvas");
  canvas.width = pxW;
  canvas.height = pxH;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = theme.keycap;
  ctx.fillRect(0, 0, pxW, pxH);
  ctx.fillStyle = theme.keyLegend;
  const short = legend.length > 1;
  const size = short
    ? Math.max(18, Math.floor(Math.min(pxW / (legend.length + 0.8), pxH * 0.22)))
    : Math.floor(pxH * 0.42);
  ctx.font = `600 ${size}px Inter, "Helvetica Neue", Helvetica, Arial, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(legend, pxW / 2, pxH / 2 + (short ? 0 : pxH * 0.02));
  const map = new THREE.CanvasTexture(canvas);
  map.colorSpace = THREE.SRGBColorSpace;
  map.anisotropy = 8;
  return map;
}

function keyCap(id, legend, wu, du) {
  const group = new THREE.Group();
  const w = wu * U - GAP;
  const d = du * U - GAP;
  const plastic = new THREE.MeshStandardMaterial({
    color: 0xe8e8e8,
    roughness: 0.55,
  });
  const body = new THREE.Mesh(new THREE.BoxGeometry(w, CAP_H, d), plastic);
  body.position.y = CAP_H / 2;
  body.castShadow = true;
  body.receiveShadow = true;

  const top = new THREE.Mesh(
    new THREE.BoxGeometry(Math.max(0.006, w - 0.0032), TOP_H, Math.max(0.006, d - 0.0032)),
    plastic.clone()
  );
  top.position.y = CAP_H + TOP_H / 2;
  top.castShadow = true;
  top.receiveShadow = true;

  const map = legendTexture(legend, wu, du);
  const face = new THREE.Mesh(
    new THREE.PlaneGeometry(Math.max(0.005, w - 0.004), Math.max(0.005, d - 0.004)),
    new THREE.MeshStandardMaterial({ map, roughness: 0.36, polygonOffset: true, polygonOffsetFactor: -1 })
  );
  face.rotation.x = -Math.PI / 2;
  face.position.y = CAP_H + TOP_H + 0.00015;

  group.add(body);
  group.add(top);
  group.add(face);
  group.userData.label = id;
  group.userData.legend = legend;
  group.userData.restY = 0;
  body.userData.label = id;
  top.userData.label = id;
  face.userData.label = id;
  return group;
}

function assertNoOverlap(placed) {
  for (let i = 0; i < placed.length; i++) {
    for (let j = i + 1; j < placed.length; j++) {
      const a = placed[i];
      const b = placed[j];
      const ax1 = a.x;
      const ax2 = a.x + a.w;
      const az1 = a.z;
      const az2 = a.z + a.d;
      const bx1 = b.x;
      const bx2 = b.x + b.w;
      const bz1 = b.z;
      const bz2 = b.z + b.d;
      const overlapX = ax1 < bx2 - 1e-6 && ax2 > bx1 + 1e-6;
      const overlapZ = az1 < bz2 - 1e-6 && az2 > bz1 + 1e-6;
      if (overlapX && overlapZ) {
        throw new Error(`Key overlap: ${a.id} vs ${b.id}`);
      }
    }
  }
}

export function buildPhysicalKeys(parent, specs) {
  const group = new THREE.Group();
  group.name = "FloorKeys";
  const plastic = new THREE.MeshStandardMaterial({
    color: 0xe8e8e8,
    roughness: 0.55,
  });
  const xs = [];
  const zs = [];
  for (const k of specs || []) {
    const half = Math.max(0.004, (Number(k.half_mm) || 8.2) / 1000);
    xs.push((k.x_mm || 0) / 1000 - half, (k.x_mm || 0) / 1000 + half);
    zs.push((k.y_mm || 0) / 1000 - half, (k.y_mm || 0) / 1000 + half);
  }
  if (xs.length) {
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minZ = Math.min(...zs);
    const maxZ = Math.max(...zs);
    const plate = new THREE.Mesh(
      new THREE.BoxGeometry(maxX - minX + 0.014, 0.0032, maxZ - minZ + 0.014),
      new THREE.MeshStandardMaterial({ color: 0x141414, roughness: 0.48 })
    );
    plate.position.set((minX + maxX) / 2, 0.0016, (minZ + maxZ) / 2);
    plate.receiveShadow = true;
    plate.castShadow = true;
    group.add(plate);
  }
  for (const k of specs || []) {
    const half = Math.max(0.004, (Number(k.half_mm) || 8.2) / 1000);
    const w = half * 2;
    const d = half * 2;
    const h = 0.0044;
    const cap = new THREE.Group();
    const body = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), plastic);
    body.position.y = h / 2;
    body.castShadow = true;
    body.receiveShadow = true;
    const map = legendTexture(String(k.label || "").toUpperCase(), 1, 1);
    const face = new THREE.Mesh(
      new THREE.PlaneGeometry(Math.max(0.006, w - 0.003), Math.max(0.006, d - 0.003)),
      new THREE.MeshStandardMaterial({
        map,
        roughness: 0.36,
        polygonOffset: true,
        polygonOffsetFactor: -1,
      })
    );
    face.rotation.x = -Math.PI / 2;
    face.position.y = h + 0.00012;
    cap.add(body);
    cap.add(face);
    cap.userData.label = k.label;
    cap.position.set((k.x_mm || 0) / 1000, (k.z_mm || 2.2) / 1000, (k.y_mm || 0) / 1000);
    group.add(cap);
  }
  parent.add(group);
  return group;
}

export function buildKeyboard(desk) {
  const board = new THREE.Group();
  board.name = "Keyboard";

  const innerW = COLS * U;
  const innerD = ROWS * U;
  const caseW = innerW + CASE_PAD * 2;
  const caseD = innerD + CASE_PAD * 2;

  const shell = new THREE.Mesh(
    new THREE.BoxGeometry(caseW, 0.018, caseD),
    new THREE.MeshStandardMaterial({ color: theme.keyWell, roughness: 0.48 })
  );
  shell.position.y = 0.009;
  shell.castShadow = true;
  shell.receiveShadow = true;
  board.add(shell);

  const well = new THREE.Mesh(
    new THREE.BoxGeometry(innerW + 0.002, 0.004, innerD + 0.002),
    new THREE.MeshStandardMaterial({ color: 0x101010, roughness: 0.7 })
  );
  well.position.y = 0.017;
  well.receiveShadow = true;
  board.add(well);

  const keys = new Map();
  const placed = [];
  const x0 = -innerW / 2;
  const z0 = -innerD / 2;

  for (const [id, legend, xu, row, wu] of LAYOUT) {
    const cap = keyCap(id, legend, wu, 1);
    const localX = x0 + (xu + wu / 2) * U;
    const localZ = z0 + (row + 0.5) * U;
    cap.position.set(localX, PLATE_Y, localZ);
    board.add(cap);
    keys.set(id, cap);
    if (id === "ESC") keys.set("ESCAPE", cap);
    placed.push({ id, x: xu, z: row, w: wu, d: 1 });
  }
  assertNoOverlap(placed);

  board.rotation.x = 0.06;
  board.position.set(0, 0.79, 0.14);
  desk.add(board);

  return {
    board,
    keys,
    worldPos(label, target = new THREE.Vector3()) {
      const cap = keys.get(label);
      if (!cap) return null;
      return cap.getWorldPosition(target);
    },
    async press(label) {
      const cap = keys.get(label);
      if (!cap) return;
      cap.position.y = PLATE_Y - 0.004;
      await new Promise((r) => setTimeout(r, 80));
      cap.position.y = PLATE_Y;
    },
  };
}

/** World Y of a keycap top after the board is parented to the desk. */
export function keycapTopY(keyboard, id = "G") {
  if (!keyboard?.board) return 0.811;
  keyboard.board.updateMatrixWorld(true);
  const cap =
    keyboard.keys.get(id) ||
    keyboard.keys.get("A") ||
    (keyboard.keys.size ? keyboard.keys.values().next().value : null);
  if (!cap) return 0.811;
  return new THREE.Box3().setFromObject(cap).max.y;
}
