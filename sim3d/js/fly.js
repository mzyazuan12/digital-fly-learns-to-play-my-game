import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

function segmentName(raw) {
  const n = String(raw || "");
  const slash = n.lastIndexOf("/");
  return slash >= 0 ? n.slice(slash + 1) : n;
}

function driveAndGeom(raw) {
  const n = String(raw || "");
  const tag = "__vis__";
  const i = n.indexOf(tag);
  if (i >= 0) return { drive: n.slice(0, i), geom: n.slice(i + tag.length) };
  return { drive: n, geom: n };
}

/** FlyBody visuals.yaml — the TuragaLab fruitfly, not NMF collision paint. */
function flybodyMaterial(geomName) {
  const n = String(geomName || "").toLowerCase();
  if (n.includes("membrane")) {
    return new THREE.MeshPhysicalMaterial({
      color: 0x89afcc,
      roughness: 0.2,
      metalness: 0.0,
      transmission: 0.55,
      thickness: 0.0003,
      transparent: true,
      opacity: 0.42,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
  }
  if (n.includes("_red")) {
    return new THREE.MeshStandardMaterial({
      color: 0xcc0700,
      roughness: 0.2,
      metalness: 0.0,
      emissive: 0xcc0700,
      emissiveIntensity: 0.28,
    });
  }
  if (n.includes("ocelli")) {
    return new THREE.MeshStandardMaterial({ color: 0x210c04, roughness: 0.35, metalness: 0.08 });
  }
  if (n.includes("bristle") || n.includes("_black")) {
    return new THREE.MeshStandardMaterial({ color: 0x0a0806, roughness: 0.62, metalness: 0.02 });
  }
  if (n.includes("wing") && n.includes("brown")) {
    return new THREE.MeshStandardMaterial({ color: 0x341407, roughness: 0.45, metalness: 0.0 });
  }
  if (n.includes("_lower")) {
    return new THREE.MeshStandardMaterial({ color: 0xcc9c62, roughness: 0.48, metalness: 0.0 });
  }
  if (n.endsWith("_brown") || n.includes("tarsus5")) {
    return new THREE.MeshStandardMaterial({ color: 0x341407, roughness: 0.5, metalness: 0.0 });
  }
  return new THREE.MeshStandardMaterial({
    color: 0xac5924,
    roughness: 0.42,
    metalness: 0.02,
  });
}

function paintMesh(obj, geomName) {
  obj.material = flybodyMaterial(geomName || obj.name);
  const n = String(geomName || obj.name || "").toLowerCase();
  const clear = Boolean(obj.material.transparent || obj.material.transmission);
  obj.castShadow = !clear;
  obj.receiveShadow = false;
  if (n.includes("_red")) {
    obj.material.side = THREE.DoubleSide;
    obj.material.toneMapped = false;
    obj.renderOrder = 2;
    obj.scale.setScalar(1.02);
  }
}

/** STL / MuJoCo body frame is Z-up. Three.js is Y-up. Swap Y↔Z so P R P is a real rotation. */
function mujocoToThreeGeometry(geometry) {
  const geom = geometry.index ? geometry.clone() : geometry.clone();
  const pos = geom.getAttribute("position");
  if (pos) {
    const arr = pos.array;
    for (let i = 0; i < arr.length; i += 3) {
      const y = arr[i + 1];
      arr[i + 1] = arr[i + 2];
      arr[i + 2] = y;
    }
    pos.needsUpdate = true;
  }
  const nrm = geom.getAttribute("normal");
  if (nrm) {
    const arr = nrm.array;
    for (let i = 0; i < arr.length; i += 3) {
      const y = arr[i + 1];
      arr[i + 1] = arr[i + 2];
      arr[i + 2] = y;
    }
    nrm.needsUpdate = true;
  }
  const idx = geom.getIndex();
  if (idx) {
    const arr = idx.array;
    for (let i = 0; i < arr.length; i += 3) {
      const a = arr[i + 1];
      arr[i + 1] = arr[i + 2];
      arr[i + 2] = a;
    }
    idx.needsUpdate = true;
  }
  geom.computeBoundingBox();
  geom.computeBoundingSphere();
  return geom;
}

function headingFromWxyz(q) {
  const w = q[0];
  const x = q[1];
  const y = q[2];
  const z = q[3];
  return Math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
}

const _perm = new THREE.Matrix4().set(1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1);
const _tmpM = new THREE.Matrix4();
const _tmpR = new THREE.Matrix4();
const _tmpQ = new THREE.Quaternion();
const _wingQ = new THREE.Quaternion();
const _wingAxis = new THREE.Vector3(1, 0, 0);
const _worldUp = new THREE.Vector3(0, 1, 0);
const _thoraxUp = new THREE.Vector3();
const _levelQ = new THREE.Quaternion();
const _pivot = new THREE.Vector3();
const _off = new THREE.Vector3();

/** Keep yaw; remove stance pitch/roll so an 8× display does not look fallen. */
function levelStance(nodes, thoraxNode) {
  if (!thoraxNode?.visible) return;
  _thoraxUp.set(0, 1, 0).applyQuaternion(thoraxNode.quaternion);
  const dorsal = Math.max(-1, Math.min(1, _thoraxUp.y));
  const tilt = Math.acos(dorsal);
  if (tilt < 0.035 || tilt > 1.05) return;
  _levelQ.setFromUnitVectors(_thoraxUp, _worldUp);
  _pivot.copy(thoraxNode.position);
  for (const node of nodes.values()) {
    if (!node.visible) continue;
    _off.copy(node.position).sub(_pivot).applyQuaternion(_levelQ);
    node.position.copy(_pivot).add(_off);
    node.quaternion.premultiply(_levelQ);
  }
}

function applyMjPose(node, posMm, quatWxyz, origin, display) {
  const x = posMm[0] / 1000;
  const y = posMm[2] / 1000;
  const z = posMm[1] / 1000;
  node.position.set(
    origin.x + (x - origin.x) * display,
    origin.y + (y - origin.y) * display,
    origin.z + (z - origin.z) * display
  );
  _tmpQ.set(quatWxyz[1], quatWxyz[2], quatWxyz[3], quatWxyz[0]);
  _tmpR.makeRotationFromQuaternion(_tmpQ);
  _tmpM.copy(_perm).multiply(_tmpR).multiply(_perm);
  node.quaternion.setFromRotationMatrix(_tmpM);
  node.scale.setScalar(display);
}

export async function createFly({ length = 0.003, display = 1, plantY = 0.0044 } = {}) {
  const root = new THREE.Group();
  root.name = "Fly";
  const body = new THREE.Group();
  body.name = "NeuroMechFly";
  root.add(body);

  const loader = new GLTFLoader();
  const gltf = await loader.loadAsync("./models/flybody.glb?v=nmf30");
  const nodes = new Map();
  function nodeFor(name) {
    let node = nodes.get(name);
    if (!node) {
      node = new THREE.Group();
      node.name = name;
      body.add(node);
      nodes.set(name, node);
    }
    return node;
  }
  const meshes = [];
  gltf.scene.updateMatrixWorld(true);
  gltf.scene.traverse((obj) => {
    if (obj.isMesh) meshes.push(obj);
  });
  for (const obj of meshes) {
    const { drive, geom } = driveAndGeom(obj.name || obj.parent?.name);
    obj.name = geom || obj.name;
    obj.geometry = mujocoToThreeGeometry(obj.geometry);
    obj.geometry.computeVertexNormals();
    paintMesh(obj, obj.name);
    obj.frustumCulled = false;
    obj.removeFromParent();
    obj.position.set(0, 0, 0);
    obj.quaternion.identity();
    if (!String(obj.name || "").toLowerCase().includes("_red")) obj.scale.set(1, 1, 1);
    nodeFor(drive || "c_thorax").add(obj);
  }
  nodeFor("c_thorax");

  const shadow = new THREE.Mesh(
    new THREE.CircleGeometry(length * 0.22 * display, 20),
    new THREE.MeshBasicMaterial({
      color: 0x1a1008,
      transparent: true,
      opacity: 0.22,
      depthWrite: false,
    })
  );
  shadow.rotation.x = -Math.PI / 2;
  shadow.position.y = plantY + 0.00012;
  shadow.renderOrder = -1;

  const key = new THREE.PointLight(0xffffff, 0.04 * display, 0.08 * display, 2);
  const fill = new THREE.PointLight(0xfff1dc, 0.015 * display, 0.06 * display, 2);

  let flying = false;
  let walking = false;
  let grooming = false;
  let wingPhase = 0;
  let heading = 0;
  const thorax = new THREE.Vector3();
  const origin = new THREE.Vector3();

  function placeLights() {
    const s = display;
    key.position.set(thorax.x + 0.01 * s, thorax.y + 0.014 * s, thorax.z + 0.008 * s);
    fill.position.set(thorax.x - 0.008 * s, thorax.y + 0.004 * s, thorax.z - 0.006 * s);
    shadow.position.set(thorax.x, plantY + 0.00012, thorax.z);
    shadow.scale.setScalar(1);
  }

  function trackFromThorax(node, posMm, quatWxyz) {
    if (node) {
      node.updateWorldMatrix(true, false);
      node.getWorldPosition(thorax);
    } else if (posMm) {
      thorax.set(posMm[0] / 1000, posMm[2] / 1000, posMm[1] / 1000);
    }
    if (quatWxyz) heading = headingFromWxyz(quatWxyz);
    placeLights();
  }

  function place(xMm, yMm, zMm, yaw) {
    heading = yaw;
    root.position.set(xMm / 1000, zMm / 1000, yMm / 1000);
    root.quaternion.set(0, -Math.sin(yaw / 2), 0, Math.cos(yaw / 2));
    thorax.set(xMm / 1000, zMm / 1000, yMm / 1000);
    placeLights();
  }

  return {
    root,
    pose: body,
    length,
    display,
    nodes,
    thorax,
    key,
    fill,
    shadow,
    get heading() {
      return heading;
    },
    setFlying(v) {
      flying = Boolean(v);
      if (flying) {
        walking = false;
        grooming = false;
      }
    },
    setWalking(v) {
      walking = Boolean(v);
      if (walking) flying = false;
    },
    setGrooming(v) {
      grooming = Boolean(v);
      if (grooming) flying = false;
    },
    setGait(_phase) {},
    applyBodies(bodies, extras = {}) {
      if (!bodies?.length) return;
      root.position.set(0, 0, 0);
      root.quaternion.identity();
      const thoraxMj = bodies.find((b) => segmentName(b.s || b.segment) === "c_thorax");
      const tpos = thoraxMj?.p || thoraxMj?.pos_mm;
      origin.set(tpos ? tpos[0] / 1000 : 0, tpos ? tpos[2] / 1000 : 0, tpos ? tpos[1] / 1000 : 0);
      const seen = new Set();
      for (const b of bodies) {
        const name = segmentName(b.s || b.segment);
        const node = nodes.get(name);
        if (!node) continue;
        const pos = b.p || b.pos_mm;
        const quat = b.q || b.quat_wxyz;
        if (!pos || !quat) continue;
        applyMjPose(node, pos, quat, origin, display);
        node.visible = true;
        seen.add(name);
      }
      for (const [name, node] of nodes) {
        if (!seen.has(name)) node.visible = false;
      }
      const thoraxNode = nodes.get("c_thorax");
      if (extras.mode !== "fly") levelStance(nodes, thoraxNode);
      if (typeof extras.wingPhase === "number") wingPhase = extras.wingPhase;
      const beat = flying || extras.mode === "fly" ? 0.55 * Math.sin(wingPhase) : 0;
      if (beat) {
        for (const side of ["l_wing", "r_wing"]) {
          const node = nodes.get(side);
          if (!node || !node.visible) continue;
          const sign = side.startsWith("l") ? 1 : -1;
          _wingQ.setFromAxisAngle(_wingAxis, sign * beat);
          node.quaternion.multiply(_wingQ);
        }
      }
      if (thoraxMj) {
        trackFromThorax(thoraxNode, tpos, thoraxMj.q || thoraxMj.quat_wxyz);
      }
      let minY = Infinity;
      const box = new THREE.Box3();
      for (const [name, node] of nodes) {
        if (!node.visible) continue;
        if (!name.includes("tarsus5")) continue;
        box.setFromObject(node);
        if (box.min.y < minY) minY = box.min.y;
      }
      if (!Number.isFinite(minY)) {
        for (const node of nodes.values()) {
          if (!node.visible) continue;
          box.setFromObject(node);
          if (box.min.y < minY) minY = box.min.y;
        }
      }
      const lift = plantY - minY;
      if (Number.isFinite(minY) && Math.abs(lift) > 1e-6) {
        for (const node of nodes.values()) {
          if (node.visible) node.position.y += lift;
        }
        thorax.y += lift;
        placeLights();
      }
    },
    applyRoot(xMm, yMm, zMm, yaw) {
      place(xMm, yMm, zMm, yaw);
    },
    update(_t, dt) {
      if (flying) wingPhase += 80 * Math.max(0, dt || 0);
      return flying || walking || grooming;
    },
    lookToward(_worldPoint) {
      return;
    },
  };
}
