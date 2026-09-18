import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { buildLivingRoom } from "./living_room.js?v=nmf30";
import { createFly } from "./fly.js?v=nmf30";

function frameOnFly(camera, controls, p, yaw, display = 1) {
  const body = 0.003 * display;
  const fx = Math.cos(yaw);
  const fz = Math.sin(yaw);
  const rx = Math.sin(yaw);
  const rz = -Math.cos(yaw);
  camera.position.set(
    p.x + fx * -1.4 * body + rx * 5.2 * body,
    p.y + 3.0 * body,
    p.z + fz * -1.4 * body + rz * 5.2 * body
  );
  controls.target.set(p.x, p.y + 0.25 * body, p.z);
}

export async function startLivingRoom({ follow = false } = {}) {
  const viewport = document.getElementById("viewport");
  const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.setSize(innerWidth, innerHeight);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.NeutralToneMapping;
  renderer.toneMappingExposure = 0.92;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  viewport.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xc5c1b8);
  scene.fog = new THREE.Fog(0xc5c1b8, 12, 30);

  const camera = new THREE.PerspectiveCamera(38, innerWidth / innerHeight, 0.0008, 40);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  controls.maxPolarAngle = Math.PI / 2.02;
  controls.minDistance = 0.02;
  controls.maxDistance = 16;
  controls.screenSpacePanning = true;

  scene.add(new THREE.HemisphereLight(0xfff7ee, 0x6d6a64, 0.38));
  const sun = new THREE.DirectionalLight(0xffffff, 1.35);
  sun.position.set(3.4, 5.2, 2.2);
  sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  sun.shadow.camera.near = 0.2;
  sun.shadow.camera.far = 18;
  sun.shadow.camera.left = -5;
  sun.shadow.camera.right = 5;
  sun.shadow.camera.top = 5;
  sun.shadow.camera.bottom = -5;
  scene.add(sun);

  async function loadRoom() {
    for (;;) {
      const res = await fetch("/api/room");
      if (res.ok) {
        const layout = await res.json();
        if (layout?.width_mm) return layout;
      }
      await new Promise((r) => setTimeout(r, 400));
    }
  }
  await Promise.race([
    document.fonts.ready,
    new Promise((r) => setTimeout(r, 800)),
  ]);
  const layout = await loadRoom();
  const built = buildLivingRoom(scene, layout);
  const spawn = layout.spawn_mm || [3726, 620, 0.6];
  const plantY = built.keyPlantY || 0.811;
  const fly = await createFly({ length: 0.003, display: 8, plantY });
  fly.setFlying(false);
  scene.add(fly.root);
  if (fly.shadow) scene.add(fly.shadow);
  if (fly.key) scene.add(fly.key);
  if (fly.fill) scene.add(fly.fill);
  fly.applyRoot(spawn[0], spawn[1], plantY * 1000, 0.15);
  frameOnFly(camera, controls, fly.thorax, fly.heading || 0.15, fly.display);

  const keys = new Set();
  const mode = { follow };
  const clock = new THREE.Clock();
  const wish = new THREE.Vector3();
  const forward = new THREE.Vector3();
  const right = new THREE.Vector3();
  const up = new THREE.Vector3(0, 1, 0);

  function thorax() {
    return fly.thorax || fly.root.position;
  }

  function frame() {
    try {
      const dt = Math.min(0.05, clock.getDelta());
      fly.update(clock.elapsedTime, dt);
      camera.getWorldDirection(forward);
      forward.y = 0;
      if (forward.lengthSq() < 1e-6) forward.set(0, 0, -1);
      else forward.normalize();
      right.crossVectors(forward, up).normalize();
      wish.set(0, 0, 0);
      if (!mode.follow) {
        if (keys.has("KeyW") || keys.has("ArrowUp")) wish.add(forward);
        if (keys.has("KeyS") || keys.has("ArrowDown")) wish.sub(forward);
        if (keys.has("KeyA") || keys.has("ArrowLeft")) wish.sub(right);
        if (keys.has("KeyD") || keys.has("ArrowRight")) wish.add(right);
        if (keys.has("KeyQ") || keys.has("PageDown")) wish.y -= 1;
        if (keys.has("KeyE") || keys.has("PageUp")) wish.y += 1;
        if (wish.lengthSq() > 0) {
          wish.normalize();
          const dist = camera.position.distanceTo(controls.target);
          const speed = THREE.MathUtils.clamp(dist * 1.8, 0.02, 2.4);
          camera.position.addScaledVector(wish, speed * dt);
          controls.target.addScaledVector(wish, speed * dt);
        }
      }
      if (mode.follow) {
        controls.enabled = false;
        frameOnFly(camera, controls, thorax(), fly.heading || 0, fly.display);
        camera.lookAt(controls.target);
      } else {
        controls.enabled = true;
        controls.update();
      }
      renderer.render(scene, camera);
    } catch (err) {
      console.error(err);
    }
    requestAnimationFrame(frame);
  }
  frame();

  window.addEventListener("keydown", (ev) => {
    if (ev.target instanceof HTMLInputElement || ev.metaKey || ev.ctrlKey) return;
    keys.add(ev.code);
  });
  window.addEventListener("keyup", (ev) => keys.delete(ev.code));
  renderer.domElement.addEventListener("pointerdown", (ev) => {
    if (!built.screen) return;
    const ray = new THREE.Raycaster();
    const mouse = new THREE.Vector2(
      (ev.clientX / innerWidth) * 2 - 1,
      -(ev.clientY / innerHeight) * 2 + 1
    );
    ray.setFromCamera(mouse, camera);
    const hits = ray.intersectObject(built.screen, false);
    if (!hits.length || !hits[0].uv) return;
    fetch("/api/browser/click", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ x: hits[0].uv.x, y: 1 - hits[0].uv.y }),
    }).catch(() => {});
  });
  window.addEventListener("resize", () => {
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });

  async function pumpScreen() {
    try {
      const res = await fetch("/api/browser/frame", { cache: "no-store" });
      if (!res.ok) return;
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => {
        built.drawScreen(img);
        URL.revokeObjectURL(url);
      };
      img.src = url;
    } catch {
      /* live browser still booting */
    }
  }
  setInterval(pumpScreen, 140);
  pumpScreen();

  const builtRoom = {
    fly,
    camera,
    controls,
    mode,
    plantY,
    drawScreen: built.drawScreen,
    setFollow(v) {
      mode.follow = Boolean(v);
      controls.enabled = !mode.follow;
      if (!mode.follow) {
        const p = thorax();
        controls.target.set(p.x, p.y + 0.0007 * (fly.display || 1), p.z);
      }
    },
  };
  window.__room = builtRoom;
  return builtRoom;
}
