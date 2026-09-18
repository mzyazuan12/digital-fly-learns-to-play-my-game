import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const REGION_TINT = {
  ol_L: 0.78,
  cb: 1.0,
  ol_R: 0.78,
  gng: 0.58,
  vnc: 0.42,
};

function ellipsoidMesh(c, r, mat) {
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 24), mat);
  mesh.position.set(c[0], c[1], c[2]);
  mesh.scale.set(r[0], r[1], r[2]);
  return mesh;
}

export async function createBrainView(host) {
  async function loadLayout() {
    for (;;) {
      const layout = await fetch("/api/brain-layout").then((r) => r.json());
      if (layout?.n_display) return layout;
      const note = document.getElementById("brain-count");
      if (note) note.textContent = layout?.error || "Loading connectome…";
      await new Promise((r) => setTimeout(r, 400));
    }
  }
  const layout = await loadLayout();
  const n = layout.n_display || 0;
  const atlas = layout.atlas || {};
  const regions = layout.regions || ["ol_L", "cb", "ol_R", "gng", "vnc"];
  const positions = new Float32Array(n * 3);
  const colors = new Float32Array(n * 3);
  const base = new Float32Array(n);
  for (let i = 0; i < n; i += 1) {
    positions[i * 3] = layout.x[i];
    positions[i * 3 + 1] = layout.y[i];
    positions[i * 3 + 2] = layout.z[i];
    const name = regions[layout.region[i]] || "cb";
    base[i] = REGION_TINT[name] ?? 0.8;
    const t = 0.22 * base[i];
    colors[i * 3] = t;
    colors[i * 3 + 1] = t;
    colors[i * 3 + 2] = t;
  }

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x050505);
  const camera = new THREE.PerspectiveCamera(40, 1, 0.05, 20);
  camera.position.set(0, 0.2, 5.4);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.localClippingEnabled = true;
  host.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, -0.4, 0);
  controls.minDistance = 1.4;
  controls.maxDistance = 7;

  scene.add(new THREE.AmbientLight(0xffffff, 0.32));
  const key = new THREE.DirectionalLight(0xffffff, 0.9);
  key.position.set(0.6, 1.4, 2.4);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 0.22);
  fill.position.set(-1.2, -0.4, -1.5);
  scene.add(fill);

  const clipPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
  const clipPlanes = [clipPlane];
  const shellMat = new THREE.MeshStandardMaterial({
    color: 0xb0b0b0,
    transparent: true,
    opacity: 0.13,
    roughness: 0.82,
    metalness: 0.04,
    depthWrite: false,
    side: THREE.DoubleSide,
    clippingPlanes: clipPlanes,
  });
  const wireMat = new THREE.MeshBasicMaterial({
    color: 0xffffff,
    wireframe: true,
    transparent: true,
    opacity: 0.11,
    depthWrite: false,
    clippingPlanes: clipPlanes,
  });

  const anatomy = new THREE.Group();
  const shells = [];
  function addShell(name, mesh) {
    mesh.name = name;
    anatomy.add(mesh);
    shells.push(mesh);
    const wire = mesh.clone();
    wire.material = wireMat;
    wire.scale.multiplyScalar(1.002);
    anatomy.add(wire);
  }
  if (atlas.ol_L && layout.coords_source !== "somaLocation") addShell("ol_L", ellipsoidMesh(atlas.ol_L.c, atlas.ol_L.r, shellMat.clone()));
  if (atlas.cb && layout.coords_source !== "somaLocation") addShell("cb", ellipsoidMesh(atlas.cb.c, atlas.cb.r, shellMat.clone()));
  if (atlas.ol_R && layout.coords_source !== "somaLocation") addShell("ol_R", ellipsoidMesh(atlas.ol_R.c, atlas.ol_R.r, shellMat.clone()));
  if (atlas.gng && layout.coords_source !== "somaLocation") addShell("gng", ellipsoidMesh(atlas.gng.c, atlas.gng.r, shellMat.clone()));
  if (atlas.vnc && layout.coords_source !== "somaLocation") addShell("vnc", ellipsoidMesh(atlas.vnc.c, atlas.vnc.r, shellMat.clone()));
  scene.add(anatomy);

  if (layout.silhouette_x?.length) {
    const sn = layout.silhouette_x.length;
    const spos = new Float32Array(sn * 3);
    const scol = new Float32Array(sn * 3);
    for (let i = 0; i < sn; i += 1) {
      spos[i * 3] = layout.silhouette_x[i];
      spos[i * 3 + 1] = layout.silhouette_y[i];
      spos[i * 3 + 2] = layout.silhouette_z[i];
      scol[i * 3] = 0.18;
      scol[i * 3 + 1] = 0.18;
      scol[i * 3 + 2] = 0.18;
    }
    const sgeo = new THREE.BufferGeometry();
    sgeo.setAttribute("position", new THREE.BufferAttribute(spos, 3));
    sgeo.setAttribute("color", new THREE.BufferAttribute(scol, 3));
    scene.add(
      new THREE.Points(
        sgeo,
        new THREE.PointsMaterial({
          size: 0.018,
          vertexColors: true,
          transparent: true,
          opacity: 0.35,
          sizeAttenuation: true,
          clippingPlanes: clipPlanes,
        })
      )
    );
  }

  const sliceHelper = new THREE.Mesh(
    new THREE.PlaneGeometry(2.8, 2.8),
    new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.07,
      side: THREE.DoubleSide,
      depthWrite: false,
    })
  );
  scene.add(sliceHelper);

  const edgePre = layout.edge_pre || [];
  const edgePost = layout.edge_post || [];
  const edgeSign = layout.edge_sign || [];
  const nEdges = edgePre.length;
  const epos = new Float32Array(nEdges * 6);
  const ecol = new Float32Array(nEdges * 6);
  const edgeGlow = new Float32Array(nEdges);
  for (let i = 0; i < nEdges; i += 1) {
    const a = edgePre[i];
    const b = edgePost[i];
    epos[i * 6] = positions[a * 3];
    epos[i * 6 + 1] = positions[a * 3 + 1];
    epos[i * 6 + 2] = positions[a * 3 + 2];
    epos[i * 6 + 3] = positions[b * 3];
    epos[i * 6 + 4] = positions[b * 3 + 1];
    epos[i * 6 + 5] = positions[b * 3 + 2];
    for (let k = 0; k < 6; k += 1) ecol[i * 6 + k] = 0.08;
  }
  const edgeGeo = new THREE.BufferGeometry();
  edgeGeo.setAttribute("position", new THREE.BufferAttribute(epos, 3));
  edgeGeo.setAttribute("color", new THREE.BufferAttribute(ecol, 3));
  const edges = new THREE.LineSegments(
    edgeGeo,
    new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.72,
      depthWrite: false,
      clippingPlanes: clipPlanes,
    })
  );
  scene.add(edges);

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  const points = new THREE.Points(
    geo,
    new THREE.PointsMaterial({
      size: 0.042,
      vertexColors: true,
      transparent: true,
      opacity: 0.95,
      sizeAttenuation: true,
      clippingPlanes: clipPlanes,
    })
  );
  scene.add(points);

  const glow = new Float32Array(n);
  const raster = document.getElementById("brain-raster");
  const rctx = raster?.getContext("2d");
  const slice = document.getElementById("brain-slice");
  const sctx = slice?.getContext("2d");
  const history = [];
  const HIST = 120;
  const rowMap = new Map();
  let axis = "z";
  let depth = 0;
  let lastActivity = null;
  let clipOn = false;

  function planeNormal() {
    if (axis === "x") return new THREE.Vector3(1, 0, 0);
    if (axis === "y") return new THREE.Vector3(0, 1, 0);
    return new THREE.Vector3(0, 0, 1);
  }

  function applyClip() {
    const nrm = planeNormal();
    clipPlane.normal.copy(nrm);
    clipPlane.constant = -depth;
    renderer.clippingPlanes = clipOn ? clipPlanes : [];
    sliceHelper.visible = clipOn;
    sliceHelper.position.copy(nrm.clone().multiplyScalar(depth));
    sliceHelper.lookAt(sliceHelper.position.clone().add(nrm));
    paintSlice();
  }

  function resize() {
    const w = host.clientWidth || 360;
    const h = host.clientHeight || 280;
    renderer.setSize(w, h);
    camera.aspect = w / Math.max(1, h);
    camera.updateProjectionMatrix();
    if (raster) {
      raster.width = w;
      raster.height = 64;
    }
    if (slice) {
      slice.width = w;
      slice.height = 140;
    }
    paintSlice();
    paintRaster();
  }
  resize();
  window.addEventListener("resize", resize);

  function paintRaster() {
    if (!rctx || !raster) return;
    const w = raster.width;
    const h = raster.height;
    rctx.fillStyle = "#050505";
    rctx.fillRect(0, 0, w, h);
    if (!history.length || n === 0) return;
    const colW = w / HIST;
    const rows = 64;
    const rowH = h / rows;
    history.forEach((hit, t) => {
      const x = Math.floor((t / HIST) * w);
      hit.forEach((idx) => {
        if (!rowMap.has(idx)) {
          if (rowMap.size >= rows) return;
          rowMap.set(idx, rowMap.size);
        }
        const row = rowMap.get(idx);
        rctx.fillStyle = "#ffffff";
        rctx.fillRect(x, row * rowH, Math.max(1, colW), Math.max(1, rowH - 0.4));
      });
    });
  }

  function project(px, py, pz) {
    const w = slice.width;
    const h = slice.height;
    let u;
    let v;
    if (axis === "z") {
      u = px;
      v = py;
    } else if (axis === "x") {
      u = pz;
      v = py;
    } else {
      u = px;
      v = pz;
    }
    return [(u + 1.55) / 3.1 * w, (1.0 - (v + 0.55) / 2.4) * h];
  }

  function paintSlice() {
    if (!sctx || !slice) return;
    const w = slice.width;
    const h = slice.height;
    sctx.fillStyle = "#050505";
    sctx.fillRect(0, 0, w, h);
    sctx.strokeStyle = "#4a4a4a";
    sctx.lineWidth = 1;
    Object.values(atlas).forEach((spec) => {
      if (layout.coords_source === "somaLocation") return;
      const [cx, cy, cz] = spec.c;
      const [rx, ry, rz] = spec.r;
      const along = axis === "x" ? cx : axis === "y" ? cy : cz;
      const rad = axis === "x" ? rx : axis === "y" ? ry : rz;
      const dist = Math.abs(along - depth);
      if (dist > rad) return;
      const scale = Math.sqrt(Math.max(0.08, 1 - (dist / rad) ** 2));
      let erx;
      let ery;
      let eu;
      let ev;
      if (axis === "z") {
        eu = cx;
        ev = cy;
        erx = rx * scale;
        ery = ry * scale;
      } else if (axis === "x") {
        eu = cz;
        ev = cy;
        erx = rz * scale;
        ery = ry * scale;
      } else {
        eu = cx;
        ev = cz;
        erx = rx * scale;
        ery = rz * scale;
      }
      const [sx, sy] = axis === "z"
        ? project(eu, ev, 0)
        : axis === "x"
          ? project(0, ev, eu)
          : project(eu, 0, ev);
      const [ex] = axis === "z"
        ? project(eu + erx, ev, 0)
        : axis === "x"
          ? project(0, ev, eu + erx)
          : project(eu + erx, 0, ev);
      const [, ey] = axis === "z"
        ? project(eu, ev + ery, 0)
        : axis === "x"
          ? project(0, ev + ery, eu)
          : project(eu, 0, ev + ery);
      sctx.beginPath();
      sctx.ellipse(sx, sy, Math.abs(ex - sx), Math.abs(ey - sy), 0, 0, Math.PI * 2);
      sctx.stroke();
    });
    const thick = 0.07;
    const spiked = new Set(lastActivity?.spiked || []);
    for (let i = 0; i < n; i += 1) {
      const px = positions[i * 3];
      const py = positions[i * 3 + 1];
      const pz = positions[i * 3 + 2];
      const coord = axis === "x" ? px : axis === "y" ? py : pz;
      if (Math.abs(coord - depth) > thick) continue;
      const [sx, sy] = project(px, py, pz);
      const fire = spiked.has(i);
      sctx.fillStyle = fire ? "#ffffff" : "#5a5a5a";
      sctx.beginPath();
      sctx.arc(sx, sy, fire ? 2.4 : 1.1, 0, Math.PI * 2);
      sctx.fill();
    }
    const fired = lastActivity?.fired_edges || [];
    sctx.strokeStyle = "rgba(255,255,255,0.55)";
    sctx.lineWidth = 0.8;
    fired.slice(0, 400).forEach((ei) => {
      const a = edgePre[ei];
      const b = edgePost[ei];
      const az = axis === "x" ? positions[a * 3] : axis === "y" ? positions[a * 3 + 1] : positions[a * 3 + 2];
      const bz = axis === "x" ? positions[b * 3] : axis === "y" ? positions[b * 3 + 1] : positions[b * 3 + 2];
      if (Math.abs(az - depth) > 0.12 || Math.abs(bz - depth) > 0.12) return;
      const [x0, y0] = project(positions[a * 3], positions[a * 3 + 1], positions[a * 3 + 2]);
      const [x1, y1] = project(positions[b * 3], positions[b * 3 + 1], positions[b * 3 + 2]);
      sctx.beginPath();
      sctx.moveTo(x0, y0);
      sctx.lineTo(x1, y1);
      sctx.stroke();
    });
  }

  function apply(activity) {
    if (!activity) return;
    lastActivity = activity;
    glow.fill(0, 0, glow.length);
    const spiked = activity.spiked || [];
    spiked.forEach((i) => {
      if (i >= 0 && i < n) glow[i] = 1;
    });
    edgeGlow.fill(0, 0, edgeGlow.length);
    (activity.fired_edges || []).forEach((i) => {
      if (i >= 0 && i < nEdges) edgeGlow[i] = 1;
    });
    history.push(spiked);
    if (history.length > HIST) history.shift();
    paintRaster();
    paintSlice();
    const bars = activity.readout || {};
    setBar("bar-walk", bars.walk);
    setBar("bar-steer-l", bars.steer_l);
    setBar("bar-steer-r", bars.steer_r);
    setBar("bar-rest", bars.rest);
    setBar("bar-groom", bars.groom);
    setBar("bar-fly", bars.fly);
    const el = (id) => document.getElementById(id);
    if (el("brain-spikes")) el("brain-spikes").textContent = String(activity.total_spikes ?? 0);
    if (el("brain-active")) el("brain-active").textContent = String(activity.active ?? 0);
    if (el("brain-window")) el("brain-window").textContent = `${Math.round(activity.window_ms || 0)} ms`;
    if (el("brain-edges")) el("brain-edges").textContent = String((activity.fired_edges || []).length);
    if (el("brain-readout")) el("brain-readout").textContent = activity.mode || "—";
    if (el("brain-count")) {
      const src = activity.coords_source || layout.coords_source || "atlas";
      el("brain-count").textContent =
        `${src} · ${(activity.n_display || n).toLocaleString()} / ${(activity.n_full || 0).toLocaleString()} cells · ` +
        `${(activity.n_edges || nEdges).toLocaleString()} synapses in view`;
    }
    const list = el("firing-cells");
    if (list) {
      const cells = activity.cells || [];
      list.innerHTML = cells.length
        ? cells
            .map(
              (c) =>
                `<li><span class="nid">${c.id}</span><span class="ncls">${c.class}</span>` +
                `<span class="nreg">${c.region || ""}</span><span class="nspk">${c.spikes}</span></li>`
            )
            .join("")
        : "<li class='empty'>No spikes in the display subset this window.</li>";
    }
    shells.forEach((mesh) => {
      const count = (activity.regions && activity.regions[mesh.name]) || 0;
      const heat = Math.min(1, count / 40);
      mesh.material.opacity = 0.10 + heat * 0.18;
      mesh.material.emissive = new THREE.Color(heat, heat, heat);
      mesh.material.emissiveIntensity = heat * 0.35;
    });
  }

  function setBar(id, value) {
    const node = document.getElementById(id);
    if (!node) return;
    node.style.width = `${Math.max(2, Math.min(100, (value || 0) * 100))}%`;
  }

  document.querySelectorAll("[data-slice-axis]").forEach((btn) => {
    btn.addEventListener("click", () => {
      axis = btn.getAttribute("data-slice-axis");
      document.querySelectorAll("[data-slice-axis]").forEach((b) => b.classList.toggle("active", b === btn));
      applyClip();
    });
  });
  const slider = document.getElementById("slice-depth");
  if (slider) {
    slider.addEventListener("input", () => {
      depth = Number(slider.value);
      applyClip();
    });
  }
  const clipToggle = document.getElementById("slice-clip");
  if (clipToggle) {
    clipToggle.addEventListener("change", () => {
      clipOn = clipToggle.checked;
      applyClip();
    });
  }

  applyClip();

  function frame() {
    for (let i = 0; i < n; i += 1) {
      glow[i] *= 0.86;
      const g = glow[i];
      const t = 0.18 * base[i] + g * 0.82;
      colors[i * 3] = t;
      colors[i * 3 + 1] = t;
      colors[i * 3 + 2] = t;
    }
    geo.attributes.color.needsUpdate = true;
    for (let i = 0; i < nEdges; i += 1) {
      edgeGlow[i] *= 0.82;
      const g = edgeGlow[i];
      const excit = (edgeSign[i] ?? 1) >= 0 ? 1 : 0.45;
      const v = (0.07 + g * 0.9) * excit;
      ecol[i * 6] = v;
      ecol[i * 6 + 1] = v;
      ecol[i * 6 + 2] = v;
      ecol[i * 6 + 3] = v * 0.45;
      ecol[i * 6 + 4] = v * 0.45;
      ecol[i * 6 + 5] = v * 0.45;
    }
    if (nEdges) edgeGeo.attributes.color.needsUpdate = true;
    anatomy.rotation.y = Math.sin(performance.now() / 9000) * 0.06;
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }
  frame();

  return { apply, layout, resize };
}
