const hud = {
  status: document.getElementById("status"),
  command: document.getElementById("command"),
  thorax: document.getElementById("thorax"),
  heading: document.getElementById("heading"),
  jointStd: document.getElementById("joint-std"),
  contact: document.getElementById("contact"),
  root: document.getElementById("root"),
  controller: document.getElementById("controller"),
};
const frame = document.getElementById("gait-frame");
const canvas = document.getElementById("joints");
const ctx = canvas.getContext("2d");

function drawJoints(trace) {
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#f4f4f4";
  ctx.font = "18px Inter, Helvetica, Arial, sans-serif";
  ctx.fillText("joint angles (rad) during current bout", 12, 24);
  if (!trace?.length) return;
  const n = trace[0].length;
  let min = Infinity;
  let max = -Infinity;
  for (const row of trace) {
    for (const v of row) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  const span = max - min || 1;
  for (let j = 0; j < n; j += 1) {
    ctx.beginPath();
    for (let i = 0; i < trace.length; i += 1) {
      const x = (i / Math.max(1, trace.length - 1)) * (w - 20) + 10;
      const y = h - 16 - ((trace[i][j] - min) / span) * (h - 48);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = `hsla(0 0% ${28 + (j % 8) * 8}% / 0.9)`;
    ctx.lineWidth = 1.2;
    ctx.stroke();
  }
}

async function pump() {
  try {
    const s = await fetch("/api/gait", { cache: "no-store" }).then((r) => r.json());
    if (!s?.ok) {
      hud.status.textContent = s?.error || "Gait runtime is starting…";
      return;
    }
    hud.status.textContent = s.paused ? "Paused." : `${s.command}. MuJoCo stepping. Brain off.`;
    hud.command.textContent = `${s.command}  L ${Number(s.left).toFixed(2)}  R ${Number(s.right).toFixed(2)}`;
    const t = s.thorax_mm || [0, 0, 0];
    hud.thorax.textContent = `${t[0].toFixed(2)}, ${t[1].toFixed(2)}, ${t[2].toFixed(2)} mm`;
    hud.heading.textContent = `${(((s.heading_rad * 180) / Math.PI) % 360 + 360) % 360 | 0}°`;
    hud.jointStd.textContent = Number(s.joint_std).toFixed(4);
    const c = (s.contact || []).map((v) => (v > 0.5 ? "■" : "·")).join(" ");
    hud.contact.textContent = c || "—";
    hud.root.textContent = s.root_written ? "FAIL" : "none";
    hud.controller.textContent = s.controller_role || s.controller;
    frame.src = `/api/gait/frame?t=${Date.now()}`;
    drawJoints(s.joint_trace || []);
  } catch {
    hud.status.textContent = "Waiting for the NeuroMechFly server…";
  }
}

for (const button of document.querySelectorAll("[data-cmd]")) {
  button.onclick = async () => {
    await fetch("/api/gait/command", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command: button.dataset.cmd }),
    });
    pump();
  };
}

document.getElementById("btn-pause").onclick = async () => {
  const s = await fetch("/api/gait", { cache: "no-store" }).then((r) => r.json());
  await fetch("/api/gait/pause", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ paused: !s.paused }),
  });
};

setInterval(pump, 120);
pump();
