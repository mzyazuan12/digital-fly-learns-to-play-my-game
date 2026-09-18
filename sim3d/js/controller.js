import * as THREE from "three";

export class BodyController {
  constructor(fly) {
    this.fly = fly;
    this.queue = [];
    this.busy = false;
    this.current = "idle";
    this.speed = 1.35;
    this.onStatus = () => {};
    this.onArrive = () => {};
  }

  enqueue(cmd) {
    this.queue.push(cmd);
  }

  clear() {
    this.queue.length = 0;
  }

  goto(worldPos, label) {
    this.enqueue({ type: "goto", pos: worldPos.clone(), label: label || "move" });
  }

  press(fn, label) {
    this.enqueue({ type: "press", fn, label });
  }

  wait(ms, label = "wait") {
    this.enqueue({ type: "wait", ms, label });
  }

  update(dt) {
    if (this.busy || !this.queue.length) {
      if (!this.queue.length && !this.busy && this.current !== "idle") {
        this.current = "idle";
        this.onStatus(this.current);
      }
      return;
    }
    const cmd = this.queue.shift();
    this.current = cmd.label || cmd.type;
    this.onStatus(this.current);
    if (cmd.type === "goto") this._goto(cmd.pos, cmd.label);
    else if (cmd.type === "press") this._press(cmd.fn, cmd.label);
    else if (cmd.type === "wait") this._wait(cmd.ms);
  }

  _goto(target, label) {
    this.busy = true;
    this.fly.setFlying(true);
    this._anim = { kind: "goto", target, label };
  }

  tickMove(dt) {
    if (!this._anim) return;
    if (this._anim.kind === "goto") {
      const root = this.fly.root;
      const t = this._anim.target;
      const hover = t.clone();
      hover.y += 0.12;
      root.position.lerp(hover, 1 - Math.exp(-this.speed * 8 * dt));
      this.fly.lookToward(hover);
      if (root.position.distanceTo(hover) < 0.04) {
        this._anim = null;
        this.busy = false;
        this.onArrive(this.current);
      }
    }
  }

  async _press(fn, label) {
    this.busy = true;
    this.fly.setFlying(false);
    const root = this.fly.root;
    const down = root.position.clone();
    down.y -= 0.03;
    const start = root.position.clone();
    const t0 = performance.now();
    await new Promise((resolve) => {
      const step = () => {
        const k = Math.min(1, (performance.now() - t0) / 140);
        root.position.lerpVectors(start, down, k);
        if (k < 1) requestAnimationFrame(step);
        else resolve();
      };
      step();
    });
    if (fn) await fn();
    const t1 = performance.now();
    const mid = root.position.clone();
    await new Promise((resolve) => {
      const step = () => {
        const k = Math.min(1, (performance.now() - t1) / 160);
        root.position.lerpVectors(mid, start, k);
        if (k < 1) requestAnimationFrame(step);
        else resolve();
      };
      step();
    });
    this.fly.setFlying(true);
    this.busy = false;
    this.onArrive(label);
  }

  _wait(ms) {
    this.busy = true;
    setTimeout(() => {
      this.busy = false;
    }, ms);
  }
}
