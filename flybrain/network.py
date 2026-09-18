"""Sparse current-based LIF on a fixed MaleCNS topology.

Event-driven synaptic delivery: only outgoing edges of neurons that spiked
are visited. Subthreshold voltages of all cells still advance each tick.

This is an approximate engineering model. Connectivity is biological;
the dynamics are not a validated fly emulation.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from flybrain.loader import Connectome
from flybrain.neurons import LIFParams, shiu_coupling


class LIFNetwork:
    def __init__(
        self,
        connectome: Connectome,
        params: LIFParams | None = None,
        seed: int = 0,
    ):
        self.connectome = connectome
        self.params = params or LIFParams()
        self.rng = np.random.default_rng(seed)
        n = connectome.n
        p = self.params
        self.n = n
        self.ptr = connectome.pre_ptr
        self.post = connectome.post.astype(np.int32, copy=False)
        self.anatomical = connectome.anatomical.astype(np.float32)
        self.edge_sign = connectome.sign.astype(np.float32)
        self.efficacy = np.ones(connectome.n_edges, dtype=np.float32)
        self._rebuild_weights()
        self.v = np.full(n, p.v_rest, dtype=np.float32)
        self.g = np.zeros(n, dtype=np.float32)
        self.drive = np.zeros(n, dtype=np.float32)
        self.refractory = np.zeros(n, dtype=np.int16)
        self.delay_slots = p.delay_slots
        # Ring length is delay+1 so a spike at t is delivered at t+delay, not now.
        self.queue_len = self.delay_slots + 1
        self.queue: list[list[int]] = [[] for _ in range(self.queue_len)]
        self.cursor = 0
        self.counts = np.zeros(n, dtype=np.int32)
        self.total_spikes = 0
        self.sim_ms = 0.0
        self.alpha_v = np.float32(math.exp(-p.dt / p.tau_m))
        self.alpha_g = np.float32(math.exp(-p.dt / p.tau_g))
        self.coupling = np.float32(shiu_coupling(p))
        self.v_rest = np.float32(p.v_rest)
        self.v_th = np.float32(p.v_threshold)
        self.ref_steps = p.refractory_steps
        self.one_minus_av = np.float32(1.0 - self.alpha_v)
        self.last_spikes = np.zeros(n, dtype=np.bool_)
        self.pre_trace = np.zeros(n, dtype=np.float32)
        self.post_trace = np.zeros(n, dtype=np.float32)

    def _rebuild_weights(self) -> None:
        gain = np.float32(self.params.contact_gain)
        self.weight = self.anatomical * self.efficacy * self.edge_sign * gain

    def reset(self, clear_plasticity: bool = False) -> None:
        p = self.params
        self.v.fill(p.v_rest)
        self.g.fill(0)
        self.drive.fill(0)
        self.refractory.fill(0)
        self.queue = [[] for _ in range(self.queue_len)]
        self.cursor = 0
        self.counts.fill(0)
        self.total_spikes = 0
        self.sim_ms = 0.0
        self.last_spikes.fill(False)
        self.pre_trace.fill(0)
        self.post_trace.fill(0)
        if clear_plasticity:
            self.efficacy.fill(1.0)
            self._rebuild_weights()

    def inject(self, indices, current: float) -> None:
        idx = np.asarray(indices, dtype=np.int32)
        if idx.size:
            self.drive[idx] = np.float32(current)

    def add_drive(self, indices, current: float) -> None:
        """Accumulate current without wiping other receptor/modulatory input."""
        idx = np.unique(np.asarray(indices, dtype=np.int32))
        if idx.size and current != 0.0:
            self.drive[idx] += np.float32(current)

    def clear_drive(self) -> None:
        self.drive.fill(0)

    def step(self, n_steps: int = 1) -> np.ndarray:
        """Advance `n_steps` ticks. Return spike counts over this window."""
        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        self.counts.fill(0)
        for _ in range(n_steps):
            self._tick()
        self.sim_ms += n_steps * self.params.dt
        return self.counts.copy()

    def run_ms(self, duration_ms: float) -> np.ndarray:
        steps = int(round(duration_ms / self.params.dt))
        if steps < 1:
            raise ValueError("Duration too short for dt")
        return self.step(steps)

    def _tick(self) -> None:
        p = self.params
        active = self.refractory == 0
        if np.any(self.refractory > 0):
            self.refractory[self.refractory > 0] -= 1
        self.v[active] = (
            self.v_rest
            + (self.v[active] - self.v_rest) * self.alpha_v
            + self.drive[active] * self.one_minus_av
            + self.g[active] * self.coupling
        )
        self.g *= self.alpha_g
        spiked = np.flatnonzero(active & (self.v > self.v_th)).astype(np.int32)
        self.last_spikes.fill(False)
        if spiked.size:
            self.last_spikes[spiked] = True
            self.counts[spiked] += 1
            self.total_spikes += int(spiked.size)
            self.queue[(self.cursor + self.delay_slots) % self.queue_len].extend(
                spiked.tolist()
            )
        due = self.queue[self.cursor % self.queue_len]
        if due:
            self._deliver(due)
            self.queue[self.cursor % self.queue_len] = []
        if spiked.size:
            self.v[spiked] = self.v_rest
            self.g[spiked] = 0
            self.refractory[spiked] = self.ref_steps
        decay = np.float32(math.exp(-p.dt / 20.0))
        self.pre_trace *= decay
        self.post_trace *= decay
        if spiked.size:
            self.pre_trace[spiked] += np.float32(1.0)
            self.post_trace[spiked] += np.float32(1.0)
        self.cursor += 1

    def _deliver(self, presynaptic: list[int]) -> None:
        g = self.g
        post = self.post
        weight = self.weight
        ptr = self.ptr
        refractory = self.refractory
        for i in presynaptic:
            start = int(ptr[i])
            end = int(ptr[i + 1])
            if start == end:
                continue
            targets = post[start:end]
            w = weight[start:end]
            live = refractory[targets] == 0
            if np.any(live):
                np.add.at(g, targets[live], w[live])

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            v=self.v,
            g=self.g,
            drive=self.drive,
            refractory=self.refractory,
            efficacy=self.efficacy,
            counts=self.counts,
            cursor=np.int64(self.cursor),
            total_spikes=np.int64(self.total_spikes),
            sim_ms=np.float64(self.sim_ms),
            pre_trace=self.pre_trace,
            post_trace=self.post_trace,
            last_spikes=self.last_spikes,
            queue=np.array(self.queue, dtype=object),
            rng_state=np.array([self.rng.bit_generator.state], dtype=object),
            seed_state=np.asarray(self.rng.bit_generator.state["state"]["state"]),
            dt=np.float64(self.params.dt),
        )

    def load(self, path: Path) -> None:
        data = np.load(path, allow_pickle=True)
        self.v = data["v"]
        self.g = data["g"]
        self.drive = data["drive"]
        self.refractory = data["refractory"]
        self.efficacy = data["efficacy"]
        self.counts = data["counts"]
        self.cursor = int(data["cursor"])
        self.total_spikes = int(data["total_spikes"])
        self.sim_ms = float(data["sim_ms"])
        if "pre_trace" in data.files:
            self.pre_trace = data["pre_trace"]
        if "post_trace" in data.files:
            self.post_trace = data["post_trace"]
        if "last_spikes" in data.files:
            self.last_spikes = data["last_spikes"]
        if "queue" in data.files:
            self.queue = [list(slot) for slot in data["queue"]]
        if "rng_state" in data.files:
            self.rng.bit_generator.state = data["rng_state"][0]
        self._rebuild_weights()
