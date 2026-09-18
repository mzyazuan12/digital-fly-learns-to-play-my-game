"""Sparse current-based dynamics on a fixed MaleCNS topology.

Connectivity is biological. Dynamics are an engineering model.

Weights are factored so learning cannot overwrite anatomy:

    weight = anatomical * functional_gain * (1 + plastic_component) * sign * contact_gain

Many VNC walking premotor neurons are nonspiking in insects. Those cells
use graded/rate output; the rest use current-based LIF (ASSUMED identical
constants unless a class-specific table says otherwise).

Injected current is tagged by source so spontaneous activity is inspectable.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from flybrain.loader import Connectome
from flybrain.neuron_model import NeuronModelTable, ParameterProvenance, assign_neuron_models
from flybrain.neurons import LIFParams, shiu_coupling


class LIFNetwork:
    def __init__(
        self,
        connectome: Connectome,
        params: LIFParams | None = None,
        seed: int = 0,
        models: NeuronModelTable | None = None,
    ):
        self.connectome = connectome
        self.params = params or LIFParams()
        self.rng = np.random.default_rng(seed)
        self.models = models or assign_neuron_models(connectome)
        n = connectome.n
        p = self.params
        self.n = n
        self.ptr = connectome.pre_ptr
        self.post = connectome.post.astype(np.int32, copy=False)
        self.anatomical = connectome.anatomical.astype(np.float32)
        self.edge_sign = connectome.sign.astype(np.float32)
        # Physiological efficacy. Not overwritten by learning.
        self.functional_gain = np.ones(connectome.n_edges, dtype=np.float32)
        # Associative / experience-dependent component. Starts at 0.
        self.plastic_component = np.zeros(connectome.n_edges, dtype=np.float32)
        self.is_graded = self.models.graded_mask().astype(bool)
        self.silent = np.zeros(n, dtype=bool)
        self.kind_provenance = self.models.provenance
        self.efficacy = np.ones(connectome.n_edges, dtype=np.float32)
        self._rebuild_weights()
        self.v = np.full(n, p.v_rest, dtype=np.float32)
        self.g = np.zeros(n, dtype=np.float32)
        self.drive = np.zeros(n, dtype=np.float32)
        self.refractory = np.zeros(n, dtype=np.int16)
        self.delay_slots = p.delay_slots
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
        self.graded_output = np.zeros(n, dtype=np.float32)
        self.drive_sources: dict[str, float] = {}
        # Small membrane noise. ASSUMED; purpose: documented channel noise,
        # not a hidden walk timer. Inspectable via drive_sources.
        self.intrinsic_noise_std = 0.35
        self.intrinsic_noise_provenance = ParameterProvenance.ASSUMED.value

    def _rebuild_weights(self) -> None:
        gain = np.float32(self.params.contact_gain)
        self.efficacy = self.functional_gain * (1.0 + self.plastic_component)
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
        self.graded_output.fill(0)
        self.drive_sources = {}
        if clear_plasticity:
            self.plastic_component.fill(0.0)
            self._rebuild_weights()

    def lesion(self, indices, silent: bool = True) -> None:
        """Silence identified cells. Restoring them (silent=False) is the control."""
        idx = np.asarray(indices, dtype=np.int32)
        if idx.size:
            self.silent[idx] = bool(silent)

    def inject(self, indices, current: float, source: str = "inject") -> None:
        idx = np.asarray(indices, dtype=np.int32)
        if idx.size:
            self.drive[idx] = np.float32(current)
            self.drive_sources[source] = self.drive_sources.get(source, 0.0) + float(current) * int(idx.size)

    def add_drive(self, indices, current: float, source: str = "unspecified") -> None:
        """Accumulate current without wiping other receptor/modulatory input."""
        idx = np.unique(np.asarray(indices, dtype=np.int32))
        if idx.size and current != 0.0:
            self.drive[idx] += np.float32(current)
            self.drive_sources[source] = self.drive_sources.get(source, 0.0) + float(current) * int(idx.size)

    def note_drive_sources(self, sources: dict[str, float]) -> None:
        for key, value in sources.items():
            self.drive_sources[key] = float(value)

    def clear_drive(self) -> None:
        self.drive.fill(0)
        self.drive_sources = {}

    def step(self, n_steps: int = 1) -> np.ndarray:
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
        live = (~self.silent) & (self.refractory == 0)
        if np.any(self.refractory > 0):
            self.refractory[self.refractory > 0] -= 1
        drive = self.drive
        if self.intrinsic_noise_std > 0:
            noise = (
                np.float32(self.intrinsic_noise_std)
                * self.rng.standard_normal(self.n).astype(np.float32)
            )
            drive = drive + noise
            self.drive_sources["intrinsic.membrane_noise"] = float(self.intrinsic_noise_std)
        self.v[live] = (
            self.v_rest
            + (self.v[live] - self.v_rest) * self.alpha_v
            + drive[live] * self.one_minus_av
            + self.g[live] * self.coupling
        )
        self.g *= self.alpha_g
        spiking = live & (~self.is_graded)
        spiked = np.flatnonzero(spiking & (self.v > self.v_th)).astype(np.int32)
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
        graded = np.flatnonzero(live & self.is_graded)
        if graded.size:
            self._deliver_graded(graded)
        else:
            self.graded_output.fill(0)
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
        if graded.size:
            self.pre_trace[graded] = np.maximum(self.pre_trace[graded], self.graded_output[graded])
            self.post_trace[graded] = np.maximum(self.post_trace[graded], self.graded_output[graded])
        self.cursor += 1

    def _deliver(self, presynaptic: list[int]) -> None:
        g = self.g
        post = self.post
        weight = self.weight
        ptr = self.ptr
        refractory = self.refractory
        silent = self.silent
        for i in presynaptic:
            if silent[i]:
                continue
            start = int(ptr[i])
            end = int(ptr[i + 1])
            if start == end:
                continue
            targets = post[start:end]
            w = weight[start:end]
            live = (refractory[targets] == 0) & (~silent[targets])
            if np.any(live):
                np.add.at(g, targets[live], w[live])

    def _deliver_graded(self, cells: np.ndarray) -> None:
        """Analog output from nonspiking cells. LITERATURE_DERIVED for VNC premotor."""
        # Squashing around rest: subthreshold depolarization becomes a rate.
        scale = np.float32(4.0)
        raw = (self.v[cells] - self.v_rest) / scale
        out = 1.0 / (1.0 + np.exp(-raw))
        self.graded_output.fill(0)
        self.graded_output[cells] = out.astype(np.float32)
        g = self.g
        post = self.post
        weight = self.weight
        ptr = self.ptr
        silent = self.silent
        for i, amp in zip(cells.tolist(), out.tolist()):
            if silent[i] or amp < 1e-3:
                continue
            start = int(ptr[i])
            end = int(ptr[i + 1])
            if start == end:
                continue
            targets = post[start:end]
            live = ~silent[targets]
            if np.any(live):
                np.add.at(g, targets[live], weight[start:end][live] * np.float32(amp))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            v=self.v,
            g=self.g,
            drive=self.drive,
            refractory=self.refractory,
            functional_gain=self.functional_gain,
            plastic_component=self.plastic_component,
            efficacy=self.efficacy,
            counts=self.counts,
            cursor=np.int64(self.cursor),
            total_spikes=np.int64(self.total_spikes),
            sim_ms=np.float64(self.sim_ms),
            pre_trace=self.pre_trace,
            post_trace=self.post_trace,
            last_spikes=self.last_spikes,
            silent=self.silent,
            is_graded=self.is_graded,
            graded_output=self.graded_output,
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
        if "functional_gain" in data.files:
            self.functional_gain = data["functional_gain"]
            self.plastic_component = data["plastic_component"]
        else:
            self.functional_gain = data["efficacy"]
            self.plastic_component = np.zeros_like(self.functional_gain)
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
        if "silent" in data.files:
            self.silent = data["silent"].astype(bool)
        if "is_graded" in data.files:
            self.is_graded = data["is_graded"].astype(bool)
        if "graded_output" in data.files:
            self.graded_output = data["graded_output"]
        if "queue" in data.files:
            self.queue = [list(slot) for slot in data["queue"]]
        if "rng_state" in data.files:
            self.rng.bit_generator.state = data["rng_state"][0]
        self._rebuild_weights()
