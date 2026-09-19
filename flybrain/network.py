"""Sparse mixed dynamics on a fixed MaleCNS topology.

Connectivity is biological. Dynamics are an engineering model.

Weights are factored so learning cannot overwrite anatomy:

    effective_weight =
        anatomical_weight
        * functional_gain               # physiology/model; not learned
        * plastic_factor                # 1 + plastic_component; learned
        * synaptic_effect_sign          # transmitter/effect assumption
        * contact_gain                  # model/calibration

plastic_factor is clipped strictly positive so experience cannot reverse
synaptic_effect_sign. Sign-changing plasticity would be a separate
explicit mechanism, not a negative multiplier.

Two propagation paths. Event-driven spike delivery is not enough:

    SPIKING  membrane → threshold → spike → delayed synaptic events
    GRADED   membrane → analog graded_release → synapses every tick

A graded cell that never sets spiked=True must still influence downstream
cells. Membership is is_graded & ~silent, not a 'live/spiking' mask.
Output amount is graded_transfer(V), not whether the cell was included.

Each tick:

    decay spike kernel / zero this tick's graded current
    deliver due spike events
    compute current graded_release(V)
    deliver current graded_release
    integrate membranes
    detect new spikes
    schedule their future spike events

Graded current is recomputed, never accumulated onto the decaying spike
kernel. Graded output is analog_output / graded_release, not a firing rate.

Injected current is tagged by source so spontaneous activity is inspectable.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from flybrain.loader import Connectome
from flybrain.neuron_model import NeuronKind, NeuronModelTable, ParameterProvenance, assign_neuron_models
from flybrain.neurons import LIFParams, SHIU_LIF_SANITY_MODEL, shiu_coupling

# Learning may scale a synapse, not invert it. Sign stays with synaptic_effect_sign.
MIN_PLASTIC_FACTOR = 0.05
MAX_PLASTIC_FACTOR = 5.0
if MIN_PLASTIC_FACTOR <= 0:
    raise RuntimeError(
        "MIN_PLASTIC_FACTOR must be > 0 so plasticity cannot reverse synaptic_effect_sign"
    )
# mV of depolarization that saturates analog release. ASSUMED squash, not a rate.
GRADED_RELEASE_SCALE_MV = 4.0
GRADED_RELEASE_EPS = 1e-3


class MixedDynamicsNetwork:
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
        # Postsynaptic effect assumption. Not the same as neurotransmitter identity.
        self.synaptic_effect_sign = connectome.sign.astype(np.float32)
        self.edge_sign = self.synaptic_effect_sign
        # Physiological efficacy. Not overwritten by learning.
        self.functional_gain = np.ones(connectome.n_edges, dtype=np.float32)
        # Associative / experience-dependent delta. Anatomical weights stay frozen.
        # plastic_factor = 1 + plastic_component (starts at 1.0).
        self.plastic_component = np.zeros(connectome.n_edges, dtype=np.float32)
        self.is_graded = self.models.graded_mask().astype(bool)
        self.silent = np.zeros(n, dtype=bool)
        self.kind_provenance = self.models.provenance
        self.efficacy = np.ones(connectome.n_edges, dtype=np.float32)
        self._rebuild_weights()
        # Birth scatter. ASSUMED. ±1.5 mV is well below (v_th − v_rest) = 7 mV,
        # so this desynchronizes the network; it is not a walk initiator.
        self.v_init_noise_std = 1.5
        self.v_init_noise_provenance = ParameterProvenance.ASSUMED.value
        self.v = (
            np.float32(p.v_rest)
            + np.float32(self.v_init_noise_std) * self.rng.standard_normal(n).astype(np.float32)
        )
        # Spike kernel decays (tau_g). Graded current is this tick only.
        self.g_spike = np.zeros(n, dtype=np.float32)
        self.g_graded = np.zeros(n, dtype=np.float32)
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
        self.last_n_spike_events = 0
        self.last_n_graded_deliveries = 0
        self.last_n_graded_considered = 0
        # Small membrane noise. ASSUMED; purpose: documented channel noise,
        # not a hidden walk timer. Inspectable via drive_sources.
        self.intrinsic_noise_std = 0.35
        self.intrinsic_noise_provenance = ParameterProvenance.ASSUMED.value

    @property
    def analog_output(self) -> np.ndarray:
        return self.graded_output

    @property
    def graded_release(self) -> np.ndarray:
        return self.graded_output

    @property
    def baseline_physiological_gain(self) -> np.ndarray:
        return self.functional_gain

    @property
    def plastic_factor(self) -> np.ndarray:
        return np.clip(1.0 + self.plastic_component, MIN_PLASTIC_FACTOR, MAX_PLASTIC_FACTOR)

    @property
    def model_id(self) -> str:
        return getattr(self.params, "model_id", SHIU_LIF_SANITY_MODEL)

    def _rebuild_weights(self) -> None:
        if np.any(self.functional_gain < 0):
            raise RuntimeError(
                "functional_gain must be >= 0; it cannot reverse synaptic_effect_sign"
            )
        gain = np.float32(self.params.contact_gain)
        lo = np.float32(MIN_PLASTIC_FACTOR - 1.0)
        hi = np.float32(MAX_PLASTIC_FACTOR - 1.0)
        np.clip(self.plastic_component, lo, hi, out=self.plastic_component)
        plastic_factor = np.clip(
            1.0 + self.plastic_component,
            MIN_PLASTIC_FACTOR,
            MAX_PLASTIC_FACTOR,
        )
        if not np.all(plastic_factor > 0):
            raise RuntimeError("plastic_factor must stay strictly positive")
        self.efficacy = self.functional_gain * plastic_factor.astype(np.float32)
        self.weight = (
            self.anatomical
            * self.functional_gain
            * plastic_factor.astype(np.float32)
            * self.synaptic_effect_sign
            * gain
        )
        nonzero = self.weight != 0
        if np.any(nonzero) and not np.all(
            np.sign(self.weight[nonzero]) == np.sign(self.synaptic_effect_sign[nonzero])
        ):
            raise RuntimeError("plasticity reversed synaptic_effect_sign")

    def reset(self, clear_plasticity: bool = False) -> None:
        p = self.params
        self.v.fill(p.v_rest)
        self.g_spike.fill(0)
        self.g_graded.fill(0)
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
        self.last_n_spike_events = 0
        self.last_n_graded_deliveries = 0
        self.last_n_graded_considered = 0
        if clear_plasticity:
            self.plastic_component.fill(0.0)
            self._rebuild_weights()

    def lesion(self, indices, silent: bool = True) -> None:
        """Silence identified cells. Restoring them (silent=False) is the control.

        Silencing blocks spikes and synaptic emission. It does not freeze the
        membrane at whatever voltage the last bombardment left behind — restore
        would then be a test of recovery from hyperpolarization, not of the
        pathway. Restored cells return to rest.
        """
        idx = np.asarray(indices, dtype=np.int32)
        if not idx.size:
            return
        self.silent[idx] = bool(silent)
        self.v[idx] = self.v_rest
        self.g_spike[idx] = 0
        self.g_graded[idx] = 0
        self.g[idx] = 0
        self.graded_output[idx] = 0
        self.refractory[idx] = 0

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
        silent = self.silent
        is_graded = self.is_graded
        # Graded cells are not gated by refractory / 'currently spiking'.
        spiking_integrable = (~silent) & (~is_graded) & (self.refractory == 0)
        graded_integrable = (~silent) & is_graded
        integrable = spiking_integrable | graded_integrable
        if np.any(self.refractory > 0):
            self.refractory[self.refractory > 0] -= 1

        self.g_spike *= self.alpha_g
        self.g_graded.fill(0)

        due = self.queue[self.cursor % self.queue_len]
        self.last_n_spike_events = len(due)
        if due:
            self._deliver(due)
            self.queue[self.cursor % self.queue_len] = []

        self._deliver_graded(np.flatnonzero(is_graded))
        np.add(self.g_spike, self.g_graded, out=self.g)

        if np.any(silent):
            dead = silent
            self.v[dead] = self.v_rest + (self.v[dead] - self.v_rest) * self.alpha_v
        drive = self.drive
        if self.intrinsic_noise_std > 0:
            noise = (
                np.float32(self.intrinsic_noise_std)
                * self.rng.standard_normal(self.n).astype(np.float32)
            )
            drive = drive + noise
            self.drive_sources["intrinsic.membrane_noise"] = float(self.intrinsic_noise_std)
        # SHIU_LIF_SANITY_MODEL.
        # Current-based LIF expressed entirely in mV/ms.
        # Used for numerical and connectome-integration sanity checks.
        # NOT the Pugliese CPG dynamical model.
        self.v[integrable] = (
            self.v_rest
            + (self.v[integrable] - self.v_rest) * self.alpha_v
            + drive[integrable] * self.one_minus_av
            + self.g[integrable] * self.coupling
        )

        spiked = np.flatnonzero(spiking_integrable & (self.v > self.v_th)).astype(np.int32)
        self.last_spikes.fill(False)
        if spiked.size:
            self.last_spikes[spiked] = True
            self.counts[spiked] += 1
            self.total_spikes += int(spiked.size)
            self.queue[(self.cursor + self.delay_slots) % self.queue_len].extend(
                spiked.tolist()
            )
            self.v[spiked] = self.v_rest
            self.g_spike[spiked] = 0
            self.g_graded[spiked] = 0
            self.g[spiked] = 0
            self.refractory[spiked] = self.ref_steps

        decay = np.float32(math.exp(-p.dt / 20.0))
        self.pre_trace *= decay
        self.post_trace *= decay
        if spiked.size:
            self.pre_trace[spiked] += np.float32(1.0)
            self.post_trace[spiked] += np.float32(1.0)
        analog_cells = np.flatnonzero(graded_integrable)
        if analog_cells.size:
            self.pre_trace[analog_cells] = np.maximum(
                self.pre_trace[analog_cells], self.graded_output[analog_cells]
            )
            self.post_trace[analog_cells] = np.maximum(
                self.post_trace[analog_cells], self.graded_output[analog_cells]
            )
        self.cursor += 1

    def _deliver(self, presynaptic: list[int]) -> None:
        g = self.g_spike
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
        """Analog synaptic output from nonspiking cells, every tick.

        `cells` is the analog population. Do not pass a live/spiking mask —
        a healthy graded cell is evaluated from V whether or not it spiked.
        Lesioned/silent cells are considered, then emit 0.

        graded_release = clip((V − V_rest) / scale, 0, 1). This is not a
        firing rate. Written into g_graded for this tick only.
        """
        cells = np.asarray(cells, dtype=np.int32)
        self.last_n_graded_considered = int(cells.size)
        self.graded_output.fill(0)
        if not cells.size:
            self.last_n_graded_deliveries = 0
            return
        scale = np.float32(GRADED_RELEASE_SCALE_MV)
        analog = np.clip((self.v[cells] - self.v_rest) / scale, 0.0, 1.0).astype(np.float32)
        analog[self.silent[cells]] = 0.0
        self.graded_output[cells] = analog
        active = analog >= np.float32(GRADED_RELEASE_EPS)
        if not np.any(active):
            self.last_n_graded_deliveries = 0
            return
        cells = cells[active]
        analog = analog[active]
        self.last_n_graded_deliveries = int(cells.size)
        g = self.g_graded
        post = self.post
        weight = self.weight
        ptr = self.ptr
        silent = self.silent
        for i, amp in zip(cells.tolist(), analog.tolist()):
            if silent[i]:
                continue
            start = int(ptr[i])
            end = int(ptr[i + 1])
            if start == end:
                continue
            targets = post[start:end]
            alive = ~silent[targets]
            if np.any(alive):
                np.add.at(g, targets[alive], weight[start:end][alive] * np.float32(amp))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            v=self.v,
            g=self.g,
            g_spike=self.g_spike,
            g_graded=self.g_graded,
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
        if "g_spike" in data.files:
            self.g_spike = data["g_spike"]
            self.g_graded = data["g_graded"]
        else:
            self.g_spike = data["g"].copy()
            self.g_graded = np.zeros_like(data["g"])
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


# Historical name. The network is mixed spiking / graded, not LIF-only.
LIFNetwork = MixedDynamicsNetwork


KNOWN_EXCITATORY_NT = frozenset({"acetylcholine"})
KNOWN_INHIBITORY_NT = frozenset({"gaba", "glutamate", "histamine"})


def dataset_validation(
    connectome: Connectome,
    *,
    models: NeuronModelTable | None = None,
    policy_name: str = "NO_SCAFFOLD",
    net: MixedDynamicsNetwork | None = None,
) -> dict:
    """Startup accounting. File size is not the verification."""
    models = models or (net.models if net is not None else assign_neuron_models(connectome))
    kind = models.kind
    n_spiking = int(np.count_nonzero(kind == NeuronKind.SPIKING_LIF.value))
    n_graded = int(np.count_nonzero(kind == NeuronKind.GRADED_RATE.value))
    n_unknown = int(connectome.n) - n_spiking - n_graded
    degrees = np.diff(connectome.pre_ptr).astype(np.int64)
    neuron_nt = np.array([str(x or "").strip().lower() for x in connectome.neurotransmitter], dtype=object)
    known = np.array(
        [(t in KNOWN_EXCITATORY_NT) or (t in KNOWN_INHIBITORY_NT) for t in neuron_nt],
        dtype=bool,
    )
    excitatory = int(np.count_nonzero(connectome.sign > 0))
    inhibitory = int(np.count_nonzero(connectome.sign < 0))
    uncertain = int((~known).astype(np.int64) @ degrees)
    dataset = str(connectome.report.get("dataset_id", "unknown"))
    malecns = "malecns" in dataset
    layers = {
        "anatomy": "MEASURED" if malecns else "SYNTHETIC",
        "transmitter": "PREDICTED/MEASURED-DERIVED" if malecns else "ASSUMED",
        "functional_gain": "ASSUMED",
        "membrane_model": "ASSUMED/LITERATURE_DERIVED",
        "motor_interface": "ENGINEERED_NEURAL_MOTOR_INTERFACE",
        "gap_junctions": "ABSENT",
        "effectome": "ABSENT",
    }
    report = {
        "dataset_id": dataset,
        "neurons": int(connectome.n),
        "directed_edges": int(connectome.n_edges),
        "anatomical_contacts": int(np.asarray(connectome.anatomical).sum(dtype=np.uint64)),
        "spiking_cells": n_spiking,
        "graded_cells": n_graded,
        "unknown_model_type": n_unknown,
        "excitatory_edges": excitatory,
        "inhibitory_edges": inhibitory,
        "uncertain_sign": uncertain,
        "no_scaffold": str(policy_name).upper() == "NO_SCAFFOLD",
        "policy_name": policy_name,
        "layers": layers,
        "functional_gain_is_not_anatomy": True,
        "gap_junctions": "ABSENT",
    }
    report["text"] = format_dataset_validation(report)
    return report


def format_dataset_validation(report: dict) -> str:
    layers = report.get("layers") or {}
    lines = [
        "MaleCNS DATASET VALIDATION",
        "",
        f"neurons:            {int(report.get('neurons', 0)):,}",
        f"directed edges:     {int(report.get('directed_edges', 0)):,}",
        f"anatomical contacts:{int(report.get('anatomical_contacts', 0)):,}",
        f"spiking cells:      {int(report.get('spiking_cells', 0)):,}",
        f"graded cells:       {int(report.get('graded_cells', 0)):,}",
        f"unknown model type: {int(report.get('unknown_model_type', 0)):,}",
        f"excitatory edges:   {int(report.get('excitatory_edges', 0)):,}",
        f"inhibitory edges:   {int(report.get('inhibitory_edges', 0)):,}",
        f"uncertain-sign:     {int(report.get('uncertain_sign', 0)):,}",
        "",
        f"NO_SCAFFOLD: {str(bool(report.get('no_scaffold'))).upper()}",
        "",
        f"anatomy:            {layers.get('anatomy', 'UNKNOWN')}",
        f"transmitter:        {layers.get('transmitter', 'UNKNOWN')}",
        f"functional_gain:    {layers.get('functional_gain', 'ASSUMED')}",
        f"membrane_model:     {layers.get('membrane_model', 'ASSUMED')}",
        f"motor_interface:    {layers.get('motor_interface', 'ENGINEERED_NEURAL_MOTOR_INTERFACE')}",
        f"gap_junctions:      {layers.get('gap_junctions', 'ABSENT')}",
    ]
    return "\n".join(lines) + "\n"
