"""Identified MaleCNS output pathways → low-level locomotor commands.

Populations are resolved from annotations (type + side + body ID), not from
anonymous index ranges. The NeuroMechFly HybridTurningController still executes
joints; that is pretrained neuromuscular scaffolding, not VNC→MN→muscle.

Literature (not a claim that our LIF model reproduces those recordings):

- DNa02 / DNa01 / DNg13 / DNb05: ipsiversive walking steering;
  R−L DNa02 difference tracks rotational velocity
  (Rayshubskiy et al.; Yang et al. 2023; Feng et al. 2024).
- DNb06: contraversive steering (Yang et al. 2023).
- DNp09 (P9): walking initiation (Bidaye et al. 2020).
- MDN: backward walking (Bidaye et al. 2014).
- DNg02: wingbeat / flight-related descending population (Namiki catalogue).

Janelia MaleCNS v1.0 body IDs we resolve when present:

- DNa02  L=523769 R=10360
- DNa01  L=10442  R=10760
- DNg13  L=11074  R=512006
- DNb05  L=10118  R=10065
- DNb06  L=10888  R=11067
- DNp09  L=10783  R=11177
- MDN    L=11288,12348  R=10763,11332
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np

from flybrain.loader import Connectome
from organism.config import (
    LEGACY_SCAFFOLD,
    MOTOR_FIDELITY_LEVEL,
    ModelPolicy,
    active_policy,
    motor_fidelity_level,
)


# Documented expected body IDs in MaleCNS v1.0 (uint64). Used for provenance,
# not as a hardcoded index map — lookup is still by type/side first.
EXPECTED_BODY_IDS = {
    "DNa02": {"L": (523769,), "R": (10360,)},
    "DNa01": {"L": (10442,), "R": (10760,)},
    "DNg13": {"L": (11074,), "R": (512006,)},
    "DNb05": {"L": (10118,), "R": (10065,)},
    "DNb06": {"L": (10888,), "R": (11067,)},
    "DNp09": {"L": (10783,), "R": (11177,)},
    "MDN": {"L": (11288, 12348), "R": (10763, 11332)},
}


@dataclass(frozen=True)
class Pathway:
    name: str
    types: tuple[str, ...]
    side: str | None
    literature: str
    maps_to: str
    body_ids: tuple[int, ...] = ()
    indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    resolved_types: tuple[str, ...] = ()


@dataclass
class MotorCommand:
    left: float
    right: float
    mode: str  # rest | walk | reverse | groom | fly
    walk_hz: float
    steer_l_hz: float
    steer_r_hz: float
    pathways_used: tuple[str, ...]
    flight_hz: float = 0.0
    groom_hz: float = 0.0
    scaffold_used: bool = False
    walk_trace: float = 0.0
    neural_only: bool = True
    walking_gate: float = 0.0
    previous_gate: float = 0.0
    controller_activation: float = 0.0
    motor_fidelity_level: int = MOTOR_FIDELITY_LEVEL


def _norm_side(value: object) -> str:
    text = str(value or "").strip().upper()
    if text in {"L", "LEFT", "LHS"}:
        return "L"
    if text in {"R", "RIGHT", "RHS"}:
        return "R"
    return ""


def _lookup_type(connectome: Connectome, typename: str) -> np.ndarray:
    exact = connectome.lookup(type_exact=typename)
    if exact.size:
        return exact
    return connectome.lookup(type_prefix=typename)


def _side_filter(connectome: Connectome, indices: np.ndarray, side: str | None) -> np.ndarray:
    if side is None or indices.size == 0:
        return indices.astype(np.int32)
    keep = [_norm_side(connectome.side[i]) == side for i in indices]
    out = indices[np.array(keep, dtype=bool)]
    return out.astype(np.int32)


def _group_rate(counts: np.ndarray, indices: np.ndarray, duration_s: float) -> float:
    if indices.size == 0 or duration_s <= 0:
        return 0.0
    return float(counts[indices].sum()) / (duration_s * max(1, indices.size))


class MotorBridge:
    """MaleCNS identified DNs → 2-vector command for the walking controller.

    Default path: analog mapping from identified DN rates. No bout timer,
    no walking_drive fallback. If DNp09 is silent, the fly rests. That is
    a scientific result, not a bug to paper over.

    LEGACY_SCAFFOLD restores the old timer overrides for comparison only.
    """

    # Numerical floor for "no descending drive", not a behavior policy.
    REST_EPS = 1e-3
    # Leaky readout of sparse DN spikes. ~80 ms, ASSUMED synaptic/membrane
    # timescale — not a 2–5 s walking-bout scheduler.
    RATE_TAU_S = 0.08

    def __init__(
        self,
        connectome: Connectome,
        *,
        legacy_scaffold: bool | None = None,
        policy: ModelPolicy | None = None,
    ):
        self.connectome = connectome
        self.legacy_scaffold = LEGACY_SCAFFOLD if legacy_scaffold is None else bool(legacy_scaffold)
        self.policy = policy or active_policy(legacy_scaffold=self.legacy_scaffold)
        self.pathways: list[Pathway] = []
        self.last_trace: dict = {}
        self._walk_afferent_ready = False
        self._walk_pre = np.zeros(0, dtype=np.int32)
        self._walk_post = np.zeros(0, dtype=np.int32)
        self._walk_edge = np.zeros(0, dtype=np.int32)
        self.notes: dict = {
            "engineered_gain": True,
            "not_motor_neuron_control": True,
            "legacy_scaffold": self.legacy_scaffold,
            "policy": self.policy.as_dict(),
            "walk_authority": "legacy_bout_timer" if self.legacy_scaffold else "identified_dn_rates",
        }
        self.walk_trace = 0.0
        self.steer_l_trace = 0.0
        self.steer_r_trace = 0.0
        self.reverse_trace = 0.0
        self.walk_indices = self._resolve(
            "walk_initiation",
            ("DNp09",),
            None,
            maps_to="forward_speed",
            literature="Bidaye et al. 2020: DNp09/P9 walking initiation",
        )
        self.steer_left = self._resolve(
            "steer_left",
            ("DNa02", "DNa01", "DNg13", "DNb05"),
            "L",
            maps_to="ipsiversive_left",
            literature="Rayshubskiy/Yang/Feng: DNa02, DNa01, DNg13, DNb05 ipsiversive steering",
        )
        self.steer_right = self._resolve(
            "steer_right",
            ("DNa02", "DNa01", "DNg13", "DNb05"),
            "R",
            maps_to="ipsiversive_right",
            literature="Rayshubskiy/Yang/Feng: right copies predict right turns",
        )
        self.contra_left = self._resolve(
            "contra_left",
            ("DNb06",),
            "L",
            maps_to="contraversive_from_left",
            literature="Yang et al. 2023: DNb06 contraversive steering",
        )
        self.contra_right = self._resolve(
            "contra_right",
            ("DNb06",),
            "R",
            maps_to="contraversive_from_right",
            literature="Yang et al. 2023: DNb06 contraversive steering",
        )
        self.reverse_indices = self._resolve(
            "reverse",
            ("MDN",),
            None,
            maps_to="backward_walk",
            literature="Bidaye et al. 2014: MDN backward walking",
        )
        self.flight_indices = self._resolve(
            "flight",
            ("DNg02", "DNg02_a", "DNg02_b", "DNg02_c"),
            None,
            maps_to="wingbeat_and_takeoff",
            literature="Namiki DN catalogue: DNg02 wing/flight-related",
        )
        self.groom_indices = self._resolve(
            "groom",
            ("aDN1", "aDN", "DNg12", "DNg11"),
            None,
            maps_to="antennal_groom",
            literature="Antennal grooming DNs; joint CPG is an engineered VNC surrogate",
        )
        chemo = connectome.lookup(superclass="cb_sensory")
        contact = connectome.lookup(superclass="vnc_sensory")
        self.chemosensory = chemo.astype(np.int32) if chemo.size else np.zeros(0, np.int32)
        self.contact_indices = contact.astype(np.int32) if contact.size else np.zeros(0, np.int32)
        self.last_command = MotorCommand(0.0, 0.0, "rest", 0.0, 0.0, 0.0, ())
        mapping_path = Path(__file__).resolve().parents[1] / "data" / "motor_bridge_mapping.json"
        if mapping_path.exists():
            self.notes["mapping_file"] = str(mapping_path)
            self.notes["mapping_architecture"] = json.loads(mapping_path.read_text()).get("architecture")
        self.notes["pathways"] = [
            {
                "name": p.name,
                "types": p.types,
                "side": p.side,
                "n": int(p.indices.size),
                "body_ids": [int(connectome.neuron_ids[i]) for i in p.indices[:12]],
                "resolved_types": p.resolved_types,
                "literature": p.literature,
                "maps_to": p.maps_to,
            }
            for p in self.pathways
        ]
        self.notes["fallback"] = self._fallback_note()
        self._rate_cells = {
            f"{typename}_{side}": _side_filter(
                self.connectome, _lookup_type(self.connectome, typename), side
            )
            for typename, side in (
                ("DNp09", "L"),
                ("DNp09", "R"),
                ("DNa01", "L"),
                ("DNa01", "R"),
                ("DNa02", "L"),
                ("DNa02", "R"),
            )
        }

    def _fallback_note(self) -> str:
        if self.walk_indices.size and self.steer_left.size and self.steer_right.size:
            return "identified_types"
        return "superclass_descending_split_lr"

    def _resolve(
        self,
        name: str,
        types: tuple[str, ...],
        side: str | None,
        *,
        maps_to: str,
        literature: str,
    ) -> np.ndarray:
        chunks = []
        found_types = []
        for typename in types:
            idx = _side_filter(self.connectome, _lookup_type(self.connectome, typename), side)
            if idx.size:
                chunks.append(idx)
                found_types.append(typename)
        if chunks:
            indices = np.unique(np.concatenate(chunks)).astype(np.int32)
        elif name in {"walk_initiation", "steer_left", "steer_right"}:
            indices = self._superclass_fallback(side)
            found_types = ("descending_neuron_superclass_fallback",)
        else:
            indices = np.zeros(0, dtype=np.int32)
            found_types = ()
        body_ids = tuple(int(self.connectome.neuron_ids[i]) for i in indices[:16])
        self.pathways.append(
            Pathway(
                name=name,
                types=types,
                side=side,
                literature=literature,
                maps_to=maps_to,
                body_ids=body_ids,
                indices=indices,
                resolved_types=tuple(found_types),
            )
        )
        return indices

    def _superclass_fallback(self, side: str | None) -> np.ndarray:
        dns = self.connectome.lookup(superclass="descending_neuron")
        if dns.size == 0:
            dns = self.connectome.lookup(superclass="descending_neuron_tbc")
        return _side_filter(self.connectome, dns, side)

    def read(
        self,
        counts: np.ndarray,
        duration_s: float,
        walking_drive: float = 0.0,
        walking_bout_s: float = 0.0,
        grooming_drive: float = 0.0,
        flight_drive: float = 0.0,
        graded_output: np.ndarray | None = None,
        net=None,
        *,
        external_command: str = "NONE",
        developer_override: str = "NONE",
        privileged_observation: str = "NONE",
        motor_mode: str = "MODE_ENGINEERED_CPG",
    ) -> MotorCommand:
        if not self.policy.allow_behavior_timers:
            walking_bout_s = 0.0
        if not self.policy.allow_motor_fallbacks:
            walking_drive = 0.0
            grooming_drive = 0.0
            flight_drive = 0.0

        walk_hz = _group_rate(counts, self.walk_indices, duration_s)
        left_hz = _group_rate(counts, self.steer_left, duration_s)
        right_hz = _group_rate(counts, self.steer_right, duration_s)
        reverse_hz = _group_rate(counts, self.reverse_indices, duration_s)
        flight_hz = _group_rate(counts, self.flight_indices, duration_s)
        groom_hz = _group_rate(counts, self.groom_indices, duration_s)
        left_hz = left_hz + 0.5 * _group_rate(counts, self.contra_right, duration_s)
        right_hz = right_hz + 0.5 * _group_rate(counts, self.contra_left, duration_s)
        if graded_output is not None and self.walk_indices.size:
            walk_hz = max(walk_hz, 40.0 * float(np.mean(graded_output[self.walk_indices])))

        previous_gate = float(np.clip(0.08 * self.walk_trace, 0.0, 1.0))
        dt = max(float(duration_s), 1e-4)
        alpha = float(1.0 - np.exp(-dt / self.RATE_TAU_S))
        self.walk_trace += alpha * (walk_hz - self.walk_trace)
        self.steer_l_trace += alpha * (left_hz - self.steer_l_trace)
        self.steer_r_trace += alpha * (right_hz - self.steer_r_trace)
        self.reverse_trace += alpha * (reverse_hz - self.reverse_trace)

        scaffold_used = False
        speed = float(np.clip(0.08 * self.walk_trace, 0.0, 1.2))
        if self.reverse_trace > self.walk_trace and self.reverse_trace > 2.0:
            speed = -float(np.clip(0.15 * self.reverse_trace, 0.0, 0.8))
        elif self.legacy_scaffold and self.policy.allow_behavior_timers and walking_bout_s > 0.05:
            speed = max(float(speed), float(np.clip(0.85 * walking_drive, 0.0, 1.15)))
            scaffold_used = True
        steer = float(np.clip(0.08 * (self.steer_r_trace - self.steer_l_trace), -0.8, 0.8))
        left = float(np.clip(abs(speed) * (1.0 - steer), 0.0, 1.4))
        right = float(np.clip(abs(speed) * (1.0 + steer), 0.0, 1.4))

        want_fly = False
        want_groom = False
        if self.legacy_scaffold and self.policy.allow_motor_fallbacks:
            want_fly = flight_drive > 0.55 and walking_bout_s <= 0.05 and walking_drive < 0.42
            want_groom = grooming_drive > 0.55 and walking_bout_s <= 0.05 and walking_drive < 0.35
            if want_fly or want_groom:
                scaffold_used = True

        if want_fly and abs(speed) < 0.45:
            mode = "fly"
            left, right = 0.0, 0.0
        elif want_groom and abs(speed) < 0.18:
            mode = "groom"
            left, right = 0.0, 0.0
        elif abs(speed) < self.REST_EPS:
            mode = "rest"
            left, right = 0.0, 0.0
        elif speed < 0:
            mode = "reverse"
        else:
            mode = "walk"
        used = tuple(
            p.name for p in self.pathways if p.indices.size and _group_rate(counts, p.indices, duration_s) > 0.2
        )
        gate = float(np.clip(abs(speed), 0.0, 1.0))
        fidelity = motor_fidelity_level(
            motor_mode, identified_dns=self.notes.get("fallback") == "identified_types"
        )
        cmd = MotorCommand(
            left=left,
            right=right,
            mode=mode,
            walk_hz=walk_hz,
            steer_l_hz=left_hz,
            steer_r_hz=right_hz,
            pathways_used=used,
            flight_hz=flight_hz,
            groom_hz=groom_hz,
            scaffold_used=scaffold_used,
            walk_trace=float(self.walk_trace),
            neural_only=not scaffold_used,
            walking_gate=gate,
            previous_gate=previous_gate,
            controller_activation=gate,
            motor_fidelity_level=fidelity,
        )
        self.last_command = cmd
        self.last_trace = self.build_trace(
            cmd,
            counts=counts,
            duration_s=duration_s,
            net=net,
            walking_bout_s=walking_bout_s,
            external_command=external_command,
            developer_override=developer_override,
            privileged_observation=privileged_observation,
        )
        return cmd

    def cell_label(self, index: int) -> str:
        ct = str(self.connectome.cell_type[index] or "") or f"body{int(self.connectome.neuron_ids[index])}"
        side = _norm_side(self.connectome.side[index])
        return f"{ct}_{side}" if side else ct

    def identified_rates(self, counts: np.ndarray, duration_s: float) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, idx in self._rate_cells.items():
            out[name] = _group_rate(counts, idx, duration_s)
        return out

    def _ensure_walk_afferents(self) -> None:
        if self._walk_afferent_ready:
            return
        self._walk_afferent_ready = True
        if not self.walk_indices.size:
            return
        n = self.connectome.n
        walk_set = np.zeros(n, dtype=bool)
        walk_set[self.walk_indices] = True
        mask = walk_set[self.connectome.post]
        if not np.any(mask):
            return
        degrees = np.diff(self.connectome.pre_ptr).astype(np.int32)
        pre = np.repeat(np.arange(n, dtype=np.int32), degrees)
        self._walk_pre = pre[mask]
        self._walk_post = self.connectome.post[mask].astype(np.int32)
        self._walk_edge = np.flatnonzero(mask).astype(np.int32)

    def why_walk_active(self, net, top_k: int = 8) -> dict:
        """Presynaptic / modulatory contributors onto DNp09 this window."""
        cells = []
        for i in self.walk_indices.tolist():
            cells.append(
                {
                    "index": int(i),
                    "label": self.cell_label(i),
                    "body_id": int(self.connectome.neuron_ids[i]),
                    "v": float(net.v[i]),
                    "g": float(net.g[i]),
                    "drive": float(net.drive[i]),
                    "spiked": bool(net.last_spikes[i]),
                    "graded_output": float(net.graded_output[i]),
                    "silent": bool(net.silent[i]),
                }
            )
        afferents: list[dict] = []
        if net is not None:
            self._ensure_walk_afferents()
            if self._walk_pre.size:
                activity = net.last_spikes.astype(np.float32)
                activity = np.maximum(activity, net.graded_output)
                contrib = activity[self._walk_pre] * net.weight[self._walk_edge]
                order = np.argsort(np.abs(contrib))[::-1]
                kept = 0
                for j in order.tolist():
                    amp = float(contrib[j])
                    if abs(amp) < 1e-6:
                        break
                    pre = int(self._walk_pre[j])
                    afferents.append(
                        {
                            "pre": pre,
                            "pre_label": self.cell_label(pre),
                            "pre_type": str(self.connectome.cell_type[pre] or ""),
                            "pre_superclass": str(self.connectome.superclass[pre] or ""),
                            "post": int(self._walk_post[j]),
                            "post_label": self.cell_label(int(self._walk_post[j])),
                            "contribution": amp,
                            "pre_spiked": bool(net.last_spikes[pre]),
                            "pre_graded": float(net.graded_output[pre]),
                            "pre_drive": float(net.drive[pre]),
                        }
                    )
                    kept += 1
                    if kept >= top_k:
                        break
        return {
            "cells": cells,
            "afferents": afferents,
            "drive_sources": dict(getattr(net, "drive_sources", {}) or {}),
        }

    def build_trace(
        self,
        cmd: MotorCommand,
        *,
        counts: np.ndarray,
        duration_s: float,
        net=None,
        walking_bout_s: float = 0.0,
        external_command: str = "NONE",
        developer_override: str = "NONE",
        privileged_observation: str = "NONE",
    ) -> dict:
        rates = self.identified_rates(counts, duration_s)
        t_s = float(getattr(net, "sim_ms", 0.0) or 0.0) / 1000.0
        bout = "NONE" if (not self.policy.allow_behavior_timers or walking_bout_s <= 0.0) else f"{walking_bout_s:.3f} s"
        why = self.why_walk_active(net) if net is not None else {"cells": [], "afferents": [], "drive_sources": {}}
        result = {
            "t_s": t_s,
            "external_command": external_command,
            "behavior_timer": bout,
            "developer_override": developer_override,
            "privileged_observation": privileged_observation,
            "rates_hz": rates,
            "walking_gate": cmd.walking_gate,
            "previous_gate": cmd.previous_gate,
            "controller_activation": cmd.controller_activation,
            "mode": cmd.mode,
            "scaffold_used": cmd.scaffold_used,
            "neural_only": cmd.neural_only,
            "motor_fidelity_level": cmd.motor_fidelity_level,
            "policy": self.policy.name,
            "n_spike_events": int(getattr(net, "last_n_spike_events", 0) or 0),
            "n_graded_deliveries": int(getattr(net, "last_n_graded_deliveries", 0) or 0),
            "why_dnp09": why,
        }
        result["text"] = format_walk_trace(result)
        return result


def format_walk_trace(trace: dict) -> str:
    rates = trace.get("rates_hz") or {}

    def hz(name: str) -> str:
        return f"{float(rates.get(name, 0.0)):6.1f} Hz"

    lines = [
        "WALK INITIATION TRACE",
        "",
        f"t = {float(trace.get('t_s', 0.0)):.3f} s",
        "",
        f"External command:       {trace.get('external_command', 'NONE')}",
        f"Behavior timer:         {trace.get('behavior_timer', 'NONE')}",
        f"Developer override:     {trace.get('developer_override', 'NONE')}",
        f"Privileged observation: {trace.get('privileged_observation', 'NONE')}",
        "",
        f"DNp09_L:       {hz('DNp09_L').strip()}",
        f"DNp09_R:       {hz('DNp09_R').strip()}",
        f"DNa01_L:       {hz('DNa01_L').strip()}",
        f"DNa01_R:       {hz('DNa01_R').strip()}",
        "",
        f"Walking gate:           {float(trace.get('walking_gate', 0.0)):.2f}",
        f"Previous gate:          {float(trace.get('previous_gate', 0.0)):.2f}",
        "",
        f"Controller activation:  {float(trace.get('controller_activation', 0.0)):.2f}",
        f"Motor fidelity:         level {int(trace.get('motor_fidelity_level', 1))}",
        "",
        "Result:",
        str(trace.get("mode", "rest")).upper(),
    ]
    why = trace.get("why_dnp09") or {}
    afferents = why.get("afferents") or []
    if afferents:
        lines += ["", "Why is DNp09 active?"]
        for row in afferents:
            lines.append(f"← {row.get('pre_label') or row.get('pre_type') or row.get('pre')}")
    sources = why.get("drive_sources") or {}
    if sources:
        lines += ["", "Drive sources:"]
        for name in sources:
            lines.append(f"← {name}")
    return "\n".join(lines) + "\n"
