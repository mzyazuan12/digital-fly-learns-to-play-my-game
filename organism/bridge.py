"""Identified MaleCNS output pathways → low-level locomotor commands.

Populations are resolved from annotations (type + side + body ID), not from
anonymous index ranges. The NeuroMechFly HybridTurningController still executes
joints; that is pretrained neuromuscular scaffolding, not VNC→MN→muscle.

Literature (not a claim that our LIF model reproduces those recordings):

- DNa02 / DNa01 / DNg13 / DNb05: ipsiversive walking steering;
  R−L DNa02 difference tracks rotational velocity
  (Rayshubskiy et al.; Yang et al. 2023; Feng et al. 2024).
- DNb06: contraversive steering (Yang et al. 2023).
- DNp09 (P9): walking initiation (Bidaye et al. 2020); freeze at strong activation.
- DNg100 (BDN2): walking command onto the VNC rhythm circuit (Pugliese 2025).
- DNb08: rhythmic searching/flailing, not coordinated walking (Pugliese 2025).
- oDN1 / DNg97: bolt-related forward walking (Sapkal et al. 2024).
- MDN: backward walking (Bidaye et al. 2014).
- Halt: Bluebell/DNg60, Brake/AN19A018, Foxglove/CB0890 (Sapkal et al. 2024).
- DNg02: wingbeat / flight-related descending population (Namiki catalogue).

Janelia MaleCNS v1.0 body IDs we resolve when present:

- DNa02  L=523769 R=10360
- DNa01  L=10442  R=10760
- DNg13  L=11074  R=512006
- DNb05  L=10118  R=10065
- DNb06  L=10888  R=11067
- DNp09  L=10783  R=11177
- DNg100 L=10045  R=10056
- DNg97  L=13805  R=230783  (oDN1)
- DNb08  L=12189,12550 R=12044,12075
- DNg60  L=11374  R=188947  (Bluebell)
- MDN    L=11288,12348  R=10763,11332
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np

from flybrain.loader import Connectome
from organism.walking_pathways import WalkingCircuit
from organism.config import (
    LEGACY_SCAFFOLD,
    MOTOR_FIDELITY_LEVEL,
    ModelPolicy,
    ScaffoldViolation,
    active_policy,
    motor_fidelity_level,
)

ENGINEERED_NEURAL_MOTOR_INTERFACE = "ENGINEERED_NEURAL_MOTOR_INTERFACE"
# 12.5 Hz of DNp09 → locomotor_drive 1.0. ASSUMED mapping onto the CPG, not biology.
SPIKE_HZ_TO_DRIVE = 0.08


class NonNeuralMotorAuthority(ScaffoldViolation):
    """NO_SCAFFOLD forbids timers, fallbacks, named gait commands, and developer motor commands."""


# Documented expected body IDs in MaleCNS v1.0 (uint64). Used for provenance,
# not as a hardcoded index map — lookup is still by type/side first.
EXPECTED_BODY_IDS = {
    "DNa02": {"L": (523769,), "R": (10360,)},
    "DNa01": {"L": (10442,), "R": (10760,)},
    "DNg13": {"L": (11074,), "R": (512006,)},
    "DNb05": {"L": (10118,), "R": (10065,)},
    "DNb06": {"L": (10888,), "R": (11067,)},
    "DNp09": {"L": (10783,), "R": (11177,)},
    "DNg100": {"L": (10045,), "R": (10056,)},
    "DNg97": {"L": (13805,), "R": (230783,)},
    "DNb08": {"L": (12189, 12550), "R": (12044, 12075)},
    "DNg60": {"L": (11374,), "R": (188947,)},
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
    locomotor_drive: float = 0.0
    steering_drive: float = 0.0
    analog_walk: float = 0.0
    motor_interface: str = ENGINEERED_NEURAL_MOTOR_INTERFACE


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
    no walking_drive fallback. If identified forward walking DNs are silent,
    the fly rests. That is a scientific result, not a bug to paper over.

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
        self._in_ptr = None
        self._in_order = None
        self.notes: dict = {
            "engineered_gain": True,
            "not_motor_neuron_control": True,
            "legacy_scaffold": self.legacy_scaffold,
            "policy": self.policy.as_dict(),
            "walk_authority": "legacy_bout_timer" if self.legacy_scaffold else "identified_dn_rates",
            "motor_interface": ENGINEERED_NEURAL_MOTOR_INTERFACE,
            "layers": {
                "anatomy": "MEASURED",
                "transmitter": "PREDICTED/MEASURED-DERIVED",
                "functional_gain": "ASSUMED",
                "membrane_model": "ASSUMED/LITERATURE_DERIVED",
                "motor_interface": ENGINEERED_NEURAL_MOTOR_INTERFACE,
            },
        }
        self.walk_trace = 0.0
        self.forward_trace = 0.0
        self.steer_l_trace = 0.0
        self.steer_r_trace = 0.0
        self.reverse_trace = 0.0
        self.circuit = WalkingCircuit(connectome)
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
        self.dng100_indices = self._resolve(
            "walking_dng100",
            ("DNg100",),
            None,
            maps_to="forward_speed",
            literature="Pugliese et al. 2025 / Sapkal et al. 2024: DNg100/BDN2 walking command",
        )
        self.odn1_indices = self._resolve(
            "walking_odn1",
            ("DNg97", "oDN1"),
            None,
            maps_to="forward_speed",
            literature="Sapkal et al. 2024: oDN1/DNg97 bolt-related forward walking",
        )
        self.dnb08_indices = self._resolve(
            "leg_search_dnb08",
            ("DNb08",),
            None,
            maps_to="observe_only",
            literature="Pugliese et al. 2025: DNb08 rhythmic searching, not walking",
        )
        self.halt_bluebell = self._resolve(
            "halt_bluebell",
            ("DNg60",),
            None,
            maps_to="halt_observe",
            literature="Sapkal et al. 2024: Bluebell/DNg60 walk-OFF",
        )
        self.halt_foxglove = self._resolve(
            "halt_foxglove",
            ("CB0890",),
            None,
            maps_to="halt_observe",
            literature="Sapkal et al. 2024 Foxglove/CB0890. May be unlabeled in MaleCNS.",
        )
        self.halt_brake = self._resolve(
            "halt_brake",
            ("AN19A018",),
            None,
            maps_to="halt_observe",
            literature="Sapkal et al. 2024 Brake; FlyBase AN19A018",
        )
        self.cpg_e1 = self._resolve(
            "cpg_E1",
            ("IN17A001",),
            None,
            maps_to="observe_only",
            literature="Pugliese et al. 2025: E1 (IN17A001)",
        )
        self.cpg_e2 = self._resolve(
            "cpg_E2",
            ("INXXX466",),
            None,
            maps_to="observe_only",
            literature="Pugliese et al. 2025: E2 (INXXX466)",
        )
        self.cpg_i1 = self._resolve(
            "cpg_I1",
            ("IN16B036",),
            None,
            maps_to="observe_only",
            literature="Pugliese et al. 2025: I1 (IN16B036)",
        )
        self.cpg_i2 = self._resolve(
            "cpg_I2",
            ("IN19A007",),
            None,
            maps_to="observe_only",
            literature="Pugliese et al. 2025: I2 (IN19A007)",
        )
        self.cpg_e3 = self._resolve(
            "cpg_E3",
            ("IN19B012",),
            None,
            maps_to="observe_only",
            literature="Pugliese et al. 2025: E3 (IN19B012)",
        )
        self.cpg_e4 = self._resolve(
            "cpg_E4",
            ("IN03A006",),
            None,
            maps_to="observe_only",
            literature="Pugliese et al. 2025: E4 (IN03A006) DNb08 relay",
        )
        self.cpg_e5 = self._resolve(
            "cpg_E5",
            ("INXXX464",),
            None,
            maps_to="observe_only",
            literature="Pugliese et al. 2025 published: E5 (INXXX464)",
        )
        self.forward_walk_indices = self.circuit.forward_walk_indices
        if self.forward_walk_indices.size == 0:
            self.forward_walk_indices = self.walk_indices
        self.halt_indices = self.circuit.halt_indices
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
        self.notes["walking_circuit"] = {
            "forward_walk_n": int(self.forward_walk_indices.size),
            "dng100_n": int(self.dng100_indices.size),
            "odn1_n": int(self.odn1_indices.size),
            "cpg_e1_n": int(self.cpg_e1.size),
            "engineered_cpg_still_executes_joints": True,
            "neural_vnc_cpg_drives_joints": False,
        }
        self._rate_cells = {
            f"{typename}_{side}": _side_filter(
                self.connectome, _lookup_type(self.connectome, typename), side
            )
            for typename, side in (
                ("DNp09", "L"),
                ("DNp09", "R"),
                ("DNg100", "L"),
                ("DNg100", "R"),
                ("DNg97", "L"),
                ("DNg97", "R"),
                ("DNb08", "L"),
                ("DNb08", "R"),
                ("DNa01", "L"),
                ("DNa01", "R"),
                ("DNa02", "L"),
                ("DNa02", "R"),
                ("DNg60", "L"),
                ("DNg60", "R"),
                ("MDN", "L"),
                ("MDN", "R"),
            )
        }

    @staticmethod
    def _blank(value: object) -> bool:
        if value is None:
            return True
        if isinstance(value, str) and value.strip().upper() in {"", "NONE"}:
            return True
        return False

    def _reject_non_neural_authority(
        self,
        *,
        walking_bout_s: float,
        walking_drive: float,
        grooming_drive: float,
        flight_drive: float,
        external_command: str,
        developer_override: str,
        privileged_observation: str,
        developer_command,
        direct_walk_fallback,
        privileged_target,
    ) -> None:
        problems: list[str] = []
        if not self.policy.allow_behavior_timers and float(walking_bout_s or 0.0) != 0.0:
            problems.append(f"walking_bout_s={walking_bout_s}")
        if not self.policy.allow_motor_fallbacks:
            if float(walking_drive or 0.0) != 0.0:
                problems.append(f"walking_drive={walking_drive}")
            if float(grooming_drive or 0.0) != 0.0:
                problems.append(f"grooming_drive={grooming_drive}")
            if float(flight_drive or 0.0) != 0.0:
                problems.append(f"flight_drive={flight_drive}")
            if not self._blank(direct_walk_fallback):
                problems.append(f"direct_walk_fallback={direct_walk_fallback}")
        if not self.policy.allow_named_gait_commands:
            cmd = str(external_command or "NONE")
            if cmd.upper() not in {"", "NONE"} and not cmd.startswith("experiment."):
                problems.append(f"gait_command={cmd}")
            if not self._blank(developer_command):
                problems.append(f"developer_command={developer_command}")
        if not self.policy.allow_privileged_world_state:
            if not self._blank(privileged_observation):
                problems.append(f"privileged_observation={privileged_observation}")
            if not self._blank(privileged_target):
                problems.append(f"privileged_target={privileged_target}")
        if not self._blank(developer_override):
            problems.append(f"developer_override={developer_override}")
        if problems:
            raise NonNeuralMotorAuthority(
                "NO_SCAFFOLD rejects non-neural motor authority: " + ", ".join(problems)
            )

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

    def reset_traces(self) -> None:
        self.walk_trace = 0.0
        self.forward_trace = 0.0
        self.steer_l_trace = 0.0
        self.steer_r_trace = 0.0
        self.reverse_trace = 0.0
        self.last_trace = {}
        self.last_command = MotorCommand(0.0, 0.0, "rest", 0.0, 0.0, 0.0, ())

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
        developer_command=None,
        direct_walk_fallback=None,
        privileged_target=None,
        physical_speed_mm_s: float = 0.0,
    ) -> MotorCommand:
        self._reject_non_neural_authority(
            walking_bout_s=walking_bout_s,
            walking_drive=walking_drive,
            grooming_drive=grooming_drive,
            flight_drive=flight_drive,
            external_command=external_command,
            developer_override=developer_override,
            privileged_observation=privileged_observation,
            developer_command=developer_command,
            direct_walk_fallback=direct_walk_fallback,
            privileged_target=privileged_target,
        )

        walk_hz = _group_rate(counts, self.walk_indices, duration_s)
        forward_hz = _group_rate(counts, self.forward_walk_indices, duration_s)
        left_hz = _group_rate(counts, self.steer_left, duration_s)
        right_hz = _group_rate(counts, self.steer_right, duration_s)
        reverse_hz = _group_rate(counts, self.reverse_indices, duration_s)
        flight_hz = _group_rate(counts, self.flight_indices, duration_s)
        groom_hz = _group_rate(counts, self.groom_indices, duration_s)
        left_hz = left_hz + 0.5 * _group_rate(counts, self.contra_right, duration_s)
        right_hz = right_hz + 0.5 * _group_rate(counts, self.contra_left, duration_s)
        analog = graded_output if graded_output is not None else getattr(net, "graded_release", None)
        analog_walk = 0.0
        analog_idx = self.forward_walk_indices if self.forward_walk_indices.size else self.walk_indices
        if analog is not None and analog_idx.size:
            analog_walk = float(np.mean(analog[analog_idx]))

        previous_drive = float(np.clip(SPIKE_HZ_TO_DRIVE * self.forward_trace + analog_walk, 0.0, 1.2))
        dt = max(float(duration_s), 1e-4)
        alpha = float(1.0 - np.exp(-dt / self.RATE_TAU_S))
        self.walk_trace += alpha * (walk_hz - self.walk_trace)
        self.forward_trace += alpha * (forward_hz - self.forward_trace)
        self.steer_l_trace += alpha * (left_hz - self.steer_l_trace)
        self.steer_r_trace += alpha * (right_hz - self.steer_r_trace)
        self.reverse_trace += alpha * (reverse_hz - self.reverse_trace)

        scaffold_used = False
        # ENGINEERED_NEURAL_MOTOR_INTERFACE: continuous decode, not if rate > 18.
        # Forward walking DNs: DNp09 + DNg100 + oDN1/DNg97. Still mapped onto FlyGym CPG.
        locomotor_drive = float(np.clip(SPIKE_HZ_TO_DRIVE * self.forward_trace + analog_walk, 0.0, 1.2))
        reverse_drive = float(np.clip(SPIKE_HZ_TO_DRIVE * self.reverse_trace, 0.0, 0.8))
        speed = locomotor_drive - reverse_drive
        if self.legacy_scaffold and self.policy.allow_behavior_timers and walking_bout_s > 0.05:
            speed = max(float(abs(speed)), float(np.clip(0.85 * walking_drive, 0.0, 1.15))) * (
                -1.0 if speed < 0 else 1.0
            )
            scaffold_used = True
        steering_drive = float(np.clip(SPIKE_HZ_TO_DRIVE * (self.steer_r_trace - self.steer_l_trace), -0.8, 0.8))
        left = float(np.clip(abs(speed) * (1.0 - steering_drive), 0.0, 1.4))
        right = float(np.clip(abs(speed) * (1.0 + steering_drive), 0.0, 1.4))

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
            previous_gate=previous_drive,
            controller_activation=gate,
            motor_fidelity_level=fidelity,
            locomotor_drive=float(locomotor_drive),
            steering_drive=float(steering_drive),
            analog_walk=float(analog_walk),
            motor_interface=ENGINEERED_NEURAL_MOTOR_INTERFACE,
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
            physical_speed_mm_s=physical_speed_mm_s,
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

    def _ensure_incoming(self) -> None:
        if self._in_ptr is not None:
            return
        n = self.connectome.n
        post = np.asarray(self.connectome.post)
        if post.size == 0:
            self._in_ptr = np.zeros(n + 1, dtype=np.int64)
            self._in_order = np.zeros(0, dtype=np.int32)
            return
        order = np.argsort(post, kind="mergesort")
        self._in_order = order.astype(np.int32, copy=False)
        counts = np.bincount(post.astype(np.int64, copy=False), minlength=n)
        ptr = np.empty(n + 1, dtype=np.int64)
        ptr[0] = 0
        np.cumsum(counts, out=ptr[1:])
        self._in_ptr = ptr

    def _incoming_edges(self, post_idx: int) -> np.ndarray:
        self._ensure_incoming()
        start = int(self._in_ptr[int(post_idx)])
        end = int(self._in_ptr[int(post_idx) + 1])
        return self._in_order[start:end]

    def _pre_of_edges(self, edges: np.ndarray) -> np.ndarray:
        if not edges.size:
            return np.zeros(0, dtype=np.int32)
        return (np.searchsorted(self.connectome.pre_ptr, edges, side="right") - 1).astype(np.int32)

    def _node_inputs(self, net, index: int, top_k: int) -> list[dict]:
        edges = self._incoming_edges(index)
        if not edges.size:
            return []
        pres = self._pre_of_edges(edges)
        analog = getattr(net, "graded_release", net.graded_output)
        activity = np.maximum(net.last_spikes.astype(np.float32), analog)
        contrib = activity[pres] * net.weight[edges]
        order = np.argsort(np.abs(contrib))[::-1]
        rows: list[dict] = []
        graded = net.is_graded
        for j in order.tolist():
            amp = float(contrib[j])
            if abs(amp) < 1e-6:
                break
            pre = int(pres[j])
            rows.append(
                {
                    "pre": pre,
                    "pre_label": self.cell_label(pre),
                    "pre_type": str(self.connectome.cell_type[pre] or ""),
                    "pre_superclass": str(self.connectome.superclass[pre] or ""),
                    "post": int(index),
                    "post_label": self.cell_label(int(index)),
                    "contribution": amp,
                    "pre_spiked": bool(net.last_spikes[pre]),
                    "pre_graded": float(analog[pre]),
                    "pre_analog": float(analog[pre]),
                    "pre_drive": float(net.drive[pre]),
                    "pre_is_graded": bool(graded[pre]),
                }
            )
            if len(rows) >= top_k:
                break
        return rows

    def causality_tree(self, net, roots, *, depth: int = 3, top_k: int = 6) -> list[dict]:
        """DNp09 ← real incoming MaleCNS edges ← active neurons ← their inputs."""

        def expand(index: int, remaining: int, seen: frozenset[int]) -> dict:
            analog = getattr(net, "graded_release", net.graded_output)
            node = {
                "index": int(index),
                "label": self.cell_label(int(index)),
                "body_id": int(self.connectome.neuron_ids[int(index)]),
                "v": float(net.v[int(index)]),
                "g": float(net.g[int(index)]),
                "drive": float(net.drive[int(index)]),
                "spiked": bool(net.last_spikes[int(index)]),
                "analog": float(analog[int(index)]),
                "silent": bool(net.silent[int(index)]),
                "inputs": [],
            }
            if remaining <= 0 or int(index) in seen:
                return node
            kids = self._node_inputs(net, int(index), top_k)
            nxt = seen | {int(index)}
            out = []
            for kid in kids:
                child = expand(int(kid["pre"]), remaining - 1, nxt)
                child["contribution"] = kid["contribution"]
                child["pre_is_graded"] = kid["pre_is_graded"]
                out.append(child)
            node["inputs"] = out
            return node

        return [expand(int(i), depth, frozenset()) for i in np.asarray(roots).tolist()]

    def why_walk_active(self, net, top_k: int = 8) -> dict:
        """Presynaptic / modulatory contributors onto DNp09 this window."""
        analog = getattr(net, "graded_release", net.graded_output)
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
                    "graded_output": float(analog[i]),
                    "analog": float(analog[i]),
                    "silent": bool(net.silent[i]),
                }
            )
        afferents: list[dict] = []
        upstream = 0.0
        graded_input = 0.0
        if net is not None:
            self._ensure_walk_afferents()
            if self._walk_pre.size:
                activity = np.maximum(net.last_spikes.astype(np.float32), analog)
                contrib = activity[self._walk_pre] * net.weight[self._walk_edge]
                upstream = float(contrib.sum())
                graded_mask = net.is_graded[self._walk_pre]
                graded_input = float(contrib[graded_mask].sum()) if np.any(graded_mask) else 0.0
                order = np.argsort(np.abs(contrib))[::-1]
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
                            "pre_graded": float(analog[pre]),
                            "pre_analog": float(analog[pre]),
                            "pre_drive": float(net.drive[pre]),
                            "pre_is_graded": bool(net.is_graded[pre]),
                        }
                    )
                    if len(afferents) >= top_k:
                        break
        sources = dict(getattr(net, "drive_sources", {}) or {})
        sensory_input = float(sum(v for k, v in sources.items() if str(k).startswith("sensory.")))
        oa = float(sources.get("neuromod.octopamine.DNp09", 0.0) or 0.0)
        modulation_x = 1.0 + (oa / 14.0 if oa else 0.0)
        tree = self.causality_tree(net, self.walk_indices, depth=3, top_k=6) if net is not None and self.walk_indices.size else []
        return {
            "cells": cells,
            "afferents": afferents,
            "drive_sources": sources,
            "upstream_input": upstream,
            "graded_input": graded_input,
            "sensory_input": sensory_input,
            "modulation_x": modulation_x,
            "causality": tree,
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
        physical_speed_mm_s: float = 0.0,
    ) -> dict:
        rates = self.identified_rates(counts, duration_s)
        t_s = float(getattr(net, "sim_ms", 0.0) or 0.0) / 1000.0
        bout = "NONE" if (not self.policy.allow_behavior_timers or walking_bout_s <= 0.0) else f"{walking_bout_s:.3f} s"
        why = (
            self.why_walk_active(net)
            if net is not None
            else {
                "cells": [],
                "afferents": [],
                "drive_sources": {},
                "upstream_input": 0.0,
                "graded_input": 0.0,
                "sensory_input": 0.0,
                "modulation_x": 1.0,
                "causality": [],
            }
        )
        experiment = external_command if str(external_command).startswith("experiment.") else "NONE"
        gait_command = "NONE" if experiment != "NONE" or self._blank(external_command) else str(external_command)
        result = {
            "t_s": t_s,
            "age_s": t_s,
            "external_command": external_command,
            "behavior_timer": bout,
            "timer_authority": bout,
            "gait_command": gait_command,
            "motor_fallback": "NONE",
            "root_motion": "NONE",
            "developer_override": developer_override,
            "privileged_observation": privileged_observation,
            "experiment": experiment,
            "rates_hz": rates,
            "walking_gate": cmd.walking_gate,
            "previous_gate": cmd.previous_gate,
            "controller_activation": cmd.controller_activation,
            "locomotor_drive": cmd.locomotor_drive,
            "steering_drive": cmd.steering_drive,
            "analog_walk": cmd.analog_walk,
            "cpg_amplitude": float(0.5 * (abs(cmd.left) + abs(cmd.right))),
            "physical_speed_mm_s": float(physical_speed_mm_s),
            "mode": cmd.mode,
            "scaffold_used": cmd.scaffold_used,
            "neural_only": cmd.neural_only,
            "motor_fidelity_level": cmd.motor_fidelity_level,
            "motor_interface": cmd.motor_interface,
            "policy": self.policy.name,
            "no_scaffold": self.policy.name == "NO_SCAFFOLD",
            "n_spike_events": int(getattr(net, "last_n_spike_events", 0) or 0),
            "n_graded_deliveries": int(getattr(net, "last_n_graded_deliveries", 0) or 0),
            "n_graded_considered": int(getattr(net, "last_n_graded_considered", 0) or 0),
            "layers": dict(self.notes.get("layers") or {}),
            "neural_sources": {
                "DNp09_L_hz": float((rates or {}).get("DNp09_L", 0.0)),
                "DNp09_R_hz": float((rates or {}).get("DNp09_R", 0.0)),
                "DNg100_L_hz": float((rates or {}).get("DNg100_L", 0.0)),
                "DNg100_R_hz": float((rates or {}).get("DNg100_R", 0.0)),
                "oDN1_L_hz": float((rates or {}).get("DNg97_L", 0.0)),
                "oDN1_R_hz": float((rates or {}).get("DNg97_R", 0.0)),
                "MDN_L_hz": float((rates or {}).get("MDN_L", 0.0)),
                "MDN_R_hz": float((rates or {}).get("MDN_R", 0.0)),
                "upstream_input": float(why.get("upstream_input", 0.0) or 0.0),
                "graded_input": float(why.get("graded_input", 0.0) or 0.0),
                "sensory_input": float(why.get("sensory_input", 0.0) or 0.0),
                "modulation_x": float(why.get("modulation_x", 1.0) or 1.0),
            },
            "why_dnp09": why,
            "walking_circuit": {
                "forward_walk_n": int(self.forward_walk_indices.size),
                "dng100_n": int(self.dng100_indices.size),
                "odn1_n": int(self.odn1_indices.size),
                "cpg_e1_n": int(self.cpg_e1.size),
                "cpg_e2_n": int(self.cpg_e2.size),
                "cpg_i1_n": int(self.cpg_i1.size),
                "engineered_cpg_still_executes_joints": True,
                "neural_vnc_cpg_drives_joints": False,
            },
        }
        result["text"] = format_walk_trace(result)
        return result


def _authority_mark(value: object) -> str:
    text = "NONE" if MotorBridge._blank(value) else str(value)
    mark = "✓" if text == "NONE" else " "
    return f"{text} {mark}".strip()


def _format_causality_lines(nodes: list, indent: int = 0) -> list[str]:
    lines: list[str] = []
    pad = "  " * indent
    for node in nodes:
        label = node.get("label") or node.get("pre_label") or str(node.get("index", "?"))
        extra = ""
        if "contribution" in node:
            extra = f"  {float(node['contribution']):+.3f}"
        lines.append(f"{pad}{label}{extra}")
        kids = node.get("inputs") or []
        if kids:
            lines.append(f"{pad}↑")
            lines.extend(_format_causality_lines(kids, indent + 1))
    return lines


def format_walk_trace(trace: dict) -> str:
    rates = trace.get("rates_hz") or {}
    neural = trace.get("neural_sources") or {}
    why = trace.get("why_dnp09") or {}
    if not neural:
        neural = {
            "DNp09_L_hz": float(rates.get("DNp09_L", 0.0)),
            "DNp09_R_hz": float(rates.get("DNp09_R", 0.0)),
            "upstream_input": float(why.get("upstream_input", 0.0) or 0.0),
            "graded_input": float(why.get("graded_input", 0.0) or 0.0),
            "sensory_input": float(why.get("sensory_input", 0.0) or 0.0),
            "modulation_x": float(why.get("modulation_x", 1.0) or 1.0),
        }
    width = 49

    def row(left: str, right: str = "") -> str:
        inner = width - 2
        if right:
            gap = inner - len(left) - len(right)
            if gap < 1:
                text = (left + " " + right)[:inner].ljust(inner)
            else:
                text = left + (" " * gap) + right
        else:
            text = left[:inner].ljust(inner)
        return "│" + text + "│"

    title = " WALK INITIATION TRACE "
    dash = "─" * max(1, (width - 2 - len(title)) // 2)
    top = "┌" + (dash + title + dash)[: width - 2].ljust(width - 2, "─") + "┐"
    bot = "└" + ("─" * (width - 2)) + "┘"
    state = str(trace.get("mode", "rest")).upper()
    if state == "WALK":
        state = "WALKING"
    lines = [
        top,
        row("Age", f"{float(trace.get('age_s', trace.get('t_s', 0.0))):.3f} s"),
        row(""),
        row("TIMER AUTHORITY", _authority_mark(trace.get("timer_authority", trace.get("behavior_timer")))),
        row("GAIT COMMAND", _authority_mark(trace.get("gait_command", "NONE"))),
        row("MOTOR FALLBACK", _authority_mark(trace.get("motor_fallback", "NONE"))),
        row("ROOT MOTION", _authority_mark(trace.get("root_motion", "NONE"))),
        row("DEVELOPER OVERRIDE", _authority_mark(trace.get("developer_override"))),
        row(""),
        row("Neural sources"),
        row("DNp09 L", f"{float(neural.get('DNp09_L_hz', 0.0)):.1f} Hz"),
        row("DNp09 R", f"{float(neural.get('DNp09_R_hz', 0.0)):.1f} Hz"),
        row("DNg100 L", f"{float(neural.get('DNg100_L_hz', 0.0)):.1f} Hz"),
        row("DNg100 R", f"{float(neural.get('DNg100_R_hz', 0.0)):.1f} Hz"),
        row("oDN1 L", f"{float(neural.get('oDN1_L_hz', 0.0)):.1f} Hz"),
        row("oDN1 R", f"{float(neural.get('oDN1_R_hz', 0.0)):.1f} Hz"),
        row("MDN L", f"{float(neural.get('MDN_L_hz', 0.0)):.1f} Hz"),
        row("MDN R", f"{float(neural.get('MDN_R_hz', 0.0)):.1f} Hz"),
        row("upstream input", f"{float(neural.get('upstream_input', 0.0)):+.2f}"),
        row("graded input", f"{float(neural.get('graded_input', 0.0)):+.2f}"),
        row("sensory input", f"{float(neural.get('sensory_input', 0.0)):+.2f}"),
        row("modulation", f"×{float(neural.get('modulation_x', 1.0)):.2f}"),
        row(""),
        row("neural locomotor drive", f"{float(trace.get('locomotor_drive', trace.get('walking_gate', 0.0))):.2f}"),
        row("neural steering", f"{float(trace.get('steering_drive', 0.0)):+.2f}"),
        row(""),
        row("CPG amplitude", f"{float(trace.get('cpg_amplitude', trace.get('controller_activation', 0.0))):.2f}"),
        row("physical speed", f"{float(trace.get('physical_speed_mm_s', 0.0)):.1f} mm/s"),
        row(""),
        row(f"STATE: {state}"),
        row(f"interface: {trace.get('motor_interface', ENGINEERED_NEURAL_MOTOR_INTERFACE)}"),
        bot,
    ]
    experiment = trace.get("experiment") or "NONE"
    if not MotorBridge._blank(experiment):
        lines.append(f"EXPERIMENT {experiment}")
    tree = why.get("causality") or []
    if tree:
        lines += ["", "Why did DNp09 become active?", "DNp09", "↑", "real incoming MaleCNS edges"]
        lines.extend(_format_causality_lines(tree, 0))
    return "\n".join(lines) + "\n"
