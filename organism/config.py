"""Project-wide model policy.

The primary objective is to measure how much autonomous organism-level
behavior can emerge from an embodied model grounded in Drosophila nervous
system anatomy. Looking alive is not a success metric.

`LEGACY_SCAFFOLD` keeps the old bout timers and gait-command overrides
available for comparison. It is off by default. Never re-enable it to
rescue a silent fly.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from enum import Enum

MODEL_VERSION = "0.3.1"

# Birth means: instantiate a NEW persistent digital individual from measured
# anatomy, initialize uncertain physiology explicitly, and let subsequent
# neural / bodily / learned state evolve through experience.
BIRTH_DEFINITION = (
    "Instantiate a new persistent digital individual from measured biological "
    "anatomy, initialize uncertain physiological variables explicitly, and allow "
    "all subsequent neural, bodily and learned state to evolve continuously "
    "through experience. This is not resurrection of the MaleCNS specimen's mind."
)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Old personality timers, walking_drive fallback, and GAIT_COMMANDS-as-decision.
# Comparison only. Default off.
LEGACY_SCAFFOLD = _env_flag("FLY_LEGACY_SCAFFOLD", False)


class MotorMode(str, Enum):
    """How descending/VNC activity becomes joint motion."""

    ENGINEERED_CPG = "MODE_ENGINEERED_CPG"
    HYBRID_VNC = "MODE_HYBRID_VNC"
    NEURAL_MOTOR = "MODE_NEURAL_MOTOR"


@dataclass(frozen=True)
class ModelPolicy:
    """What the organism is allowed to use to produce behavior.

    The CPG leg controller is scaffolding. The decision to engage it is not.
    """

    allow_behavior_timers: bool = False
    allow_motor_fallbacks: bool = False
    allow_named_gait_commands: bool = False
    allow_root_motion: bool = False
    allow_privileged_world_state: bool = False
    allow_pretrained_low_level_gait: bool = True
    name: str = "NO_SCAFFOLD"

    def as_dict(self) -> dict:
        return asdict(self)


NO_SCAFFOLD = ModelPolicy()
LEGACY_POLICY = ModelPolicy(
    allow_behavior_timers=True,
    allow_motor_fallbacks=True,
    allow_named_gait_commands=True,
    allow_root_motion=False,
    allow_privileged_world_state=True,
    allow_pretrained_low_level_gait=True,
    name="LEGACY_SCAFFOLD",
)


def active_policy(*, legacy_scaffold: bool | None = None) -> ModelPolicy:
    if legacy_scaffold is None:
        legacy_scaffold = LEGACY_SCAFFOLD
    return LEGACY_POLICY if legacy_scaffold else NO_SCAFFOLD


# How far descending activity is from muscles.
#   0  handwritten left/right → CPG
#   1  identified descending neurons → CPG          ← current default
#   2  DN → VNC populations → CPG
#   3  VNC → identified motor neurons → muscle groups
#   4  MNs → individual muscle dynamics → FlyBody
MOTOR_FIDELITY_LEVEL = 1


def motor_fidelity_level(mode: MotorMode | str, *, identified_dns: bool = True) -> int:
    if isinstance(mode, str):
        mode = MotorMode(mode)
    if mode is MotorMode.NEURAL_MOTOR:
        return 3
    if mode is MotorMode.HYBRID_VNC:
        return 2
    return 1 if identified_dns else 0


from flybrain.neuron_model import NeuronKind, ParameterProvenance

DEFAULT_MOTOR_MODE = MotorMode.ENGINEERED_CPG
DEFAULT_POLICY = NO_SCAFFOLD

__all__ = [
    "MODEL_VERSION",
    "BIRTH_DEFINITION",
    "LEGACY_SCAFFOLD",
    "ModelPolicy",
    "NO_SCAFFOLD",
    "LEGACY_POLICY",
    "active_policy",
    "MOTOR_FIDELITY_LEVEL",
    "motor_fidelity_level",
    "MotorMode",
    "ParameterProvenance",
    "NeuronKind",
    "DEFAULT_MOTOR_MODE",
    "DEFAULT_POLICY",
]
