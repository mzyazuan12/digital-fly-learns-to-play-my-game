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
from enum import Enum

MODEL_VERSION = "0.3.0"

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


from flybrain.neuron_model import NeuronKind, ParameterProvenance

DEFAULT_MOTOR_MODE = MotorMode.ENGINEERED_CPG

__all__ = [
    "MODEL_VERSION",
    "BIRTH_DEFINITION",
    "LEGACY_SCAFFOLD",
    "MotorMode",
    "ParameterProvenance",
    "NeuronKind",
    "DEFAULT_MOTOR_MODE",
]
