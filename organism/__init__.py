"""Persistent embodied Drosophila: MaleCNS brain, FlyBody body, any world."""

from organism.fly import FlyIdentity, VirtualFly
from organism.provenance import BehaviorSource, StepRecord
from organism.body import Pose, flygym_available
from organism.config import LEGACY_SCAFFOLD, MODEL_VERSION, MotorMode
from organism.birth import birth_fly

__all__ = [
    "VirtualFly",
    "FlyIdentity",
    "BehaviorSource",
    "StepRecord",
    "Pose",
    "flygym_available",
    "LEGACY_SCAFFOLD",
    "MODEL_VERSION",
    "MotorMode",
    "birth_fly",
]
