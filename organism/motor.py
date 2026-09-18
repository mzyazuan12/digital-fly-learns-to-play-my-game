"""Low-level locomotor scaffolding.

Identified descending pathways live in `organism.bridge.MotorBridge`.
This module re-exports that bridge. The NeuroMechFly HybridTurningController
still executes joints; that is not VNC → motor neuron → muscle control.
"""

from organism.bridge import MotorBridge, MotorCommand, Pathway

__all__ = ["MotorBridge", "MotorCommand", "Pathway"]
