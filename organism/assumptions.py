"""Modeling assumptions that are NOT MaleCNS measurements.

Keep this list next to the code that uses the values. The connectome files
specify neurons, directed edges, synapse counts, annotations, and predicted
transmitters. They do not specify membranes, muscles, photoreceptors, or a
locomotor controller.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Assumption:
    id: str
    statement: str
    used_by: str
    biological: bool = False


ASSUMPTIONS: tuple[Assumption, ...] = (
    Assumption(
        id="lif_dynamics",
        statement="Every MaleCNS cell is a current-based LIF proxy (Shiu/DoomFly constants).",
        used_by="flybrain.network.LIFNetwork",
    ),
    Assumption(
        id="nt_sign",
        statement="Acetylcholine excitatory; GABA/glutamate/histamine inhibitory; modulators default +.",
        used_by="flybrain.neurons.nt_sign",
    ),
    Assumption(
        id="brain_physics_decimation",
        statement="Physics may run faster than the LIF tick; the brain is stepped every N physics ticks.",
        used_by="organism.loop.SensorimotorLoop",
    ),
    Assumption(
        id="geometric_retina",
        statement="Mock / fallback vision is two cosine receptive fields, not a full ommatidial lattice.",
        used_by="organism.sensory",
    ),
    Assumption(
        id="neuromechfly_ommatidia",
        statement="Embodied vision is FlyGym NeuroMechFly add_vision + get_ommatidia_readouts (yellow/pale channels). The MuJoCo scene is FlatGroundWorld, not the Three.js living room.",
        used_by="organism.physics.FlyGymBody.sense",
        biological=True,
    ),
    Assumption(
        id="sensory_current_gain",
        statement="Physical intensities are mapped to nA-like LIF currents with hand-set gains.",
        used_by="organism.sensory.SensorySystem",
    ),
    Assumption(
        id="identified_dn_bridge",
        statement="MotorBridge reads DNp09/DNa02/DNa01/DNg13/DNb05/DNb06/MDN by type+side; gains onto the walking controller are engineered.",
        used_by="organism.bridge.MotorBridge",
        biological=True,
    ),
    Assumption(
        id="walk_rest_threshold",
        statement="Silent identified walk DNs produce rest. This rest/walk switch is a hand-implemented transition, not a VNC circuit.",
        used_by="organism.bridge.MotorBridge",
    ),
    Assumption(
        id="internal_state_modulation",
        statement="Arousal/hunger/fatigue/etc. add tonic current into identified populations; they do not call find_food().",
        used_by="organism.physiology.Physiology",
    ),
    Assumption(
        id="hybrid_turning_controller",
        statement="Leg trajectories come from FlyGym HybridTurningController + NeuroMechFly preprogrammed steps. This is an engineered VNC/muscle surrogate, not recovered motor circuitry.",
        used_by="organism.gait / organism.physics",
    ),
    Assumption(
        id="no_root_motion_walking",
        statement="Embodied/biological mode never writes the MuJoCo free-root pose as a substitute for walking.",
        used_by="organism.gait.ArticulatedFly",
    ),
    Assumption(
        id="mock_unicycle",
        statement="When MuJoCo is unavailable, a unicycle mock body stands in for NeuroMechFly physics.",
        used_by="organism.body.MockBody",
    ),
    Assumption(
        id="no_toy_behavior_wiring",
        statement="The miniature graph has MaleCNS-like type names and random recurrent edges only. No phototaxis or contact-avoidance shortcuts.",
        used_by="organism.toy.miniature_connectome",
    ),
    Assumption(
        id="reward_modulated_eligibility",
        statement="Learning currently scales existing MaleCNS edges; it is not validated fly plasticity.",
        used_by="flybrain.plasticity.RewardModulatedPlasticity",
    ),
    Assumption(
        id="flight_wingbeat_bridge",
        statement="DNg02 / flight_drive switch NeuroMechFly into a standing wingbeat. MuJoCo has no aerodynamics, so this is not takeoff physics.",
        used_by="organism.bridge.MotorBridge / organism.gait.ArticulatedFly",
    ),
)


def as_dict() -> dict[str, str]:
    return {item.id: item.statement for item in ASSUMPTIONS}
