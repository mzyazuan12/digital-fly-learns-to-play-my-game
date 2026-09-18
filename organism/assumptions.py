"""Modeling assumptions that are NOT MaleCNS measurements.

The connectome files specify neurons, directed edges, synapse counts,
annotations, and predicted transmitters. They do not specify membranes,
muscles, photoreceptors, a locomotor controller, or consciousness.
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
        id="objective",
        statement="Success is whether silencing a biological pathway removes a behavior, not whether the fly looks alive.",
        used_by="organism.config",
    ),
    Assumption(
        id="birth_not_resurrection",
        statement="Birth creates a new digital individual from MaleCNS anatomy. The EM specimen's live physiological/subjective state is not in the files and is not reconstructed.",
        used_by="organism.birth",
        biological=True,
    ),
    Assumption(
        id="lif_dynamics",
        statement="Unlabeled cells are current-based LIF proxies (Shiu/DoomFly constants). This is ASSUMED, not a MaleCNS measurement.",
        used_by="flybrain.network.MixedDynamicsNetwork",
    ),
    Assumption(
        id="graded_vnc_premotor",
        statement="VNC premotor / local interneuron labels use graded/rate dynamics because many insect walking premotor neurons are nonspiking (literature). Not measured per MaleCNS cell.",
        used_by="flybrain.neuron_model",
        biological=True,
    ),
    Assumption(
        id="nt_sign",
        statement="Acetylcholine excitatory; GABA/glutamate/histamine inhibitory; modulators default +.",
        used_by="flybrain.neurons.nt_sign",
    ),
    Assumption(
        id="graded_analog_every_tick",
        statement="Nonspiking cells emit analog graded_release every integration step from V−V_rest, independent of a live/spiking mask. Graded current is recomputed into g_graded each tick and does not accumulate on the decaying spike kernel. That output is not a firing rate.",
        used_by="flybrain.network.MixedDynamicsNetwork._deliver_graded",
        biological=True,
    ),
    Assumption(
        id="plastic_factor_bounds",
        statement="plastic_factor = clip(1+plastic_component, 0.05, 5.0). Learning cannot reverse synaptic_effect_sign. Anatomical counts stay frozen.",
        used_by="flybrain.network.MixedDynamicsNetwork._rebuild_weights",
        biological=True,
    ),
    Assumption(
        id="weight_factorization",
        statement="Anatomical synapse counts stay frozen. Functional gain is physiological efficacy. Plastic component is learned. synaptic_effect_sign is a transmitter/effect assumption, not overwritten by 1+plastic_component going negative.",
        used_by="flybrain.network.MixedDynamicsNetwork",
        biological=True,
    ),
    Assumption(
        id="brain_physics_decimation",
        statement="Physics may run faster than the neural tick; the brain is stepped every N physics ticks.",
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
        statement="Physical intensities are mapped to nA-like currents with hand-set gains.",
        used_by="organism.sensory.SensorySystem",
    ),
    Assumption(
        id="engineered_neural_motor_interface",
        statement="MotorBridge maps identified DN activity to a continuous locomotor_drive and steering_drive (ENGINEERED_NEURAL_MOTOR_INTERFACE). This is not a 18 Hz walk threshold and not a bout timer. NO_SCAFFOLD raises if a timer, fallback, named gait command, or developer motor command is supplied.",
        used_by="organism.bridge.MotorBridge",
        biological=True,
    ),
    Assumption(
        id="no_bout_timer_default",
        statement="Default path has no walking_bout_s, no walking_drive fallback, and no random groom/flight timers. Those exist only behind LEGACY_SCAFFOLD.",
        used_by="organism.bridge.MotorBridge / organism.physiology",
    ),
    Assumption(
        id="leaky_dn_readout",
        statement="DN rates are leaky-integrated with tau≈80 ms (ASSUMED). This estimates sparse spikes; it is not a 2–5 s behavior bout.",
        used_by="organism.bridge.MotorBridge",
    ),
    Assumption(
        id="neuromodulation",
        statement="Hunger/arousal update dopamine/octopamine/serotonin. Modulators change identified circuits. They do not call walk() or find_food().",
        used_by="organism.neuromodulation",
        biological=True,
    ),
    Assumption(
        id="octopamine_dnp09_gain",
        statement="Octopamine adds modest current to DNp09. Literature links OA to insect locomotion; the numeric gain is ASSUMED. If DNp09 does not spike, the fly rests.",
        used_by="organism.neuromodulation.Neuromodulation",
    ),
    Assumption(
        id="vnc_walking_cpg_traced_not_executed",
        statement="MaleCNS contains DNg100, DNb08, DNg97/oDN1, and the published E1/E2/I1 walking CPG types (Pugliese 2025). Their anatomy is traced. Joint trajectories still come from FlyGym HybridTurningController until MODE_NEURAL_MOTOR exists. Standing is silent locomotor DNs, not a stand() command.",
        used_by="organism.walking_pathways / organism.bridge.MotorBridge",
        biological=True,
    ),
    Assumption(
        id="hybrid_turning_controller",
        statement="In MODE_ENGINEERED_CPG, leg trajectories come from FlyGym HybridTurningController. Engineered VNC/muscle surrogate, not recovered motor circuitry.",
        used_by="organism.gait / organism.physics",
    ),
    Assumption(
        id="motor_neuron_map_incomplete",
        statement="MotorNeuronMuscleMap uses MANC/FANC annotations and published muscle targets. Cross-sex FANC→MaleCNS transfers are INFERRED, not measured from the MaleCNS specimen.",
        used_by="organism.motor_map",
        biological=True,
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
        statement="The miniature graph has MaleCNS-like type names. Extra edges are random except documented DN recurrence, KC→MBON, chemo→KC, and DNp09→VNC premotor.",
        used_by="organism.toy.miniature_connectome",
    ),
    Assumption(
        id="reward_modulated_eligibility",
        statement="A global reward-modulated rule on every MaleCNS edge still exists for comparison; the organism default is mushroom-body KC→MBON plasticity.",
        used_by="flybrain.plasticity.RewardModulatedPlasticity",
    ),
    Assumption(
        id="mushroom_body_plasticity",
        statement="Default learning is KC→MBON, dopamine-modulated. Global reward plasticity on every synapse is not used.",
        used_by="flybrain.mushroom_body.MushroomBodyLearning",
        biological=True,
    ),
    Assumption(
        id="intrinsic_noise",
        statement="Small membrane noise (std 0.35, ASSUMED) is an inspectable drive source. It is not a hidden walk timer and is not fake visual firing.",
        used_by="flybrain.network.MixedDynamicsNetwork",
    ),
    Assumption(
        id="flight_wingbeat_bridge",
        statement="Flight/groom modes are not selected by timers in the default path. The gait lab may still run engineered wingbeat/groom CPGs.",
        used_by="organism.bridge.MotorBridge / organism.gait.ArticulatedFly",
    ),
    Assumption(
        id="no_consciousness_variable",
        statement="No internal variable is labeled consciousness. Whether neural dynamics amount to subjective experience is outside what this simulation can establish.",
        used_by="organism.fly / organism.birth",
    ),
)


def as_dict() -> dict[str, str]:
    return {item.id: item.statement for item in ASSUMPTIONS}
