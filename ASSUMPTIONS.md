# Modeling assumptions vs biological measurements

The MaleCNS v1.0 Feather files specify:

- which neurons exist (body IDs)
- which directed connections exist, and their synapse **counts**
- annotations (type, class, superclass, side, …)
- predicted neurotransmitters

They do **not** specify membrane time constants, synaptic receptors, delays,
plasticity rules, photoreceptors, muscles, or a task. They do not contain
the live physiological state of the chemically fixed specimen. Everything
below is a declared model. Code that uses these values points at
`organism/assumptions.py`.

Birth, in this project, means: instantiate a **new** persistent digital
individual from measured anatomy and let its state evolve. It does not mean
resurrection of the original fly's mind.

## What is biological in this repository

- MaleCNS v1.0 retained graph (166,700 neurons, 25,582,938 directed edges)
- FlyBody / NeuroMechFly articulated anatomy (when FlyGym is installed)
- MuJoCo rigid-body physics of that articulated model
- Identified descending types (DNp09, DNa02, …) resolved from annotations
- Annotated motor neurons, where present, plus published muscle-target maps
  (incomplete; FANC→MaleCNS transfers are cross-sex inference)

## Neuron dynamics (not in the connectome)

Default unlabeled cells: coarse current-based LIF (Shiu / DoomFly constants),
labeled **ASSUMED**.

| symbol | value | role |
| --- | --- | --- |
| `v_rest` | −52 mV | leak reversal |
| `v_threshold` | −45 mV | spike threshold |
| `tau_m` | 20 ms | membrane |
| `tau_g` | 5 ms | synaptic current |
| `t_ref` | 2.2 ms | refractory |
| `delay` | 1.8 ms | axonal delay |
| `contact_gain` | 0.275 mV / contact | synaptic scale |
| default `dt` | 0.1 ms | milestone-1 kernel |
| organism `dt` | 1.0 ms | closed-loop speed, same topology |

VNC premotor / local interneuron labels use **analog graded_release**
(**LITERATURE_DERIVED**: many insect walking premotor neurons are
nonspiking). That output is not a firing rate. Graded cells emit every
integration step from `V − V_rest`, whether or not anyone spiked.

Small inspectable membrane noise (std 0.35) is **ASSUMED**. It is not fake
visual firing and not a hidden walk timer. Drive sources are logged.

`functional_gain` initializes to 1. That does **not** mean physiological
weight = anatomical synapse count. Conversion from contact count to
postsynaptic effect is **ASSUMED**. Telemetry layers:

```text
anatomy:            MEASURED
transmitter:        PREDICTED/MEASURED-DERIVED
functional_gain:    ASSUMED
membrane_model:     ASSUMED/LITERATURE_DERIVED
motor_interface:    ENGINEERED_NEURAL_MOTOR_INTERFACE
```

## Synapse sign (policy, not receptors)

- acetylcholine → excitatory
- GABA, glutamate, histamine → inhibitory (insect central-synapse convention)
- unclear / modulators / missing → excitatory default

There are hundreds of thousands of `unclear` predictions. Treating them as +
is a choice, copied from DoomFly, not a measurement.

## Connectome vs effectome

```text
weight = anatomical_count × functional_gain × plastic_factor × synaptic_effect_sign × contact_gain
```

Anatomical counts are frozen. Learning writes `plastic_component` only, and
only on identified KC→MBON edges by default. `plastic_factor` is clipped to
`[0.05, 5]` so `1 + plastic_component` cannot reverse `synaptic_effect_sign`.
Neurotransmitter identity and postsynaptic effect are not the same concept;
the multiplier is the effect sign. Functional gain is physiological
efficacy, not a place to hide a behavior scheduler. `functional_gain = 1`
at birth is an initialization, not a claim that the effectome equals the
connectome. Graded current is this tick's `graded_release(V)` written into
`g_graded`; it does not ride the decaying spike kernel.

## Graph inclusion

Same retained-graph policy as DoomFly:

- keep annotations with an assigned superclass
- drop explicit `Glia`
- keep every released edge between retained bodies, including autapses
- no extra weight cutoff

Published accounting: **166,700** retained neurons, **25,582,938** directed
edges, **124,177,617** synaptic contacts. The paper’s 166,691 figure uses a
slightly different inclusion convention.

## Body and motor

Default motor mode is `MODE_ENGINEERED_CPG`:

```text
MaleCNS identified DNs (rates)
       → analog 2-vector
       → FlyGym HybridTurningController
       → FlyBody / NeuroMechFly position actuators
       → MuJoCo
```

This is **not** descending neuron → VNC CPG → motor neuron → muscle. The
CPG is pretrained scaffolding. Rest vs walk follows a continuous
`locomotor_drive` decoded from identified walking DNs (DNp09, DNg100,
DNg97/oDN1), labeled `ENGINEERED_NEURAL_MOTOR_INTERFACE`. DNb08 is
traced but not decoded as walking. The published VNC CPG types
(E1=IN17A001, E2=INXXX466, I1=IN16B036, I2=IN19B007, E3=IN19B012,
E4=IN03A006, E5=INXXX464) are resolved from **each bodyId's LegNp
PreSyn+PostSyn counts**, not type-level ROI totals and not soma-Z.
IN19A007 exists in MaleCNS but is **not** I2. Type explorer pages pool
T1+T2+T3. DNg100 itself is two descending neurons, resolved from MaleCNS
`type == DNg100`. Pugliese `DNg100_Stim` injects **MANC T1 matrix index
31 / MANC body 10093** (type DNg100). In that MANC table, body 10056 is
`vMS16`. The same integer can appear independently in MaleCNS as a
different cell; do not collapse the namespaces.
Those names tag cells for measurement/lesion; they never mean
`if DNg100: walk()`. `MODE_NEURAL_CPG` records E1/E2/I1 timing and does
not move joints yet. There is **no** `walking_bout_s`
override on the default path. NO_SCAFFOLD **rejects** a timer, motor
fallback, named gait command, or developer motor command rather than
silently ignoring it. Standing is what happens when those DNs are
silent, not `GAIT_COMMANDS["stand"]`.

`MODE_HYBRID_VNC` logs motor-neuron activity beside the CPG.
`MODE_NEURAL_CPG` is the research path where phase should come from the
identified VNC motif. `MODE_NEURAL_MOTOR` is reserved until muscle
actuation exists.

A `MotorNeuronMuscleMap` records, for every mapped motor neuron: MaleCNS
body ID, MANC type, MN type, side, body part, muscle, joint/action, source,
confidence, and whether the mapping is direct or inferred. Female FANC
muscle targets transferred onto MaleCNS/MANC are labeled
`inferred_cross_sex`.

When FlyGym is missing, a unicycle `MockBody` stands in. Mock trajectories
are not FlyBody results.

The `/gait` page still uses `GAIT_COMMANDS` as a physics-lab keyboard. That
is not the organism's decision source.

## Sensory system (scaffolding)

The brain is not given `target_position`, `keyboard_key`, `correct_answer`,
or world object coordinates.

Vision is two cosine receptive fields, or FlyGym ommatidia when
`add_vision()` succeeds. Intensities become currents with hand-set gains.
Proprioception and contact are joint/contact readings mapped onto annotated
sensory superclasses. Antenna / olfaction / gustation interfaces exist.

Population assignment is a stable engineered interface stored with the
individual. It is not a claim that those cells are a phototaxis circuit.

The miniature graph used in CI labels cells with MaleCNS-like types
(DNp09, DNa02, KC, MBON, MN, …) so bridges resolve the same pathways.
Extra edges are mostly random. It is **not** a phototaxis controller.

## Neuromodulation

Internal variables (arousal, hunger, fatigue) are allowed. They update
dopamine, octopamine, and serotonin. Modulators change identified circuits.
They do **not** call `walk()` or `find_food()`.

Octopamine adds modest current to DNp09 (locomotion literature; numeric
gain **ASSUMED**). If DNp09 does not spike, the fly rests.

Bout timers (`walking_bout_s`, random groom/flight) exist only behind
`FLY_LEGACY_SCAFFOLD=1` for comparison.

## Plasticity

Default: mushroom-body associative rule on **KC→MBON** edges, using
dopamine / DAN activity. Anatomical counts stay frozen.

The older global reward-modulated eligibility rule still exists in
`flybrain.plasticity` and is not the organism default.

Weights are **not** reset when the fly changes worlds or tasks.

## Shiritori (one environment, not a brain API)

The keyboard BCI in `flybrain/populations.py` and `shiritori/action_decoder.py`
is an older engineered interface for the letter-grid experiment. It must not
be called from `VirtualFly`. The intended embodied pipeline is screen pixels
→ eyes → MaleCNS → body → physical key. That pipeline is not complete.

## Consciousness

No internal variable is labeled consciousness. Measurable properties
(activity, recurrence, memory, state-dependent behavior) may be logged.
Whether those amount to subjective experience is outside what this
simulation can establish.

## What would count as a result

A behavior is neural/autonomous only if:

1. removing the relevant neural pathway disrupts it
2. no behavior timer directly invokes it
3. no developer command directly selects it
4. neural activity causally precedes the motor output
5. sensory perturbations alter it through the nervous system
6. provenance identifies the biological/model pathway

Immediate experiment: DNp09 current initiates walking via the CPG;
silencing DNp09 abolishes it; restoring DNp09 restores it. Next
experiment: sustained DNg100 current on the real MaleCNS graph, recording
per-leg E1/E2/I1/I2 and leg motor neurons. If the downstream trace is a
tonic plateau rather than a 7–15 Hz rhythm, that is a dynamics result —
do not change the graph. If spontaneous walking does not emerge without a
timer, record that and look for the missing physiological mechanism. Do
not add another timer.
