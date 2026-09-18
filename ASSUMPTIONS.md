# Modeling assumptions vs biological measurements

The MaleCNS v1.0 Feather files specify:

- which neurons exist (body IDs)
- which directed connections exist, and their synapse **counts**
- annotations (type, class, superclass, side, …)
- predicted neurotransmitters

They do **not** specify membrane time constants, synaptic receptors, delays,
plasticity rules, photoreceptors, muscles, or a task. Everything below is a
declared model. Code that uses these values points at `organism/assumptions.py`.

## What is biological in this repository

- MaleCNS v1.0 retained graph (166,700 neurons, 25,582,938 directed edges)
- FlyBody / NeuroMechFly articulated anatomy (when FlyGym is installed)
- MuJoCo rigid-body physics of that articulated model

## Neuron dynamics (not in the connectome)

Coarse current-based LIF, matching the publicly documented Shiu / DoomFly
constants so the kernel is reproducible:

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

Visual cells in the real fly often use graded transmission. Here every cell
is a spiking LIF proxy.

## Synapse sign (policy, not receptors)

- acetylcholine → excitatory
- GABA, glutamate, histamine → inhibitory (insect central-synapse convention)
- unclear / modulators / missing → excitatory default

There are hundreds of thousands of `unclear` predictions. Treating them as +
is a choice, copied from DoomFly, not a measurement.

## Graph inclusion

Same retained-graph policy as DoomFly:

- keep annotations with an assigned superclass
- drop explicit `Glia`
- keep every released edge between retained bodies, including autapses
- no extra weight cutoff

Published accounting: **166,700** retained neurons, **25,582,938** directed
edges, **124,177,617** synaptic contacts. The paper’s 166,691 figure uses a
slightly different inclusion convention.

## Body and motor (scaffolding)

Initial locomotion is:

```text
MaleCNS → left/right descending rates → 2-vector command
       → FlyGym HybridTurningController
       → FlyBody / NeuroMechFly position actuators
       → MuJoCo
```

This is **not** descending neuron → VNC → motor neuron → muscle. The CPG
and preprogrammed steps are pretrained scaffolding. Rest vs walk is a
hand-implemented threshold on identified DN rates (DNp09 and steering DNs
resolved from annotations). Each step is logged as
`biological_connectome`, `pretrained_locomotion_controller`,
`hand_implemented_transition`, `learned_plasticity`, or
`developer_override`.

When FlyGym is missing, a unicycle `MockBody` stands in. Mock trajectories
are not FlyBody results.

A later replacement path, without rewriting the organism:

```text
MaleCNS → DNs → VNC → motor neurons → muscle models → joints
```

## Sensory system (scaffolding)

The brain is not given `target_position`, `keyboard_key`, or `correct_answer`.

First-goal vision is two cosine receptive fields (or FlyGym ommatidia when
`add_vision()` succeeds). Intensities become LIF currents with hand-set gains.
Proprioception and contact are joint/contact readings mapped onto annotated
sensory superclasses. Antenna / olfaction / gustation interfaces exist and
are mostly quiet until those physics are added.

Population assignment (which annotated cells are “left eye”) is a stable
engineered interface stored with the individual. It is not a claim that
those cells are a phototaxis circuit.

The miniature graph used in CI labels cells with MaleCNS-like types
(DNp09, DNa02, …) so `MotorBridge` can resolve the same pathways. Extra
edges are random. It is **not** a phototaxis or collision-avoidance
controller. Real MaleCNS phototaxis is not assumed.

Internal physiological variables (arousal, hunger, fatigue, walking bouts)
add tonic current into identified populations. They do not select actions.

## Plasticity (experimental)

Reward-modulated eligibility traces scale **existing** MaleCNS edges.
Anatomical counts stay frozen. No new synapses are created.

Not implemented, only reserved: STDP-like traces as the sole rule,
identified dopaminergic credit assignment, homeostatic set-points.

DoomFly is explicit that its plasticity experiments have **not** demonstrated
validated learning. This project starts from the same honesty: log synapse
changes, and do not call weight movement “learning.”

Weights are **not** reset when the fly changes worlds or tasks.

## Shiritori (one environment, not a brain API)

The keyboard BCI in `flybrain/populations.py` and `shiritori/action_decoder.py`
is an older engineered interface for the letter-grid experiment. It must not
be called from `VirtualFly`. The intended embodied pipeline is screen pixels
→ eyes → MaleCNS → body → physical key. That pipeline is not complete.

## What would count as a result

Immediate goal: one persistent fly exists in the living room with no task
and mixes rest, walking, turning, and stopping. Brain state survives world
changes and save/load.

A later task result is a change in **that same fly**, measured against
frozen / shuffled / rewired controls, with developer teleports excluded.
