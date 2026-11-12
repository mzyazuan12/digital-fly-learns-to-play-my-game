# Paper 2, stage 2B: design of the closed-loop bridge

Design only. Written 12 November 2026 at commit `d658d5d`, before any stage 2B code or run exists. It answers how the MaleCNS front-leg rate model (4,310 neurons) will be coupled to a MuJoCo fly body for the [pre-registered protocol](paper2_protocol.md). Nothing here changes the protocol. The items in [Section 10](#10-protocol-amendments-this-design-needs) must be added to the protocol's Amendments section by the owner before any closed-loop run.

Labels used throughout:

- **ENGINEERED**: not connectome and not published physiology. The same label goes in code.
- **GUESS**: a choice with no source behind it, flagged for review.
- **NEEDS SOURCE**: a claim that has to be tied to a publication before the bridge is frozen.
- **MEASURED**: a number obtained in this repository while writing this design (quick probes, scripts not committed; see [Section 12](#12-what-was-measured-for-this-design)).

---

## 0. Decisions in one page

| Question | Decision |
|---|---|
| Integrator | Fixed-step RK4, h = 0.1 ms, one neural step per MuJoCo step (both at 0.1 ms). Measured against `integrate_rate`: r_min = 0.99996 over active neurons, same active set. |
| Body | FlyGym 2.1 `FlyBody` in `TetheredWorld` (thorax held as a mocap body), legs in air. Front legs actuated by the neural bridge; the other four legs held at the neutral pose by their own position servos. |
| Drive for the test conditions | Both DNg100 (10045 + 10056) at I = 400 (Paper 1 §5.7 "drive both" arm), so that each front leg moves and the mismatched control has a second leg to take signals from. The bridge is first built and validated on the left leg under DNg100_R alone. |
| MN → joint | Module-level agonist/antagonist difference on four FlyBody DOFs per leg, turned into a joint-angle target for FlyBody's own position actuators (equilibrium-point scheme). One offset and one scale per DOF, fixed by recorded walking kinematics on calibration replicates. ENGINEERED. |
| Body → sensory neurons | Leg identity from `rootSide` + MaleCNS `entryNerve`, checked against the side of the motor neurons each afferent contacts. Threshold-referenced input current `I = g_c · 7.5 · s_i · u_i(t)`, with u from the leg's joint state normalised by recorded walking. ENGINEERED. |
| Which afferents get input in 2B | Front-leg hair plates (10 left) and femoral chordotonal neurons (8 left), mirrored on the right, plus 1 right-leg campaniform sensillum. Everything else gets zero, with a stated reason per group. |
| Gain criterion | Motor: joint-angle mean and 5–95% range of the actuated DOFs match recorded tethered walking (FlyGym's bundled NeuroMechFly v1 clip). Sensory: peak afferent rate r* = 100 Hz at the 95th percentile of the recorded stimulus. No coordination read-out is computed until the frozen file is committed. |
| Calibration vs test replicates | Calibration: replicates 960–1023 of the seed-1 block. Test: 0–127. Disjoint. |
| Main risk | A single afferent type (6 left VProN hair plates, type SNpp45) makes 831 synapses onto the 6 left coxa-stance motor neurons. At 10 Hz this input is about three times those neurons' thresholds. If its tuning direction is assigned wrongly, or if it is simply strong, stance timing will follow the body, and the result will be a reflex rather than re-phasing of the CPG. The tuning direction has no source yet ([B1](#b1)). |

---

## 1. What the open-loop network hands to the body

These facts constrain everything below.

- **Only three motor modules move the leg.** In the authors' 1,024 mCNS replicates, coxa swing, coxa stance and femur/trochanter flex are active in 100%; femur/trochanter extend in 7.2%, femur reductor in 0.4%, tibia extend in 0.2%. Tibia flex, tarsus control and substrate grip are absent from the summary (`docs/papers/data/mcns_motor_phase.json`). A median of 8 motor neurons are active (Paper 1 §5.4).
- **Rates are low.** In seed-1 replicates 0–2 (`outputs/mcns_dng100_battery_20261111/intact_10056`), active left motor neurons peak at 2.6–16 Hz with means of 0.5–8.8 Hz, against a 200 Hz cap. The coxa-swing pool has a large tonic component (for example the 1759 promotor: mean 8.8 Hz, range 2.3–16 Hz).
- **No afferent is active open loop.** All 232 sensory neurons sit at zero because nothing drives them. They receive 5,800 synapses from the network, 634 of them excitatory, so presynaptic inhibition of afferents is in the connectome and will act once they are driven.
- **Consequence for 2B.** Under open-loop drive the body will move the coxa and the trochanter–femur joint. The tibia and tarsus will stay where their servos hold them unless feedback recruits their motor neurons. The proprioceptors that see movement in 2B are therefore the ones sensing the coxa and trochanter (hair plates), not the femur–tibia joint (chordotonal organ).

---

## 2. Co-simulation (question 1)

### 2.1 Integrator

`integrate_rate` runs SciPy RK45 over the full 2 s window and cannot accept input computed during the run. The stepped integrator (`flybrain/rate_stepper.py`, new) uses classical fixed-step RK4 with the same right-hand side as `rate_equation_half_tanh`:

```text
I_ext(t) = I_DN · 1[pulse_start ≤ t ≤ pulse_end] + I_sens,k      (I_sens,k held for the whole step)
f(t, R)  = (max(0, r_max · tanh((a / r_max) · (I_ext + W̃ R − θ))) − R) / τ
R_{k+1}  = RK4(f, t_k, R_k, h)
```

- W̃ is the same CSR matrix `reweight_connectivity` produces. Parameters come from `sample_cell_parameters_block(4310, 1024, seed=1)` plus `set_sizes`, exactly as in Paper 1.
- The state is not clipped during integration. `integrate_rate` clips only its output, so the stepper clips only what it records.
- Rates are recorded every 10th step (1 ms), the sampling `integrate_rate` uses through `t_eval`. All existing read-outs (`experiment/leg_coordination.py`, the authors' `compute_oscillation_score`) then run unchanged.

**MEASURED: equivalence to `integrate_rate`.** Replicate 0 (seed 1, DNg100_R, I = 400) against its stored RK45 trace, over neurons active in either run, t ≥ 250 ms:

| h | wall time (2 s, one core, numpy + CSR) | mean r | min r | max abs diff | same active set |
|---|---:|---:|---:|---:|:---:|
| 1.0 ms | 1.3 s | 0.99882 | 0.98357 | 0.17 Hz | yes |
| 0.5 ms | 2.9 s | 0.99984 | 0.99728 | 0.075 Hz | yes |
| 0.25 ms | 6.5 s | 0.99998 | 0.99959 | 0.038 Hz | yes |
| **0.1 ms** | 12.7 s | **0.999997** | **0.99996** | 0.030 Hz | yes |

The stiffest neurons in this network are the small afferents: normalised size as low as 0.02 gives a gain a/s of up to 46. h = 0.5 ms fails a per-neuron r > 0.999 criterion. **Choice: h = 0.1 ms, equal to FlyBody's MuJoCo timestep, one neural step per physics step.** This needs no hold-or-interpolate rule between the two clocks. A 0.2 ms neural step with 2 physics substeps is allowed only if it passes gate G1 below; the gate is the criterion, not this table.

### 2.2 Order of operations in one tick (t_k → t_k + h)

1. Read body state s_k from MuJoCo: joint angles and velocities (`get_joint_angles`, `get_joint_velocities`) and position-actuator forces (`get_actuator_forces`).
2. Sensory currents `I_sens,k = T(s_k)` ([Section 4](#4-body-state--proprioceptor-input-question-3)), or the condition's substitute ([Section 6](#6-protocol-conditions-question-5)).
3. One RK4 step of the rate model, from R_k to R_{k+1}, with `I_sens,k` held.
4. Motor-neuron rates R_{k+1} → joint targets θ*_{k+1} ([Section 3](#3-motor-neuron--joint-mapping-question-2)) → `set_actuator_inputs`.
5. `sim.step()` → s_{k+1}.

The loop latency is therefore one tick (0.1 ms) plus the neural and mechanical dynamics. No conduction or synaptic delay is added in the primary design: delay is a strong phase knob and there is no source for its value in the fly (NEEDS SOURCE if one is ever added).

Body warm-up: 100 ms of physics with all motor-neuron rates at zero (targets at their u = 0 values) before t = 0. The network starts from R = 0 at t = 0, and DNg100 drive starts at 20 ms, as in Paper 1. Run length is T = 2 s, as in Paper 1 and stage 2A; read-outs exclude the first 250 ms.

### 2.3 Gate G1: stepped integrator matches `integrate_rate` (before any body is attached)

On the 8 saved replicates of `intact_10056` and the 8 of `intact_both`, each re-integrated fresh with `integrate_rate` in float64 rather than taken from the float32 files:

- min Pearson r ≥ 0.999 over every neuron active in either run (t ≥ 250 ms);
- identical active set and identical set of active motor neurons;
- motor rhythm frequency within 0.1 Hz; within-leg stance-vs-swing lag within 0.01 cycle;
- passing `I_sens ≡ 0` through the new sensory hook gives traces bit-identical to the stepper without the hook.

---

## 3. Motor neuron → joint mapping (question 2)

### 3.1 What exists in the repository

- `data/motor_neuron_muscle_map.json` has 4 placeholder entries (confidence 0.15–0.2, no body IDs, "muscle target not resolved"). `organism/motor_map.py` resolves nothing usable. **There is no motor map in the repository to reuse.**
- `data/motor_bridge_mapping.json` and `organism/bridge.py` decode descending neurons into FlyGym `HybridTurningController` amplitudes. That is the engineered CPG path; stage 2B does not use it.
- The usable source is the authors' wTable itself: `type` (muscle-named motor neuron types such as "Pleural remotor/abductor MN"), `motor module` and `step contribution` for 129 motor neurons. Motor neurons have `somaSide` (67 L, 62 R) and no `rootSide`. MaleCNS annotations add `exitNerve` (ProLN 41 L / 35 R, ProAN 11/12, VProN 11/11, DProN 4/4). Leg = `somaSide`, which agrees with Paper 1 (DNg100_R drives left-soma motor neurons).

### 3.2 The body's leg degrees of freedom (FlyGym 2.1, MEASURED)

`FlyBody` with `FlyBodyJointPreset.LEGS_ONLY` and `FlyBodyActuatedDOFPreset.LEGS_ACTIVE_ONLY` gives 7 position-actuated DOFs per leg (42 in total). Front-left:

| DOF | range (rad) | stiffness / damping | actuator (FlyBody `actuators.yaml`) |
|---|---|---|---|
| `c_thorax-lf_coxa-pitch` | −0.2 … 1.7 | 1 / 1 | position, kp 80 |
| `c_thorax-lf_coxa-roll` | −0.8 … 0.8 | 1 / 1 | position, kp 80 |
| `c_thorax-lf_coxa-yaw` | −1 … 0.7 | 1 / 1 | position, kp 80 |
| `lf_coxa-lf_trochanterfemur-pitch` | −0.15 … 2.0 | 1 / 1 | position, kp 80 |
| `lf_coxa-lf_trochanterfemur-roll` | −1 … 1 | 1 / 1 | position, kp 80 |
| `lf_trochanterfemur-lf_tibia-pitch` | −1.35 … 1.3 | 1 / 0.4 | position, kp 40 |
| `lf_tibia-lf_tarsus1-pitch` | −0.7 … 1.2 | 1 / 1 | position, kp 40 |
| tarsus 1–5 links (4 pitch joints) | ±0.36 | 10 / 0.2 | passive |

Timestep 0.1 ms. Joints are limited (FlyBody), and the control range equals the joint range. Commands go through `Simulation.set_actuator_inputs(fly, ActuatorType.POSITION, targets)` in `fly.get_actuated_jointdofs_order(...)` order. The body built today by `organism/gait.py` is NeuroMechFly (`make_locomotion_fly`: kp 45, force range ±65, stiffness 0.05, damping 0.06, unlimited joints). Stage 2B uses FlyBody, as the protocol names it, and leaves `organism/gait.py` untouched.

**MEASURED: mechanical transfer at the rhythm frequency** (LF coxa pitch, tethered, sinusoidal command):

| Actuation | 2 Hz: gain, lag | 11.5 Hz: gain, lag |
|---|---|---|
| FlyBody position actuator (target → angle) | 0.98, 9° | **0.76, 40°** (0.11 cycle) |
| Torque (MOTOR) actuator on the same joint (torque → angle) | 0.083 rad per unit, 68° | **0.012 rad per unit, 94°** (0.26 cycle) |

The two actuator models differ by 0.15 cycle of feedback phase at 11.5 Hz. That is about half the shift the within-leg test looks for (0.84 → 0.5 cycle). The actuator model is therefore a result-relevant ENGINEERED choice: it is fixed here, and the other model is a required robustness arm (R1 in [Section 7](#7-fake-positive-risks-and-the-checks-that-catch-them-question-6)).

### 3.3 Mapping table (left front leg; the right leg uses the same table, since FlyBody's leg axes are mirror-symmetric)

| Motor module (wTable) | Left MN types (n) | Muscle(s) (Miller numbering, as named in FlyMimic) | FlyBody DOF | Role |
|---|---|---|---|---|
| coxa swing | Tergopleural/Pleural promotor (4), Sternal anterior rotator (2), Sternal adductor (1) | tergopleural promotor 28a/b, pleural promotor 30, sternal anterior rotator 31, sternal adductor 33 | coxa pitch | agonist (protraction) |
| coxa stance | Pleural remotor/abductor (2), Sternal posterior rotator (4) | pleural remotor and abductor 29, sternal posterior rotator 32 | coxa pitch | antagonist (retraction) |
| femur/tr flex | Tr flexor (7), Acc. tr flexor (3) | trochanter flexor a/b, accessory trochanter flexor | trochanter–femur pitch | agonist (flexion) |
| femur/tr extend | Sternotrochanter (2), Tergotr. (4), Tr extensor (2) | sterno-tergo-trochanter extensor a/b, trochanter extensor | trochanter–femur pitch | antagonist (extension) |
| tibia flex | Ti flexor (5), Acc. ti flexor (10) | tibia flexor | femur–tibia pitch | agonist |
| tibia extend | Ti extensor (2) | tibia extensor | femur–tibia pitch | antagonist |
| tarsus control | Ta depressor (5), Ta levator (2) | (not in FlyMimic) | tibia–tarsus1 pitch | depressor vs levator |
| femur reductor | Fe reductor (4) | (not in FlyMimic) | trochanter–femur roll | unopposed against the servo; GUESS (DOF assignment) |
| substrate grip | ltm, ltm1-tibia, ltm2-femur (8) | long tendon muscle (not in either model) | none: logged, not actuated | claws are not actuated in FlyBody |

- Coxa roll and coxa yaw get no neural drive in the primary design. They are held at the recorded walking mean, labelled ENGINEERED: the rotator, adductor and abductor muscles also act on these axes, but no source fixes their share in FlyBody's axis convention.
- **Signs:** which direction of each FlyBody DOF is protraction, flexion or depression is established by a geometric unit test, not by name. Activating only coxa swing must move the left tarsus tip anteriorly in the thorax frame; activating only femur/tr flex must shorten the coxa–tarsus distance; and so on.
- **Cross-check against a muscle model (MEASURED).** FlyGym ships FlyMimic, an OpenSim-derived muscle model of the left front leg (`flygym/assets/model/musculoskeletal/best_combined_arm_damping_stiff_cvt3.xml`; 15 Hill-type muscles; site names cite Miller's muscle numbers). It compiles without its (non-bundled) meshes because its inertials are explicit. Its moment arms at the default pose (×10⁻³ model length units; MuJoCo sign convention d(length)/d(angle)):

| Muscle | Coxa "yaw" (x axis) | Coxa "pitch" (y) | Coxa "roll" (z) | Tr "pitch" | Tr "roll" | Tibia pitch |
|---|---:|---:|---:|---:|---:|---:|
| tergopleural promotor a | 40 | 23 | −24 | | | |
| tergopleural promotor b | 8 | 17 | −5 | | | |
| pleural promotor | 31 | 23 | 19 | | | |
| sternal anterior rotator | 84 | 6 | 9 | | | |
| sternal adductor | 114 | −55 | 61 | | | |
| pleural remotor and abductor | −35 | −22 | 4 | | | |
| sternal posterior rotator | 8 | −41 | −37 | | | |
| trochanter flexor a / b | | | | 32 / 24 | −13 / −13 | |
| accessory trochanter flexor | | | | 21 | −17 | |
| sterno-tergo-trochanter extensor a / b | | | | −29 / −31 | 6 / −15 | |
| trochanter extensor | | | | −10 | 14 | |
| tibia flexor / extensor | | | | | | −15 / 33 |

  On the axis that carries the read-out (coxa "pitch", the lateral y axis = protraction/retraction), the promotors and the two stance muscles have opposite signs, as the module table assumes. Trochanter flexors and extensors are opposed on trochanter pitch, and tibia flexor and extensor on tibia pitch. Two exceptions are recorded rather than hidden. The sternal adductor, labelled "coxa swing", has a retractor-sign pitch moment and acts mainly about the x axis. The sternal anterior rotator also acts mainly about x. FlyMimic's axis names do not match FlyBody's ("yaw" is the x axis in FlyMimic), so any mapping between the two must use axis vectors, never names.

### 3.4 The agonist–antagonist scheme (ENGINEERED)

For leg ℓ, module m and actuated DOF j:

```text
D_m(t)  = Σ_{i ∈ MNs of module m on leg ℓ} R_i(t)                  [Hz]  (recruitment adds force)
u_j(t)  = D_ag(j)(t) − D_ant(j)(t)                                  [Hz]
θ*_j(t) = clip( μ_j + Δ_j · (u_j(t) − c_j) / s_j ,  joint range )   [rad] → FlyBody position actuator
```

- c_j and s_j are the mean and the 5–95% half-range of u_j over the calibration replicates in the deafferented body (t ≥ 250 ms). They are frozen constants, not running estimates, so in closed loop a change in tonic drive does move posture.
- μ_j and Δ_j start at the recorded walking mean and 5–95% half-range of DOF j. Each is then corrected multiplicatively (at most 3 iterations) for the actuator's attenuation, until the deafferented joint-angle mean and range match the recording ([Section 5](#5-gain-calibration-independent-of-coordination-question-4)).
- DOFs with no drive during calibration (s_j < 0.5 Hz: in practice tibia, tarsus and trochanter–femur roll): c_j = 0, s_j = the median s over the driven DOFs, μ_j and Δ_j taken from the recording uncorrected. Fixed now so that feedback-recruited joints still have a defined mapping.
- **No extra muscle low-pass in the primary design.** The motor-neuron τ (~20 ms) and the body's own mechanics are the only filters, consistent with FlyBody (position servos) and with FlyMimic, whose activation time constants are 0.1/0.4 ms. A 10/40 ms first-order activation (MuJoCo's default muscle dynamics, not a fly measurement) is optional robustness arm R6.
- Why an equilibrium-point target rather than torque: it uses FlyBody's own published actuator model unchanged. Torque on FlyBody's joint damping (b = 1) makes every joint an integrator at 11.5 Hz (94° lag, 0.012 rad per unit torque), so the loop phase would be set by one damping constant whose value nobody measured. Both choices are ENGINEERED, which is why R1 runs the other.

Everything in 3.4 is labelled `ENGINEERED` in code, and every constant goes into the frozen file.

---

## 4. Body state → proprioceptor input (question 3)

### 4.1 Which afferents belong to which leg

Sensory neurons in the wTable have `rootSide` (L/R; one has neither) and no `somaSide`. MaleCNS v1.0 body annotations (`body-annotations-male-cns-v1.0-minconf-0.5.feather`) add `entryNerve`. **Rule: leg = (`rootSide`, `entryNerve`).** Prothoracic leg nerve (ProLN) → that side's front leg. VProN, DProN and ProAN → that side's front leg (coxa/trochanter region), pending [B3](#b3). PrN (prosternal nerve) → not a leg. MesoLN / MetaLN → middle / hind leg.

**MEASURED check against connectivity.** Each afferent with ≥ 10 synapses onto motor neurons was compared with the side of the motor neurons it contacts (MN `somaSide` = leg). For ProLN, VProN, DProN and ProAN, 53 of 54 such afferents send ≥ 80% of their MN synapses to the `rootSide` leg (median 100%). The exception is campaniform 1078099767 (ProLN, rootSide R), which makes all 39 of its MN synapses onto left motor neurons; it is excluded as ambiguous. The intersegmental campaniform sensilla (MesoLN/MetaLN) are inconsistent (1 of 4 and 2 of 3 consistent), as expected for afferents crossing segments. A third check is available offline: the fraction of each afferent's presynapses in `LegNp(T1)(L)` versus `(R)` from the local `syn-points` file (13 GB). It is a validation test for step 2 of the plan, not run here.

### 4.2 Inventory and the 2B inclusion decision

Left side (right-side counts in brackets). Synapse counts are direct synapses onto left motor neurons or core CPG cells, from W.

| Group (subclass, nerve, type) | n L [R] | Normalised size (median) | Direct targets (left leg) | 2B input | Reason |
|---|---:|---:|---|---|---|
| hair plate, VProN, SNpp45 | 6 [5] | 0.59 | **coxa stance 831**, reductor 108, swing 12 | **coxa-pitch limit detector** | front-leg hair plate; direction per [B1](#b1) |
| hair plate, DProN, SNpp52 | 3 [3] | 0.82 | femur/tr extend 175, coxa stance 148, swing 53, reductor 53; **I2 69**, I1 5 | **trochanter–femur-pitch limit detector** (GUESS for the joint) | as above |
| hair plate, ProLN, SNpp45 | 1 [0] | 0.12 | none onto MNs | same rule as type SNpp45 | type-level rule |
| chordotonal organ, ProLN (SNpp39 ×2, 41, 50, 51 ×3, 59) | 8 [4] | 0.16 | SNpp51: coxa swing 95, tibia flex 66, femur/tr extend 54, I2 20; others small | **femur–tibia angle + velocity** (direction-free default) | femoral chordotonal organ; subtype per [B1](#b1) |
| campaniform sensilla, ProLN, SNpp53 | 0 [2] | 1.0 | (right leg) | right trochanter load for 877244; 1078099767 excluded (4.1) | the network has **no left front-leg campaniform sensilla** |
| campaniform sensilla, MesoLN/MetaLN, SNpp53 | 4 [4] | 0.6–1.3 | femur/tr extend ~120 | 0 | middle/hind-leg afferents; those legs do not move in 2B (they enter in 2C) |
| hair plate, PrN, SNpp19 | 18 [16] | 0.64 | coxa swing 5, reductor 47 | 0 | prosternal organ (neck), likely: the head is fixed (NEEDS SOURCE, [B3](#b3)) |
| mechanosensory bristle, ProLN | 53 [46] | 0.08 | reductor 220, tibia extend 40 | 0 | tactile; no contact with the leg in air |
| "leg", ProLN (mostly untyped) | 38 [9] | 0.08 | femur/tr extend 96, tibia extend 76, reductor 54, …; I2 52 | 0 | modality unknown |
| "leg", ProAN, SNxxxx | 4 [4] | 0.47 | **coxa swing 277** | 0 | modality unknown (a large exclusion, stated) |
| DNx01/DNx02, AN/ON-entry, neck | 3 + 3 | — | small | 0 | not leg proprioceptors |

So in 2B the closed loop runs through 18 left afferents: 10 hair plates (6 VProN, 3 DProN, and 1 ProLN SNpp45 that has no direct motor-neuron synapses) and 8 chordotonal neurons. On the right it runs through 8 hair plates, 4 chordotonal neurons and 1 campaniform sensillum. Excluding the "leg" subclass and bristles is a choice. It is fixed now, before any closed-loop result, and listed as a limitation.

### 4.3 Transduction (ENGINEERED)

For afferent i, with normalised size s_i = size_i / nanmedian(size):

```text
I_sens,i(t) = g_c · 7.5 · s_i · u_i(t),     u_i ≥ 0 dimensionless,  g_c per sensory class
```

- **Units and scale.** The rate model's threshold for neuron i is θ_raw · s_i with θ_raw ~ N(7.5, 0.6), and its gain is a_raw / s_i. With this input, the steady rate is `r_max · tanh((a_raw / r_max) · (7.5 · g_c · u − θ_raw))₊`, which **does not depend on s_i**. A small afferent and a large one respond alike to the same stimulus, and the replicate's own random draws of a, θ and r_max keep their effect. u = 0 gives zero input, identical to open loop. The threshold is crossed at u ≈ 1/g_c.
- **g_c from a target rate r\*.** The rate at u = 1 for nominal parameters (a = 1, θ = 7.5, r_max = 200) is set to r\*: `g_c = 1 + (200/7.5) · atanh(r*/200)`. For r\* = 100 Hz, g_c = 15.65. r\* is the only sensory gain, and it is fixed a priori ([Section 5](#5-gain-calibration-independent-of-coordination-question-4)). Presynaptic inhibition from the network then subtracts from this input, as the connectome dictates.
- **Stimulus features u_i** (all percentiles from kinematic replay of recorded walking, [Section 5.2](#52-sensory-normalisation-recorded-walking-only-no-network-runs)):

| Class | Joint signal | u_i |
|---|---|---|
| Hair plate (limit detector) | angle of its DOF, direction d = ±1 | `[d·(θ − θ_on)]₊ / (θ_95,d − θ_on)`, with θ_on = 80th percentile toward d and u = 1 at the 95th. The limit-detector idea is NEEDS SOURCE (Pratt et al. 2024, unverified). The 80/95 percentiles are a GUESS. |
| Chordotonal (default, direction-free) | femur–tibia angle and angular velocity | `0.5 · |θ − θ_50| / p95(|θ − θ_50|) + 0.5 · |θ̇| / p95(|θ̇|)` |
| Campaniform (right front leg only) | trochanter–femur pitch actuator force (cuticular strain proxy) | `|F| / p95(|F|)` |

- **Why a direction-free chordotonal default.** Femoral chordotonal neurons are position-tuned (claw), direction-tuned (hook) or vibration-tuned (club) (Mamiya, Gurung & Tuthill 2018). Which MaleCNS types (SNpp39/41/50/51/59) are which is not known here ([B1](#b1)). Absolute values avoid inventing a direction, but they produce a 2f component if the tibia oscillates. Check F11 watches for this. In practice the tibia is held unless feedback recruits tibia motor neurons, so the chordotonal input will be small.
- **Hair-plate direction.** The joint each hair-plate type senses and which limit excites it decide the sign of the strongest feedback path in the network (SNpp45 → coxa stance, 831 synapses). Choosing that sign from the wiring (for example "it excites remotors, so it must detect protraction") would build the answer in. It must come from an anatomical or physiological source ([B1](#b1)). Without one, the pre-registered fallback is a 2 × 2 factorial over the directions of SNpp45 and SNpp52 ([Section 6.3](#63-replicates-arms-and-cost)).

---

## 5. Gain calibration independent of coordination (question 4)

### 5.1 Motor calibration: deafferented body, calibration replicates only

- **Target data (MEASURED).** `flygym_demo/ball_flybody_data/assets/ball_flybody_clip.npz`, shipped with FlyGym 2.1: 1 s at 100 fps of tethered walking on a ball from the NeuroMechFly v1 dataset (Lobato-Rios et al. 2022), joint angles reconstructed by inverse kinematics on FlyBody, so already in FlyBody's convention. Left front leg, 5th–95th percentile: coxa pitch 0.14 … 0.98 rad (48°), trochanter–femur pitch −0.10 … 1.21 (75°), femur–tibia pitch −0.55 … 0.86 (81°), tibia–tarsus1 pitch 0.23 … 0.87 (37°), trochanter–femur roll −0.25 … 0.74 (57°), coxa roll −0.15 … 0.80, coxa yaw −0.19 … 0.48. The dominant frequency of LF coxa pitch is about 7 Hz. **Limitation:** this is one fly and 1 s, at a slower step rate than the network's 11.5 Hz. A multi-fly joint-angle dataset (for example Karashchuk et al. 2021, Anipose) would be better, but needs conversion to FlyBody's convention ([B5](#b5)). Only ranges and means are matched, never frequency or phase.
- **Procedure.** Bilateral drive, deafferented body (I_sens ≡ 0), replicates 960–1023 (64), T = 2 s, statistics over t ≥ 250 ms pooled across replicates. Compute c_j and s_j. Set μ_j and Δ_j. Iterate μ_j and Δ_j (at most 3 times) until, for coxa pitch and trochanter–femur pitch of each front leg, the median replicate's mean is within ±0.1 rad of the recording and its 5–95% range within ±20%. Undriven DOFs follow the fallback in 3.4.
- **What calibration may compute:** joint-angle percentiles, means, motor-neuron rate percentiles. **What it may not compute:** any lag, phase, PLV, cross-correlation or coherence. This is enforced by a test that inspects the calibration module's imports (the AST may not reference `experiment.leg_coordination` or any phase function).

### 5.2 Sensory normalisation: recorded walking only, no network runs

Replay the recorded clip on the tethered FlyBody (cubic-spline interpolation of the 100 fps angles to 0.1 ms, position servos tracking it; the right leg replays the RF trace). Record each feature's distribution (θ, θ̇, actuator force), and take θ_on, θ_95 and p95 from it. With r\* fixed, nothing else is fitted. No neural simulation is involved, so the sensory gains cannot carry information about network phase.

**r\* = 100 Hz** (half the model's mean rate cap). NEEDS SOURCE: no Drosophila leg-proprioceptor spike rates are cited here. Mamiya et al. 2018 is calcium imaging. Larger insects' leg afferents are commonly reported at a few hundred Hz, but that too needs a citation before use. r\* ∈ {50, 200} Hz is robustness arm R2.

### 5.3 Freeze

1. The calibration script writes `data/paper2/stage2B_bridge_frozen.json`. It contains: the sensory inclusion table (body IDs, leg, class, feature, direction), the MN → DOF table, the constants μ, Δ, c and s per leg and DOF, the feature percentiles, g_c, r\*, h, T, the drive, calibration and test replicate IDs, surrogate seeds, the actuator model, the FlyGym (2.1.0) and MuJoCo (3.9.0) versions, the data SHA-256 hashes (wTable, W, MaleCNS annotations, walking clip) and the git commit. It also stores a `sha256` of its own canonical JSON.
2. Calibration acceptance results and every iteration are written beside it (a failure is kept, not overwritten).
3. **The owner commits** the frozen file and the protocol amendment that quotes its hash. The design author does not commit.
4. The stage 2B runner refuses to start unless the frozen file's hash equals the hash quoted in the protocol amendment, and unless the bridge source files are clean in git. Every output records the hash.
5. Any change after this point means a new frozen file, a new amendment with its reason, and re-running every condition and replicate. No partial reruns.

---

## 6. Protocol conditions (question 5)

### 6.1 Body set-up for all conditions

- **Tethered, legs in air (recommended for 2B).** `TetheredWorld` holds the thorax as a mocap body (no free joint, no drift). Both front legs are actuated by the bridge; the four other legs are held at FlyBody's neutral pose by their own position servos. Adhesion is off, and there is no ground and no contact.
- **Why tethered rather than on the ground.** (1) Only the front legs are under neural control. A free fly on five servo-held legs would stand rather than walk, and posture or falling would become a confound that 2B does not test. (2) The network contains no left front-leg campaniform sensilla and no load-sensing afferents with a known modality, so ground contact would add little sensory information to the 2B loop. Tactile bristles are the exception, and their transduction would be a new engineered layer. (3) The FlyGym API supports tethered FlyBody directly. Runs are deterministic and cheap. (4) Inter-leg mechanical coupling through the ground belongs to 2C, where the protocol puts it.
- **What tethered-in-air cannot test.** Load feedback. A negative 2B result therefore does not show that the body loop cannot coordinate legs. It shows that unloaded proprioceptive feedback through these afferents does not. A treadmill (passive ball) variant is a possible later amendment; it is not part of this design.

### 6.2 Conditions

Drive: DNg100_L (10045) + DNg100_R (10056), I = 400 each, pulse 20 ms – 1.999 s, as in Paper 1 §5.7. Primary within-leg read-out: T1L (DNg100_R's leg, the Paper 1 baseline). Secondary: T1R. Exploratory only: T1L–T1R coupling, which is formally a 2C question.

| Condition | Body | Sensory input `I_sens(t)` | Note |
|---|---|---|---|
| **open loop** | none (stepper alone) | 0 | the baseline |
| **deafferented body** | moves under the bridge | 0 | In 2B nothing else connects body to network, so its neural traces must be **bit-identical** to open loop. It is a plumbing check here, not an independent control. Stated in the results. |
| **closed loop** | moves under the bridge | each leg's afferents ← that leg's state through T | |
| **mismatched** | moves under the bridge | left afferents ← right leg's state; right afferents ← left leg's state (the only derangement of two legs), same T and normalisation (FlyBody legs are mirror-symmetric, so no sign flips; verified by the replay test) | needs bilateral drive; see 6.4 for the unilateral fallback |
| **scrambled timing** | moves under the bridge | surrogate of this replicate's closed-loop body-state channels, then T | see below |

**Scrambled-timing surrogate.**

- Record, in closed-loop replicate k, every body-state channel that T reads (angles, velocities and forces of both legs; 0.1 ms; 0–2 s).
- Remove each channel's mean, then apply **multivariate Fourier phase randomisation**: one set of uniform random phases per frequency, shared by all channels (Prichard & Theiler 1994). This keeps every channel's power spectrum and all cross-spectra exactly, including the exact angle → velocity relation, since differentiation is linear. Add the means back.
- Clip the surrogate angles to the joint ranges and pass them through the same T.
- Rerun replicate k with the body moving under the bridge but the afferents reading the surrogate.
- Surrogate seed = hash(seed, k, "scramble"). Scrambling is applied to body state before the rectifying transduction, so rectification does not leave negative "rates" to repair. IAAFT (Schreiber & Schmitz 1996) is not used because its amplitude step breaks the angle/velocity consistency.
- Dependency: the closed-loop run of each replicate must finish first.

### 6.3 Replicates, arms and cost

- Test replicates 0–127 of the seed-1 block (the protocol requires ≥ 64 for (a)). Calibration replicates 960–1023 never enter a read-out.
- **Primary family:** 5 conditions × 128 replicates. If [B1](#b1) is unresolved at freeze, run all 4 combinations of hair-plate direction (SNpp45 ± × SNpp52 ±): 4 × 4 body conditions + 1 open loop.
- **Rule for the factorial, fixed now:** if a source exists, only its combination is primary. If none exists, the paper reports how many of the 4 combinations pass in closed loop with all controls failing. A positive claim needs every passing combination to have failing controls. A pass in one combination is reported with that combination named and with check F1's verdict; the passing combination is never singled out as "the" result.
- **Cost (MEASURED basis):** ~13 s neural plus ~1–2 s physics per 2 s run on one core, so ~15 s per run. The primary family without the factorial is 640 runs (≈ 2.7 core-hours); with the factorial ≈ 2,200 runs (≈ 9 core-hours). Robustness arms ([Section 7](#7-fake-positive-risks-and-the-checks-that-catch-them-question-6)) add closed loop + 3 body controls on 64 replicates each.

### 6.4 If the owner prefers unilateral drive (DNg100_R only)

The right front leg would be nearly still (Paper 1: occasional single right motor neurons below 1 Hz). Right → left feedback would then be a near-constant input, close to deafferented, and so a weak control. The fallback is a **cross-replicate derangement**: replicate k's left afferents read the left-leg body state recorded in closed-loop replicate π(k), where π is a fixed derangement of 0–127. The signal is a real leg's own kinematics, but caused by a different network. This mirrors the protocol's cross-replicate null. It needs an amendment because it is "a different fly's leg", not "a different leg".

---

## 7. Fake-positive risks and the checks that catch them (question 6)

| # | Risk | Size | Check (all computed and reported after unblinding; none used to choose anything) |
|---|---|---|---|
| F1 | **Reflex imposition.** One afferent group dominates the stance pool, so stance follows the body and the CPG is not re-phased. | MEASURED: the 6 left VProN SNpp45 make 74–231 synapses onto each of 5 of the 6 left coxa-stance motor neurons (mean thresholds 13–24). At ρ Hz in all six, the input to the 214-synapse remotor is 0.03 · 214 · ρ ≈ 6.4ρ: about 3× threshold at 10 Hz, about 27× at 100 Hz. Its open-loop network input ranges from −48 to +56. | Split each stance motor neuron's suprathreshold input into the direct-afferent part and the rest (W̃ columns split by presynaptic class). If the afferent part is > 50% when the neuron is above threshold, label the result **"monosynaptic-reflex-driven alternation"**, not emergent coordination. Also report the fraction of time stance motor neurons spend above 0.9 r_max. |
| F2 | Rhythm moved to the body or the loop, not the CPG | — | Report E1/E2 frequency and phase relative to the coxa angle, against open loop. **Closed loop with E1 + E2 removed** (Paper 1 lesion), 64 replicates: if alternation persists without the CPG, the oscillation belongs to the reflex loop. |
| F3 | Actuator or controller dynamics set the phase | MEASURED: 40° vs 94° body lag at 11.5 Hz for position vs torque actuation | **R1:** rerun closed loop + controls with torque (MOTOR) actuators, calibrated by the same criterion. A conclusion that flips is reported as actuator-dependent. Also report the measured gain and phase of each actuated DOF (from a chirp on the deafferented body) in the paper. |
| F4 | Feedback gain sets the answer | g_c is fixed by r\* | **R2:** r\* = 50 and 200 Hz. Report pass/fail at all three. A pass at only one gain is reported as gain-dependent. |
| F5 | "Any lagged copy of the motor command" would do; the body is incidental | — | Diagnostic: replace the body by each DOF's measured linear transfer function (target → angle) and keep the same T. If the result is the same, write "feedback of the motor command filtered by the leg", not "body physics". |
| F6 | Body resonance imposes frequency | FlyBody tethered leg under position servo: first-order-like, corner ≈ 1/(2π · 12 ms) ≈ 13 Hz, no resonance peak seen at 2 or 11.5 Hz | Report the closed-loop frequency against open loop and against the body's corner frequency. Flag frequency locking near the corner. |
| F7 | Plumbing leak | — | Bit-identity: deafferented = open loop; closed loop with g_c = 0 = open loop; mismatched with a left-to-left map = closed loop. These are run as tests, not only claimed. |
| F8 | Calibration leakage | — | Disjoint replicate sets, the AST import ban (5.1), and the frozen hash checked by the runner. |
| F9 | Read-out artefacts: a saturated or tonic module gives a meaningless z-score lag | — | A lag counts only when both coxa modules are active and each module trace has the authors' oscillation score ≥ 0.5 with frequencies within 1 Hz. Otherwise the replicate is "unmeasurable" and counts as **not passing** (protocol rule 3: kept, not dropped). |
| F10 | Harmonics from direction-free features | — | Spectrum of each afferent's input against the motor-neuron rhythm. A dominant 2f peak is flagged. |
| F11 | Joint-limit impacts drive the hair plates in bursts | FlyBody joints are limited (`solreflimit` 1 ms) | Fraction of time at limits, per DOF and condition. |
| F12 | Tonic afferent drive set by the pre-stimulus posture | the u = 0 posture may sit near a hair plate's onset | Afferent rates logged for 0–20 ms (before DN drive) and for t ≥ 250 ms. |
| F13 | Numerical artefact of the coupled system | — | 16 replicates rerun at h = 0.05 ms: read-outs must not change (lag within 0.01 cycle). Instability or NaN is a failed replicate, kept. |
| F14 | Body model dependence | — | **R3 (recommended):** the same experiment, unilateral (6.4), on the FlyMimic muscle leg. Motor neuron type → named Hill muscle (ctrl = clip(D_type / s_global, 0, 1)), one global scale calibrated to the same coxa range, model's own passive properties. Physics needs no download (it compiles mesh-free); rendering needs its meshes ([B6](#b6)). |

The protocol's own controls (mismatched, scrambled, deafferented) remain the pass criterion. F1–F14 decide how a pass is described.

---

## 8. Implementation plan (question 7)

Each step ends with a test that must pass before the next step starts. New files only. Existing code is imported, not edited.

| Step | Files to create | Validation test |
|---|---|---|
| 0 | none | Owner resolves or accepts the defaults for B1–B6 and adds the amendments in [Section 10](#10-protocol-amendments-this-design-needs). |
| 1 | `flybrain/rate_stepper.py` (RK4 stepper with an `I_ext` hook; CSR); `tests/test_rate_stepper.py` | **G1** ([2.3](#23-gate-g1-stepped-integrator-matches-integrate_rate-before-any-body-is-attached)) on 8 + 8 saved replicates; the zero-hook bit-identity test; determinism (two runs bit-equal). |
| 2 | `experiment/paper2_sensory_table.py` → `data/paper2/stage2B_sensory_table.json`; `tests/test_paper2_sensory_table.py` | The counts in 4.2 are reproduced exactly. Every included afferent has ≥ 80% ipsilateral MN synapses (or < 10 MN synapses). No PrN, MesoLN or MetaLN afferent is included. 1078099767 is excluded. Optional: the `LegNp(T1)` presynapse ROI fraction from `syn-points` is ≥ 0.8 on the assigned side. |
| 3 | `organism/flybody_tethered.py` (builds tethered FlyBody, held legs, front-leg actuators of the chosen type, a state getter); `tests/test_flybody_tethered.py` | 7 actuated DOFs per leg with the names in 3.2. Held legs stay within 0.01 rad over 2 s. Two runs are bit-identical. A chirp transfer function is saved per front-leg DOF. Replay of the recorded clip tracks the targets with RMS < 0.05 rad. |
| 4 | `organism/mn_joint_bridge.py` + `data/paper2/stage2B_mn_joint_table.json`; `tests/test_mn_joint_bridge.py` | All 129 motor neurons are assigned or explicitly "logged, not actuated". Geometric sign tests: swing → tarsus tip anterior; flex → coxa–tarsus distance shorter; tibia flex → tibia toward femur. FlyMimic moment-arm sign agreement on pitch for promotor, remotor, posterior rotator, trochanter and tibia flexors and extensors (mesh-free compile). |
| 5 | `organism/proprio_transduction.py`; `tests/test_proprio_transduction.py` | u = 0 → I = 0. Size invariance (same u gives the same rate for s = 0.02 and s = 1 at nominal parameters). u = 1 → 100 ± 10 Hz median over a replicate's draws. Replay normalisation reproduces the stored percentiles. |
| 6 | `experiment/paper2_cosim.py` (tick loop of 2.2, the five conditions, surrogate generator); `tests/test_paper2_cosim.py` | F7 bit-identities. Surrogate: per-channel power spectrum preserved (relative error < 1e-10), cross-spectra preserved, seed-reproducible. Order-of-operations test on a 2-neuron toy network with a known one-tick delay. |
| 7 | `experiment/paper2_calibrate_2B.py` → `data/paper2/stage2B_bridge_frozen.json`; `tests/test_paper2_calibrate.py` | Refuses an existing output. Uses only replicates 960–1023. The AST import ban. Acceptance criteria evaluated and written, pass or fail. |
| 8 | (owner) commit of the frozen file + amendment quoting its hash | The runner's hash check passes on a clean tree and fails on a dirty one. |
| 9 | `experiment/paper2_stage2B_run.py` | Refuses without a matching hash. Writes rates (1 ms), body state and afferent input per replicate. Failures are kept. The closed loop of replicate k is written before its scrambled run starts. |
| 10 | `experiment/paper2_stage2B_readout.py` → `docs/papers/data/stage2B_*.json`, then `docs/papers/paper2_stage2B.md` | Reuses `experiment/leg_coordination.py` unchanged, then F1–F14. The read-out script is the first code in stage 2B allowed to compute a lag. |

---

## 9. Engineered parts, in one list

The paper's ENGINEERED table: the MN-module → DOF table (3.3); the equilibrium-point scheme and its per-DOF affine constants (3.4); position actuation through FlyBody servos; no ab/adduction or rotation drive; no added muscle dynamics or delay; the afferent inclusion and exclusion choices (4.2); threshold-referenced transduction; the hair-plate onset and saturation percentiles; the direction-free chordotonal feature; the campaniform force proxy; r\* = 100 Hz; tethered in air; held non-front legs. Alternatives tried are R1 (torque), R2 (r\*), R3 (FlyMimic) and the hair-plate direction factorial.

---

## 10. Protocol amendments this design needs

To be added by the owner, with dates, before any closed-loop run:

1. **2B drive.** The test conditions use bilateral DNg100 (10045 + 10056, I = 400 each; Paper 1 §5.7 arm), so that the mismatched control has a second moving leg. Primary within-leg read-out: T1L. Bridge construction and validation use DNg100_R alone.
2. **2B body.** FlyGym 2.1 FlyBody, tethered (`TetheredWorld`), legs in air. Front legs under the bridge; the other legs held by position servos at the neutral pose.
3. **2B mismatched control** = swap of left and right front-leg signals. Fallback if unilateral: cross-replicate derangement (6.4).
4. **Deafferented = open loop in 2B.** No other body-to-network path exists, so the two are bit-identical by construction. Both are run, and the identity is a pipeline test.
5. **Hair-plate direction.** Taken from the source in B1 if one is found before freeze; otherwise the 2 × 2 factorial and reporting rule of 6.3.
6. **Frozen bridge hash.** The SHA-256 of `data/paper2/stage2B_bridge_frozen.json`.

---

## 11. Blocking unknowns and NEEDS SOURCE items

<a id="b1"></a>**B1. Afferent subtype → joint and tuning direction (blocks the primary hair-plate assignment).** Hair plates: VProN SNpp45 (6 L / 5 R), DProN SNpp52 (3 / 3), ProLN SNpp45 (1 / 0). Chordotonal organ: SNpp39, 41, 50, 51, 59 (claw, hook or club?). Candidate sources, all to be verified: Pratt et al. 2024 (hair plates as limit detectors, Tuthill lab; citation not verified); FANC ↔ MANC sensory-type matching from the Tuthill-lab and MANC annotation papers; Mamiya et al. 2018 for FeCO coding. This is the single most result-relevant unknown (F1). Without it the factorial runs.

<a id="b2"></a>**B2. Owner choice of bilateral versus unilateral drive** for the test conditions (6.2 versus 6.4).

<a id="b3"></a>**B3. Nerve identities.** VProN, DProN and ProAN as front-leg (coxa/trochanter) afferent nerves, and PrN as the prosternal (neck) organ nerve. NEEDS SOURCE (for example Court et al. 2020 VNC nomenclature, and the MANC/MaleCNS sensory annotations). The ROI check in step 2 is the local evidence.

<a id="b4"></a>**B4. Drosophila proprioceptor spike-rate range** for r\*. NEEDS SOURCE. The default is 100 Hz, with R2.

<a id="b5"></a>**B5. Walking kinematics target.** The bundled clip is one fly and 1 s. A multi-fly dataset in FlyBody's convention would be better. Not blocking: the default is stated and the limitation reported.

<a id="b6"></a>**B6. FlyMimic provenance and meshes.** The model's publication (it is converted from OpenSim with MyoConverter; possibly Ozdil et al., Ramdya lab, unverified) is NEEDS SOURCE before R3 is reported. Its meshes (about 14 MB per FlyGym's docstring, from `datasets.epfl.ch`) are not cached. They are needed only to render R3, and downloading them needs the owner's OK.

**Accepted risk (not a blocker):** afferent synapses carry the same 0.03 weight per synapse as every other synapse (rule 1 forbids changing it). Paper 1 validated that scale only with all afferents silent. As F1 shows, any plausible afferent rate makes hair-plate input many times motor-neuron threshold. The owner should accept, before results exist, that a positive 2B result may turn out to be reflex-dominated, and that it will then be reported that way.

---

## 12. What was measured for this design

The probes were short and their scripts were not committed: the RK4-versus-RK45 comparison on replicate 0 (Section 2.1); FlyBody and NeuroMechFly tethered builds with DOF, actuator and range listing; a 1 s sinusoidal frequency-response probe on LF coxa pitch for position and torque actuation (3.2); a mesh-free compile of FlyMimic and its moment arms (3.3); sensory and motor-neuron connectivity tallies from W and the MaleCNS annotations (4.1, 4.2, F1); joint-angle percentiles of the bundled walking clip (5.1). Package versions: flygym 2.1.0, mujoco 3.9.0, scipy 1.18.1, numpy 2.5.3.

## References used (only ones this design relies on)

- Pugliese et al. 2025/2026, bioRxiv 10.1101/2025.09.12.675944 (network, rate model, read-outs; the shipped wTable).
- Lobato-Rios V. et al. 2022, NeuroMechFly, *Nature Methods* (source of the walking clip FlyGym bundles).
- Vaxenburg R. et al. 2025, *Nature*, doi 10.1038/s41586-025-09029-4 (FlyBody; as cited in FlyGym's docstring).
- Mamiya A., Gurung P., Tuthill J.C. 2018, Neural coding of leg proprioception in *Drosophila*, *Neuron*.
- Azevedo A. et al. 2024, *Nature* (female VNC; leg motor-neuron muscle targets; cited in the repository's motor map).
- Miller A. 1950, internal anatomy of the *Drosophila* imago, in *Biology of Drosophila* (muscle numbering used by FlyMimic site names).
- Prichard D. & Theiler J. 1994, *Phys. Rev. Lett.* (multivariate phase-randomised surrogates); Schreiber T. & Schmitz A. 1996, *Phys. Rev. Lett.* (IAAFT, not used).
- Karashchuk P. et al. 2021, Anipose, *Cell Reports* (candidate better kinematics source, B5).
- To verify before citing: Pratt et al. 2024 (hair plates as limit detectors); Court et al. 2020 (VNC nerve nomenclature); the FlyMimic / OpenSim leg model publication.
