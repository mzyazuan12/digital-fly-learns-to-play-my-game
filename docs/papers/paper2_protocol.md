# Paper 2 protocol: where does leg coordination come from?

Pre-registered 11 November 2026, before any Paper 2 experiment was run. Changes after this date are listed under [Amendments](#amendments) with their reason; the original text is not edited.

## Background

Paper 1 established that DNg100 drive on the male CNS front-leg network produces an ~11.5 Hz motor rhythm through the E1/E2 CPG, but:

1. **Within a leg**, coxa-stance motor neurons lead coxa-swing by ~0.16 cycle (circular mean 0.84 cycles, R = 0.94) instead of alternating.
2. **Between legs**, the left and right front-leg rhythms run at different frequencies with no consistent phase relation.

Pugliese et al. report the same on MANC. Within each leg there is "a consistent phase offset between antagonistic muscles". Between legs, "phase coupling was absent across the six leg CPGs in the full connectome simulation" (23,532 neurons). Their conclusion: DNg100 drive alone "is sufficient to generate within-leg motor coordination but not realistic interleg coordination", and "proprioceptive feedback, biomechanical coupling, or other neural pathways may be necessary". They name coupling the VNC simulation to a physics body model as the way to test this. That test is the subject of Paper 2.

## Question

Does closing the loop through a physical body (motor neurons → leg joints → physics → proprioceptors → the same connectome) produce (a) swing/stance alternation within a leg and (b) consistent phase coupling between legs, where the open-loop connectome produces neither?

## Stages

| Stage | Network | Body | Purpose |
|---|---|---|---|
| 2A | Full MANC VNC, 23,532 neurons, bilateral DNg100 | none | Reproduce the authors' open-loop negative result in our implementation at full scale; validate against their archived run if available |
| 2B | MaleCNS front-leg network (4,310) | FlyBody front legs, other legs passive or held | Build and validate the motor-neuron → joint bridge and proprioceptor → sensory-neuron feedback; within-leg test (a) |
| 2C | Six-leg network (full MANC VNC; full MaleCNS VNC once neuron sizes are available) | Full FlyBody on flat ground | Between-leg test (b) |

## Conditions (2B and 2C)

| Condition | Description |
|---|---|
| **open loop** | Same network and drive; body passive (or absent); sensory neurons receive no input |
| **closed loop** | Proprioceptors of each leg are driven by that leg's own joint state and load from the physics |
| **mismatched feedback** (control) | Each leg's proprioceptors are driven by a *different* leg's signals (fixed derangement) |
| **scrambled-timing feedback** (control) | Same feedback signals with their phases randomized (Fourier phase-scrambled, power spectrum kept) |
| **deafferented body** (control) | Body moves under neural command; all proprioceptor inputs set to zero |

The prediction "body feedback produces coordination" is supported only if closed loop shows the effect and all three controls do not.

## Read-outs (fixed now)

- **Within-leg phase:** cross-correlation-peak lag of coxa-stance (remotor) relative to coxa-swing (promotor) mean z-scored motor-neuron traces, in cycles, after a 250 ms transient. Summarised by the circular mean and resultant length *R* across replicates.
- **Between-leg phase:** the same lag between corresponding motor modules (coxa swing) of each leg pair, plus a within-run phase-locking value (PLV) from Hilbert phases of the band-passed (±3 Hz around the run's rhythm frequency) traces.
- **Null for between-leg phase:** the authors' method: pair each replicate's leg *i* with leg *j* from a different, randomly chosen replicate.
- **Tripod index (2C only):** fraction of cycles in which the two tripods (L1, R2, L3 and R1, L2, R3) are each internally in phase within ±0.125 cycle and in antiphase with each other within ±0.125 cycle.

## Pass criteria (fixed now)

- **(a) Within-leg alternation:** circular mean of remotor-vs-promotor lag in [0.375, 0.625] cycles, *R* ≥ 0.8, across ≥ 64 replicates.
- **(b) Between-leg coupling** for a leg pair: *R* ≥ 0.8 across replicates **and** above the 99th percentile of *R* under the cross-replicate null, **and** median within-run PLV ≥ 0.8.
- **Tripod coordination (2C):** median tripod index ≥ 0.5 and above the 99th percentile of the cross-replicate null.

A result is reported as positive only if it passes in closed loop and fails in every control. Partial results (for example coupling of contralateral but not ipsilateral pairs) are reported as such.

## Rules that cannot be changed after seeing results

1. **No neural tuning toward coordination.** Rate-model parameters, connectome weights, size normalization and stimulus amplitude stay at the values validated in Paper 1. No parameter is adjusted after measuring phase or coupling.
2. **Bridge gains are fixed by an independent criterion.** The motor-neuron → joint gains and the joint-state → proprioceptor gains are set once to make joint angle ranges and sensory firing rates fall within published ranges for walking flies. They are then frozen and committed before any coordination read-out is computed.
3. **Every replicate is kept.** Failed, unstable or non-rhythmic runs are reported, not dropped.
4. **Engineered parts are labelled.** Anything that is not connectome or published physiology (muscle model, joint mapping, sensor transduction) is marked `ENGINEERED` in code and in the paper, with the alternatives tried.

## Open dependencies

- Authors' archived full-MANC DNg100 run (Zenodo 22260924, `DNg100_Stim_fullManc.zip`, 2.0 GB) for the 2A trace-level check. Needs the owner's OK to download.
- MaleCNS neuPrint `size` for all VNC neurons, for a six-leg MaleCNS network. Needs a neuPrint API token.
- Motor-module → FlyBody joint mapping: coverage is partial in `data/motor_neuron_muscle_map.json`; 2B starts with the coxa (swing/stance) and femur/trochanter modules, which carry the Paper 1 phase result.

## Amendments

**A1 (12 November 2026), before any closed-loop run.** Adopted from the stage 2B design, §10 ([paper2_stage2B_design.md](paper2_stage2B_design.md)). The owner chose bilateral drive.

1. **2B drive:** both DNg100 (MaleCNS 10045 + 10056, I = 400 each, Paper 1 §5.7 arm). The primary within-leg read-out is T1L. Bridge construction and validation use DNg100_R alone.
2. **2B body:** FlyGym 2.1 FlyBody, tethered, legs in air. Both front legs are under the bridge; the other four legs are held by position servos at the neutral pose.
3. **2B mismatched control:** the left and right front-leg feedback signals are swapped.
4. **Deafferented equals open loop in 2B,** because there is no other body-to-network path. Both are run, and their bit-identity is a pipeline test.
5. **Hair-plate tuning direction:** taken from a verified published source if one is found before the gains are frozen. Otherwise all four direction combinations are run and reported, with none designated primary.
6. **Frozen bridge:** the runner refuses to start unless the SHA-256 of `data/paper2/stage2B_bridge_frozen.json` matches the hash quoted in a later amendment, committed before any coordination read-out.
7. **Accepted in advance:** afferent synapses keep the network-wide 0.03 weight per synapse (rule 1). A positive result may therefore be reflex-dominated (strong hair-plate input to the coxa-stance motor neurons). If the design's F-checks show that, it will be reported as reflex-driven alternation, not as re-phasing of the CPG.

**A2 (12 November 2026), gate G1 reference.** G1, as specified in the design (§2.3), failed on 3 of 16 replicates. Min per-neuron r was 0.9963, 0.9986 and 0.9990 against the 0.999 bar. Active sets, motor-neuron sets, frequency and swing/stance lag were identical in all 16. The failing neurons are almost flat after 250 ms (s.d. ≈ 0.0006 Hz). Against a tighter RK45 reference (rtol 1e-9, atol 1e-11), the stepper at h = 0.1 ms gives r = 0.99985, and r = 0.99999 at h = 0.025 ms. The original-tolerance reference itself correlates with the tighter one at only 0.9977, so the error was in the reference. The gate now compares against the tighter reference with the same 0.999 bar. The failed run is kept at `outputs/paper2_gate_g1_20261112`. The same check exposed a timing bug in the stepper: drive onset drifted by one step through accumulated `t += h`, and the RK4 sub-stages sampled the onset inside a step. The stepper now holds all inputs (DN drive and sensory current) constant over each step and computes time as k·h. With that fix, G1 passes on all 16 replicates: min r = 0.9999998, identical active and motor-neuron sets, identical frequency and swing/stance lag (`outputs/paper2_gate_g1_final_20261112`).

**A3 (12 November 2026), step-3 body gate.** The design's step-3 check (replaying the bundled walking clip on FlyBody's front legs with RMS < 0.05 rad) fails: per-DOF RMS is 0.07–0.25 rad, mostly from the servos' own ~10 ms lag (damping / (stiffness + kp) ≈ 12 ms). The clip conversion was verified separately against FlyGym's shipped FlyBody step file, to 1e-6. Replacement gate: the measured transfer function of every front-leg DOF is recorded in `data/paper2/stage2B_body_transfer.json` and reproduced by a live run. The failing test stays in the suite as a strict expected failure. At 11.5 Hz the measured values are: position servo gain 0.74 and lag 41.6° (coxa, trochanter–femur), 0.80 / 35° (tibia), 0.48 / 60.5° (tarsus1); torque 0.014 rad per unit, ~89°. These supersede the design's §3.2 probe values (0.76 / 40°, 0.012 / 94°), which were taken while the leg was still drifting. Sensory normalisation (design §5.2) uses the clip's own angles and velocities, not the lagged replay. No coordination read-out has been computed.

