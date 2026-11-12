# Paper 2, stage 2B: closing the loop through the front legs

Results record for stage 2B of the [pre-registered protocol](paper2_protocol.md) (amendments A1–A7), using the bridge specified in [paper2_stage2B_design.md](paper2_stage2B_design.md). Run 12 November 2026.

## Question

Does feeding the front legs' own joint state back into the connectome through its identified proprioceptors convert the open-loop rhythm into (a) swing/stance alternation within a leg or (b) left/right coupling, where the open-loop network produces neither?

## Set-up

- **Brain:** the authors' 4,310-neuron MaleCNS front-leg network with Paper 1's validated rate model, stepped in lockstep with the body by fixed-step RK4 at 0.1 ms (gate G1 against the reference integrator: min r = 0.9999998 on 16 replicates). Drive: both DNg100 (10045 and 10056) at I = 400 (A1).
- **Body:** FlyGym 2.1 FlyBody, tethered, legs in air. Both front legs are position-actuated by the bridge; the other four legs are held at the neutral pose. Measured body lag at 11.5 Hz is 0.74 gain and 42° for coxa and femur (A3).
- **Motor neurons → joints:** all 129 front-leg motor neurons are assigned to agonist/antagonist channels on coxa pitch, trochanter–femur roll and pitch, tibia and tarsus1. Signs come from FlyBody geometry, cross-checked against FlyMimic moment arms.
- **Joints → afferents:** 31 identified proprioceptors (18 left, 13 right: hair plates, femoral chordotonal neurons, one right campaniform), each assigned to its leg by root side and entry nerve. Current is I = g·7.5·sᵢ·u, with g set so that u = 1 gives the target rate r\* (100 Hz primary). Hair-plate direction is unknown, so all four direction combinations are run with none designated primary (A1.5).
- **Calibration:** motor offsets and scales were fitted on separate replicates (960–1023) to the joint ranges of recorded walking, then frozen (SHA-256 `91bc54c3…`, A7). 7 of 8 criteria passed; the right femur's resting angle sits 0.132 rad off, against a 0.1 limit, and is a stated limitation.
- **Test set:** replicates 0–63. Conditions: open loop, deafferented, closed loop, scrambled-timing feedback (Fourier phase-randomised body state) and mismatched feedback (left/right swapped), each across the 4 hair-plate cells. Primary set 896 runs; robustness arm R2 at r\* = 50 and 150 Hz, 768 runs each. No run failed or produced invalid rates. Deafferented equals open loop bit for bit in 64 of 64 replicates (pipeline check).

## Pre-registered result

**Negative.** No hair-plate cell passes within-leg alternation (T1L or T1R) or left/right coupling in closed loop, at any of the three feedback gains. No positive verdict exists under the protocol's rule (closed loop passes and every control fails).

Within-leg read-out on the left front leg: R_all counts unmeasurable replicates as not passing. A replicate is measurable when both coxa modules are rhythmic (score ≥ 0.5) with frequencies within 1 Hz.

| Condition | r\* = 100 Hz (primary): mean lag, R_all, measurable/64 | 50 Hz | 150 Hz |
|---|---|---|---|
| open loop / deafferented | 0.81, **0.99**, 64 | (same) | (same) |
| closed loop, protraction–extension | 0.16, 0.16, 10 | 0.15, 0.49, 33 | 0.35, 0.05, 3 |
| closed loop, protraction–flexion | 0.59, 0.06, 28 | 0.74, 0.17, 24 | 0.88, 0.13, 18 |
| closed loop, retraction–extension | —, 0.00, 0 | 0.53, 0.04, 3 | —, 0.00, 0 |
| closed loop, retraction–flexion | —, 0.00, 0 | —, 0.00, 0 | —, 0.00, 0 |

Pass requires a mean lag in [0.375, 0.625] and R_all ≥ 0.8. Full tables, including the right leg, left/right coupling and every control, are in `docs/papers/data/stage2B_{primary,R2_r50,R2_r150}_readout.json`.

## What happens instead

Diagnostics on the primary set (medians over 64 replicates, after 250 ms; `docs/papers/data/stage2B_primary_diagnostics.json`):

| Condition | Left MN mean rate | Time at coxa-pitch limit | Coxa-pitch range |
|---|---:|---:|---:|
| deafferented | 0.25 Hz | 0% | 0.91 rad |
| closed loop, protraction–extension | 25 Hz | 100% | 0.00 rad |
| closed loop, protraction–flexion | 0.7 Hz | 33% | 0.99 rad |
| closed loop, retraction–extension | 20 Hz | 98% | 0.09 rad |
| closed loop, retraction–flexion | 18 Hz | 100% | 0.00 rad |
| scrambled, protraction–extension | 23 Hz | 100% | 0.00 rad |
| mismatched, protraction–extension | 25 Hz | 100% | 0.00 rad |

1. **Latching.** In three of four hair-plate cells, the feedback drives the leg into a joint limit and holds it there. Motor neurons fire ~100× harder than without feedback, and the CPG rhythm is replaced by tonic drive. Scrambled and mismatched feedback latch the leg the same way. The latching therefore comes from the strength of the feedback, not its timing.
2. **Disruption.** In the remaining cell (protraction–flexion), the leg keeps moving, but the coxa rhythm survives in fewer than half the replicates and its swing/stance phase is not concentrated.
3. **Lower gain does not rescue it.** At r\* = 50 Hz the latching is partial (33 of 64 replicates measurable in one cell), but stance then lags swing by ~0.15 cycle, further from alternation than the open-loop 0.81.
4. **High-R, low-PLV artefacts.** Several left/right pairs at 150 Hz have R near 1 but PLV ≤ 0.4, because legs pinned at limits give a fixed, non-oscillatory lag. The PLV criterion and the cross-replicate null reject them as intended.

## Interpretation

This is the risk accepted in advance (A1.7, design F1). Afferent synapses carry the network-wide weight of 0.03 per synapse, validated in Paper 1 only with every afferent silent. At physiological afferent rates, a single hair-plate group's input to the coxa-stance motor neurons is many times their threshold. The connectome supplies synapse counts but not the efficacy of sensory synapses, and with the efficacy implied by the central-synapse scale, proprioceptive feedback overwhelms the front-leg CPG instead of pacing it.

The result does **not** show that body feedback cannot coordinate legs. It shows that, in this rate model, unloaded proprioceptive feedback through these afferents at the validated central synaptic scale does not, and that sensory synaptic efficacy is a missing quantity the next stage must constrain from data rather than fit.

## Limitations

- **Tethered, unloaded:** no load feedback; the network has no left front-leg campaniform sensilla.
- **Hair-plate direction unknown:** all four combinations were run.
- **Right femur:** its calibration criterion failed (A7).
- **Torque actuation (R1) not run:** it needs its own torque calibration and frozen bridge.
- **Single walking clip** for joint ranges and sensory normalisation.

## Next steps

1. **Sensory efficacy (new pre-registered stage).** Constrain hair-plate and chordotonal → motor-neuron/interneuron efficacy from published electrophysiology, or treat it as an explicit scanned parameter with its own pre-registered criteria. Report the regime, if any, where feedback paces rather than latches the CPG.
2. **R1 torque arm**, with its own calibration and freeze.
3. **Stage 2C** (six legs, ground contact) only after the efficacy question has an answer.
