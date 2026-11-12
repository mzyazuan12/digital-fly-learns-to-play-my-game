# Paper 2, stage 2A: the six-leg nerve cord without a body

Results record for stage 2A of the [pre-registered protocol](paper2_protocol.md). Run 11 November 2026.

## What was run

- **Network:** the authors' full MANC VNC (`wTable/W_20251006`, 23,532 neurons, all six leg neuropils), the network behind their Figure 4 and Extended Data Figure 9b.
- **Stimulus:** both DNg100 (indices 59 and 282, MANC bodies 10093 and 10339) at I = 380. Taken from the authors' archived configs (Zenodo 22260924, `DNg100_Stim_fullManc.zip`, MD5 0f3d47ef…4120). The front-leg value I = 250 leaves this network almost silent (26–41 active neurons, no motor neurons), because DNg100's normalized size is computed against a smaller median here.
- **Authors' runs:** 4 × 32 replicates, seeds 641, 98, 445, 424.
- **Ours:** 128 replicates, seed 1, same equation, tolerances and read-outs.

## Agreement with the authors

- **Parameters:** our sampler with set_sizes reproduces all four runs' tau, gain, threshold and fr_cap to ≤ 2.7 × 10⁻⁷ relative, which is float32 precision.
- **Stimulus:** the stored input currents are non-zero only at 59 and 282, value 380.
- **Traces:** for replicates 0–3 of run 30662613, integrated with the authors' parameters, r = 0.99997–0.999998 over all active neurons. The active sets are identical, including replicate 2, where 8,039 neurons are active.

## Read-outs (pre-registered)

Per leg, authors / ours: fraction of replicates rhythmic (score ≥ 0.5), median frequency, median motor neurons active, and replicates in which the swing/stance lag could be measured (both coxa modules active):

| Leg | Rhythmic | Frequency (Hz) | MNs active | Swing/stance measurable | Alternation pass |
|---|---|---|---|---|---|
| T1L | 0.76 / 0.72 | 10.1 / 9.8 | 2 / 2 | 4 / 7 | no |
| T1R | 0.60 / 0.57 | 11.1 / 10.8 | 1 / 1 | 2 / 5 | no |
| T2L | 0.48 / 0.43 | 9.5 / 7.5 | 1 / 1 | 1 / 0 | no |
| T2R | 0.81 / 0.78 | 11.5 / 11.2 | 2 / 2 | 1 / 6 | no |
| T3L | 0.17 / 0.17 | 6.0 / 6.7 | 2 / 2 | 0 / 3 | no |
| T3R | 0.81 / 0.75 | 9.8 / 9.7 | 11 / 11 | 66 / 66 | no |

Between legs (E1 traces). Each cell gives R across replicates / median within-run PLV. Pass needs R ≥ 0.8, R above the null 99th percentile, and PLV ≥ 0.8:

| Pair | Authors R / PLV | Ours R / PLV | Null R p99 (ours) | Mean lag, cycles (ours) | Coupled |
|---|---|---|---|---|---|
| T1L-T1R | 0.10 / 0.29 | 0.12 / 0.22 | 0.22 | 0.01 | no |
| T1L-T2L | 0.39 / 0.66 | 0.51 / 0.73 | 0.24 | 0.93 | no |
| T1L-T2R | 0.11 / 0.22 | 0.03 / 0.16 | 0.21 | 0.68 | no |
| T1L-T3L | 0.22 / 0.42 | 0.13 / 0.37 | 0.27 | 0.91 | no |
| T1L-T3R | 0.20 / 0.24 | 0.13 / 0.23 | 0.20 | 0.87 | no |
| T1R-T2L | 0.21 / 0.27 | 0.29 / 0.32 | 0.26 | 0.82 | no |
| T1R-T2R | 0.30 / 0.44 | 0.30 / 0.38 | 0.22 | 0.98 | no |
| T1R-T3L | 0.16 / 0.19 | 0.07 / 0.23 | 0.23 | 0.88 | no |
| T1R-T3R | 0.09 / 0.53 | 0.13 / 0.57 | 0.20 | 0.95 | no |
| T2L-T2R | 0.06 / 0.16 | 0.11 / 0.18 | 0.24 | 0.94 | no |
| T2L-T3L | 0.50 / 0.59 | 0.50 / 0.63 | 0.30 | 0.10 | no |
| T2L-T3R | 0.02 / 0.17 | 0.09 / 0.20 | 0.27 | 0.96 | no |
| T2R-T3L | 0.10 / 0.20 | 0.25 / 0.21 | 0.26 | 0.99 | no |
| T2R-T3R | 0.75 / 0.99 | 0.83 / 0.98 | 0.20 | 0.08 | **pass** (ours only) |
| T3L-T3R | 0.12 / 0.17 | 0.06 / 0.22 | 0.19 | 0.94 | no |

## Findings

1. **The authors' negative result holds in our implementation, at full scale.** Left and right legs are never coupled: every contralateral pair is at or below the null. There is no tripod, because that needs contralateral antiphase.
2. **Same-side neighbours partly lock, in phase.** T1L–T2L and T2L–T3L reach R ≈ 0.5 (null p99 ≈ 0.24–0.30). Right middle–right hind locks strongly in both samples (PLV 0.98–0.99, R 0.75 authors / 0.83 ours), and formally passes in ours, at a lag of ~0.08 cycle. In a walking fly, adjacent same-side legs step in antiphase. So the only coupling the nerve cord produces alone has the wrong sign for walking. We have not seen this partial ipsilateral locking reported before; the authors summarise the six-leg result as absent phase coupling.
3. **No leg shows swing/stance alternation.** In most legs both coxa modules are rarely active together; where they are (T3R, 66 of 128 replicates), stance lags swing by ~0.72 cycle (R 0.65–0.81), not 0.5.
4. **Rhythm is uneven across legs.** The fraction of rhythmic replicates ranges from 0.17 (T3L) to ~0.8 (T2R, T3R), as the authors note for the hind legs.

## Consequence

Without a body, the connectome does not produce a walking gait. The full nerve cord gives per-leg rhythms, no left/right coordination, and in-phase rather than alternating same-side locking. This is the baseline stage 2B/2C must beat: if closing the loop through FlyBody produces contralateral antiphase or ipsilateral antiphase that passes the protocol, and the mismatched, scrambled and deafferented controls do not, the coordination comes from the body loop.

## Notes

- A NaN frequency for a leg whose E1 had no clear period broke the JSON save after all 128 of our simulations had finished. The rates had already been saved. The NaN is now written as null, and the read-outs were recomputed from the saved rates; no simulation was rerun or changed.
- Commands: `.venv/bin/python -m experiment.full_manc_compare` and `.venv/bin/python -m experiment.full_vnc_dng100 --stim-amplitude 380 --seed 1 --param-block 128`. Summaries: `docs/papers/data/stage2A_*.json`.
