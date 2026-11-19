# Paper 2, stage 2M: design of the next pre-registered mechanism test

Design only, written 19 November 2026 after the stage 2B-κ result ([paper2_stage2Bk.md](paper2_stage2Bk.md)) and before any stage 2M run. Nothing here is committed. The only computations behind it are the probes in §2: connectome tallies, a re-analysis of the A9 calibration audit file, and two 2 s open-loop network runs on calibration replicates 960 and 961. None of them computes a phase, lag or coupling value.

Binding context: [protocol](paper2_protocol.md) rules 1–4 and amendments A1–A9, the [stage 2B design](paper2_stage2B_design.md) (F1–F14), and the [literature check](paper2_sensory_efficacy_literature.md). The only sources cited are ones that file marks VERIFIED.

## 0. Decisions in one page

- **Mechanism (primary): candidate (a), size-principle afferent efficacy, in its "equal-current" form.** Every motor neuron (MN) that receives direct input from the included afferents gets the same total afferent synaptic weight. The afferents' synapses onto each MN keep their connectome proportions, and the model's existing size scaling (threshold × s, gain / s) then sets the recruitment order. There is no new free parameter: κ is recalibrated with the A8 criteria.
- **Why (a) and not (b):** the measurements in §2 locate the A9 failure in the direct hair-plate → large coxa-stance MN synapses. The 9A-targeted (hook-like) afferents carry 1.1% of direct afferent → MN synapses. In the model, 9A inhibition of them is already present, and is tonic and DNg100-driven. No verified Drosophila magnitude is available to calibrate presynaptic inhibition (PI) against.
- **Calibration criterion (independent of coordination):** the same as A8 (C1 and C2, effectiveness), plus a new recruitment-order index O. Same grid, same calibration replicates (960–1023), same clip stimulus, DN drive off, no body.
- **Confirmatory test:** the A8 test at κ\*₂M (replicates 0–63, 5 conditions × 4 hair-plate cells, 896 runs). The read-out and pass rules are unchanged. It runs only if the calibration shows feedback that is effective and size-ordered (owner decision D2).
- **Descriptive arms:** a κ scan (2,304 runs) and a size-specificity control with the normalisation factors permuted (256 runs).
- **Frozen files stay frozen:** the work uses new files only. The A7 bridge hash (`91bc54c3…`), the A9 κ file hash (`0e59ac6b…`) and the code hashes recorded in the A9 file all verify today (§10.4).
- **Cost:** about 4.5 h wall time on 9 workers (§11).

## 1. What stages 2B and 2B-κ leave open

- At the central scale (κ = 1), afferent feedback latches the leg or disrupts the CPG (stage 2B).
- With one sensory multiplier κ fixed by Azevedo et al. 2020's constraints, κ\* = 0.02 and feedback is ineffective. No κ gives alternation or left/right coupling.
- On the whole grid, afferent drive never recruits small MNs while sparing large ones (A9).

The 2B-κ record names the missing ingredient as something that sets which MNs feedback recruits. Stage 2M tests the most direct physiological candidate for that.

## 2. What was measured for this design

All counts come from the shipped `W_20260210_vncRoisOnly.csv` (signed synapse counts, pre × post; SHA-256 prefix `255236ba`) and the wTable (`593a75c5`). s is wTable `size` divided by the 4,310-row median (998,603,428.5).

### 2.1 Direct afferent → MN wiring is anti-size-ordered

- **Volume and location.** The 31 included afferents make 12,764 output synapses. All are positive (no negative entries). 2,133 of them go directly onto MNs, in 140 (afferent, MN) pairs onto 44 of the 129 front-leg MNs (30 left, 14 right). Every contact is at least 5 synapses.
- **By channel:**

  | Channel | Direct MN synapses |
  |---|---:|
  | Coxa hair plate | 1,172 |
  | Trochanter hair plate | 704 |
  | Chordotonal | 257 |
  | Campaniform | 0 |

  Hair plates (SNpp45, SNpp52) carry 88% of the direct MN input.
- **Contact probability rises with MN size** (size terciles as in A8):

  | Leg | Smallest tercile contacted | Middle | Largest | Direct synapses (smallest / middle / largest) |
  |---|---|---|---|---|
  | Left | 5/23 | 7/22 | 18/22 | 103 / 243 / 1,291 |
  | Right | **0/21** | 5/21 | 9/20 | 0 / 69 / 427 |

- **Among contacted MNs, synapse count rises with size.** The log–log slope of n_j against s_j is 0.74 (left) and 1.22 (right). That almost cancels the model's 1/s size scaling. Spearman ρ between size and the drive-to-threshold ratio n_j / s_j is −0.20 (left) and +0.13 (right). The most strongly driven MN per unit threshold is a largest-tercile coxa-stance MN in both legs.
- **Total input** (all presynaptic partners) scales as s^2.17 (left) and s^2.13 (right).
- **The biggest contacts** are all coxa stance:

  | MN (leg) | Synapses | s |
  |---|---:|---:|
  | Pleural remotor/abductor (L) | 302 | 2.88 |
  | Pleural remotor/abductor (L) | 223 | 3.13 |
  | Sternal posterior rotator (L) | 194 | 1.94 |
  | Sternal posterior rotator (L) | 129 | 1.75 |
  | Sternal posterior rotator (L) | 118 | 2.04 |
  | Pleural remotor/abductor (R) | 93 | 2.98 |

### 2.2 The MNs that break C1 in the A9 calibration are those same MNs

Re-analysis of the A9 audit file (`outputs/paper2_stage2Bk_calibration_20261116T025804Z/mn_max_rates_after_250ms.npz`, active = maximum rate > 1 Hz after 250 ms):

- **First MNs recruited.** At κ = 0.03–0.05, the first MNs recruited in the protraction cells are the largest-tercile left Sternal posterior rotator MN (194 synapses; active in 78% of replicates at κ = 0.03) and the two Pleural remotor/abductor MNs (302 and 223 synapses; 100% at κ = 0.05).
- **At κ = 0.1** a Tergotr. MN with no direct afferent contact is also active, so indirect (interneuron) paths contribute too.
- **Recruitment order index under A8.** κ_rec,j is the smallest grid κ at which MN j is active in ≥ 50% of replicates, and O = Spearman ρ(s_j, κ_rec,j) over the MNs recruited somewhere on the grid. Under the size principle, O > 0.

  | Cell | O | MNs recruited (of 129) |
  |---|---:|---:|
  | protraction–extension | −0.15 | 68 |
  | protraction–flexion | −0.21 | 66 |
  | retraction–extension | −0.26 | 72 |
  | retraction–flexion | −0.23 | 72 |

  The model recruits in reverse size order. This value is computed after the fact from data already published in the A9 record. It is descriptive only and does not enter any κ\* rule.

### 2.3 9A neurons and the included afferents (candidate b)

- **9A → afferent synapses.** The network has 93 IN09A-type neurons, all GABAergic. They make **451 synapses, in 15 (pre, post) pairs**, onto included afferents:

  | Afferent target | Synapses from 9A | Share of its inhibitory input |
  |---|---:|---:|
  | All 5 SNpp39 | 318 | 96% (of 331) |
  | 1 SNpp41 | 108 | 94% (of 115) |
  | 1 SNpp59 | 12 | 48% (of 25) |
  | 2 left SNpp45 hair plates | 13 (all from IN09A001_L) | small |

  The 9A neurons that make them:

  | 9A neuron | Synapses onto included afferents | Targets |
  |---|---:|---|
  | IN09A021_R | 180 | 3 right SNpp39 |
  | IN09A012_L | 143 | SNpp41 100, SNpp39 38, SNpp59 5 |
  | IN09A021_L | 65 | 1 left SNpp39 |
  | IN09A033_L, IN09A033_R | 15, 8 | — |
  | IN09A016_L | 15 | — |
  | IN09A012_R | 12 | — |

  72% of IN09A012_L's output goes to sensory axons, including 113 synapses onto a non-included SNxx30.
- **Hook-like by connectivity (inference, not a verified type mapping).** Dallmann et al. 2025 report that in FANC, 83.1% of the GABAergic input synapses on hook axons come from the 9A hemilineage, and one chief 9A neuron gives 56.8% of their presynaptic input. SNpp39 and SNpp41 match that profile. SNpp50 and SNpp51 receive no 9A input, which suggests claw or club. This bears on the stage 2B design's open item B1.
- **Leverage.** These 7 hook-like afferents make 24 of the 2,133 direct afferent → MN synapses (1.1%) and 955 of 12,764 outputs (7.5%).
- **Activity (probe; open loop, replicates 960 and 961, DNg100 drive on, no afferent input):**
  - IN09A012_L fires at 8.2–10.8 Hz and IN09A012_R at 6.1–6.3 Hz. Both are tonic (s.d. ≤ 0.43 Hz). They receive 58 and 51 synapses directly from DNg100.
  - IN09A021_L/R, IN09A033, IN09A016 and IN09A001_L are silent.
  - IN09A021 is itself driven by SNpp39 (up to 116 synapses), so it forms a recurrent hook → 9A → hook loop.
  - All included afferents are silent in open loop (maximum 0 Hz). Open loop therefore does not depend on afferent column weights.
- **The existing subtractive PI is not weak everywhere.**
  - On SNpp41: 0.03 · 100 · (8.2–10.8 Hz) = 25–32 input units, against its full sensory current at u = 1 of g · 7.5 · s = 15.65 · 7.5 · 0.145 = 17.0 (r\* = 100 Hz). With nominal parameters (replicate parameters vary), it fires only for u ≳ 1.5–2.
  - On the left SNpp39 (row 4188): 9–12 units against 12.6.
  - So under DNg100 drive, the model already applies a near-veto, tonic PI to 2 of the 7 hook-like afferents. For the other 5, PI is absent until their own activity recruits IN09A021.

  The 2B-κ interpretation ("their effect is a weak subtraction") is therefore only partly accurate; see owner decision D5.
- **Hair plates' axonal inhibition** is 449 synapses: IN19A076 (left, 211), untyped GABAergic neurons (right, 163), IN06A006 (45), and 9A only 13. IN19A076 is silent in open loop. Pratt et al. 2026 observed no suppression of hair-plate calcium during spontaneous movement.

## 3. Choice of mechanism

| | Grounding | Independent calibration criterion | Fits frozen bridge/runner with new files only | Leverage on the observed failure |
|---|---|---|---|---|
| **(a) equal-current afferent → MN efficacy** | Measured principle: size-ordered recruitment by feedback (Azevedo 2020); connectome counts (§2.1) | A8's C1/C2/effectiveness, plus order index O. Verified target, no coordination | Yes: rescales the 140 entries in afferent columns × MN rows, like κ | Direct: acts on the synapses that break C1 (§2.2) |
| (a′) exponent family f_j = s_j^−γ | Same principle | Same, plus a γ grid | Yes | Needs γ ≳ 1.1–1.6 to order the key left-leg pairs. Spearman of the drive ratio with size is only −0.27 to −0.77 at γ = 1–2 (right leg weaker). The best-driven right MN stays in the middle tercile. One extra fitted parameter with no measurement |
| (a″) input-fraction normalisation (÷ total input N_j) | Common connectomic convention (no verified source in the literature file) | Same | Yes | Spearman −0.66 / −0.59. Implies an effective s^−3.2 penalty, since N_j ∝ s^2.2 |
| (b) divisive 9A gating of hook afferents | Identified cells (Dallmann 2025) | None verified for Drosophila; locust 50% EPSP reduction is ABSTRACT-ONLY | Yes | 1.1% of direct MN input. Tonic and DN-driven in this network, so no phase dependence to exploit. Hair plates carry no 9A PI |
| (c) load feedback | Strong in principle | Needs loaded stance | No: tethered legs in air, no left campaniform afferent | Needs stage 2C |
| (d) spiking/graded units | — | None identified | No: replaces the Paper 1 rate model (rule 1) and needs a new G1 gate | Unknown |

**Choice: (a), equal-current form.**

1. It is the only candidate whose target is a verified physiological measurement that the model demonstrably violates (O < 0, §2.2).
2. It acts where the failure is.
3. It adds no parameter beyond κ, which is recalibrated by the existing A8 rule.

(a′) and (a″) are recorded alternatives (D1). (b) is deferred to a later stage that first splits the chordotonal channel into claw and hook (§7.4).

## 4. Mechanism: equations and the entries of W it touches

**Derivation (mine, flagged ENGINEERED; not a source claim).**

- In the rate model, neuron j's steady rate is r_max · tanh((a / r_max) · (I_j / s_j − θ))₊, because `set_sizes` divides gain by s_j and multiplies threshold by s_j. Input is effectively divided by size, so the model already contains the intrinsic part of the size principle.
- Size-ordered recruitment by a common input then requires that the synaptic current I_j from that input does not grow with s_j.
- Connectome counts violate this (§2.1). Azevedo et al. 2020 observe the outcome the principle predicts: PSPs largest in slow MNs, smallest in fast, and "we never observed feedback-evoked spikes in fast motor neurons".
- Stage 2M therefore removes the size dependence of total afferent current per MN and keeps everything else.

**Definitions.**

- A = the 31 included afferents (`included_2B` rows of `stage2B_sensory_table.json`).
- M = the 129 wTable rows with a non-empty `motor module`.
- N_ij = synapse count from i to j in the shipped W.
- n_j = Σ_{i∈A} N_ij.
- C = {j ∈ M : n_j > 0}, 44 MNs.
- n_ref = median_{j∈C} n_j = **28**.
- W̃ = `Network.W`, the post × pre CSR matrix: 0.03 × signed counts, transposed by `reweight_connectivity`.

**Rule.** Row j is postsynaptic and column i is presynaptic.

```
f_j = n_ref / n_j      for j in C;      f_j = 1 otherwise

W2M_ji(κ) = κ · f_j · W̃_ji    for i in A, j in M
          = κ · W̃_ji          for i in A, j not in M    (as A8)
          = W̃_ji              for i not in A          (unchanged)
```

**Properties.** Each is checked by a test in §10.

- Σ_{i∈A} W2M_ji = 0.03 · κ · 28 = **0.84 κ** for every j ∈ C.
- The afferent drive to MN j in threshold units is 0.84 κ · r̄_j / (θ s_j), where r̄_j = Σ_i N_ij R_i / n_j is the synapse-weighted mean rate of its afferents. At equal r̄_j, it falls with s_j.
- Changed entries relative to the κ network: **140** (columns A × rows C), covering 2,133 synapses. f ranges from 0.093 (302-synapse remotor) to 5.6 (5-synapse contacts). 21 MNs are amplified, 21 reduced and 2 unchanged.
- No new non-zeros, no sign changes. Central synapses (columns outside A) and afferent → interneuron synapses (beyond κ) are unchanged.
- f ≡ 1 reproduces the A8 κ network exactly, and κ = 1 with f ≡ 1 reproduces stage 2B.

**What it cannot do.** It cannot create contacts. Right-leg smallest-tercile MNs have no direct afferent input, so on T1R direct recruitment can at best begin in the middle tercile. Indirect (interneuron) recruitment, which is active at κ ≥ 0.1 in A9, is still scaled only by κ, so C1 may still bind at a κ where small MNs are not yet recruited. That outcome is pre-registered as a possible negative (§5.4).

## 5. Independent calibration

### 5.1 Set-up (identical to A8)

- Replicates 960–1023 of the seed-1 block. DN drive off. No body.
- Afferents are driven by the transduction of the recorded walking clip (1 s, differentiated, tiled to 2 s), both front legs, r\* = 100 Hz, in each of the 4 hair-plate cells.
- The stimulus SHA-256 must equal A9's, `c6ede5aa3cc305d4…`.
- RK4 at h = 0.1 ms, afferent input held per step. Grid: the A8 grid, 12 values of κ from 1 to 0.001.
- 12 κ × 4 cells × 64 replicates = 3,072 network-only runs.

### 5.2 Criteria (fixed before any 2M run)

- **C1 and C2:** exactly as A8 (no largest-tercile MN active; at most one third of each leg's MNs active). A (κ, cell) passes if both hold together in ≥ 95% of replicates.
- **κ\*₂M:** the largest grid κ passing in all 4 cells.
- **Effectiveness label at κ\*₂M:** as A8. A cell is effective if ≥ 50% of replicates have ≥ 1 active smallest-tercile MN on either leg. The label is "effective" (all cells), "mixed" (some cells) or "feedback ineffective" (no cell).
- **Order index O_c (new):**
  - Per cell, κ_rec,j is the smallest grid κ at which MN j is active in ≥ 50% of replicates (first crossing; the number of non-monotone MNs is reported).
  - O_c is the Spearman ρ between s_j and κ_rec,j over MNs recruited somewhere on the grid.
  - The label is **"size-ordered"** if O_c > 0 in all 4 cells, each with ≥ 10 recruited MNs. Otherwise "not size-ordered".
  - The A8 baseline O_c (−0.15 to −0.26, §2.2) is reported alongside.
- **Status "ready"** requires all three: κ\*₂M exists, the label is effective or mixed, and the recruitment is size-ordered (D2). Otherwise the confirmatory arm is refused and the calibration table is the stage's result.

### 5.3 Freeze

The frozen file `data/paper2/stage2M_frozen.json` records:

- the 44-row factor table (row, bodyId, side, module, s, n_j, f_j) and n_ref;
- the full criterion table and O_c per cell;
- κ\*₂M and its labels, and the status;
- the hashes of A7, A9, data and code (§10.4).

Its SHA-256 is committed in a follow-up amendment (A11) before any test run. The runner refuses any other hash.

### 5.4 Outcomes, written down now

| Calibration outcome | Meaning | Next |
|---|---|---|
| ready | The size principle is restored by equal-current afferent → MN efficacy | Confirmatory test |
| κ\* exists but "feedback ineffective" | Indirect paths recruit large MNs before direct paths recruit small ones | Report. The candidate becomes afferent → premotor efficacy (new amendment) |
| κ\* exists, effective, but O ≤ 0 in some cell | Recruitment is broadened but not size-ordered | Report. Confirm refused (D2) |
| no κ\* | C1/C2 fail at every κ ≥ 0.001 | Report. This would be a pipeline anomaly, since A8 passes at κ ≤ 0.02; investigate |

## 6. Confirmatory test

- **Hypothesis H2M:** with afferent → MN efficacy normalised so that feedback recruits MNs in size order, closed-loop proprioceptive feedback converts the open-loop swing/stance lag (0.81 cycle) into alternation, or couples the legs, and every control fails.
- **Set-up:** as stage 2B primary and A8 (A1–A7). Bilateral DNg100 at I = 400, tethered FlyBody, position actuation, frozen A7 bridge, r\* = 100 Hz, 4 hair-plate cells with none primary. Network W2M(κ\*₂M).
- **Runs:** replicates 0–63, five conditions (open loop, deafferented, closed loop, scrambled, mismatched), 896 runs.
- **Read-out and pass rules:** `experiment/paper2_stage2B_readout.py` unchanged (SHA-256 `9765fd82…`). Within-leg: circular mean lag in [0.375, 0.625] and R_all ≥ 0.8. Left/right: R ≥ 0.8, above the null's 99th percentile, and median PLV ≥ 0.8. Positive only if closed loop passes and all three feedback controls fail, cell by cell. The factorial rule of the stage 2B design §6.3 applies.
- **Description of any pass:** set by the F-checks in §8. A pass is never described as "emergent coordination" unless F1, F2, F15 and F17 all clear.

## 7. Descriptive arms (no pass claims)

1. **S, κ scan:** all 12 grid κ; closed loop, scrambled and mismatched; replicates 0–15; 4 cells; 2,304 runs. Runs whatever the calibration status. Open loop and deafferented are κ- and f-independent, because afferents are silent in open loop (§2.3). The stage 2B primary outputs serve for them, and a test checks this.
2. **P, size specificity (F17):** f permuted among contacted MNs within each leg (preserving each leg's multiset of factors, hence n_ref). Seeds `default_rng([20261119, p])`, p = 0–3. κ = κ\*₂M, closed loop, replicates 0–15, 4 cells; 256 runs. Runs if κ\*₂M exists. If permuted factors do as well as size-ordered ones, a pass is described as "redistributed afferent → MN weight", not "size principle".
3. **F2, CPG lesion:** only for cells that pass. Closed loop with E1 and E2 removed (Paper 1 lesion), replicates 0–63.
4. **Not in stage 2M:** (b) 9A gating, (c) load and (d) spiking. For (b), the prerequisites are: split the chordotonal channel into claw-like (SNpp50/51) and hook-like (SNpp39/41/59) transduction, decide whether the existing subtractive 9A input is kept or replaced, and find a verified PI magnitude.

## 8. Fake-positive risks and checks

| | Risk | Check (reported for every passing cell) |
|---|---|---|
| F1 | Reflex imposition (stage 2B design) | Direct-afferent share of each stance MN's suprathreshold input. If > 50%, the label is "monosynaptic-reflex-driven alternation" |
| F2 | Rhythm moved into the loop | Closed loop with E1+E2 lesioned (§7.3) |
| F15 | **Read-out composition.** The read-out averages z-scored traces of all active MNs. Feedback-recruited small MNs firing just above 1 Hz enter with equal weight, so a lag change can come from which MNs are in the mean, not from re-phasing | Recompute the lag using only MNs active in that replicate's open-loop run. If F15 fails while the read-out passes, the label is "alternation carried by feedback-recruited MNs" |
| F16 | z-scoring hides torque relevance | The module rate-sum trace (the bridge's own agonist sum) lag, reported beside the read-out |
| F17 | Any reweighting would do | Permuted-f arm P (§7.2) |
| F18 | Amplified weak contacts (f up to 5.6 on 5–27-synapse contacts, 21 MNs) dominate | F15 recomputed with the amplified MNs excluded. Lists the MNs that carry the effect |
| F19 | Latching mistaken for locking | Time at joint limit and fraction of time MNs are above 0.9 r_max, as in stage 2B diagnostics. Together with the PLV criterion, this catches high-R, low-PLV artefacts |
| F7 | Plumbing | Tests: f ≡ 1 equals the A8 κ network; κ = 1 with f ≡ 1 equals stage 2B; deafferented equals open loop bit for bit |
| F8 | Calibration leakage | Calibration only on 960–1023; AST ban on lag/phase/coherence code in the mechanism, calibration and runner modules; hashes checked by the runner |
| F9 | Unmeasurable replicates | Unchanged: they count as not passing |
| F20 | Right leg cannot show direct small-MN recruitment (§4) | T1R results reported with that caveat. The effectiveness label is per leg as well as per cell |

## 9. Amendment A10 (ready to paste)

**A10 (DATE), stage 2M: size-principle afferent efficacy. Pre-registered after the stage 2B-κ negative result and before any stage 2M run.**

*Why.* Stage 2B-κ found no single sensory scale at which afferent drive recruits small motor neurons while sparing large ones (A9). In the connectome, the included afferents contact 18 of 22 largest-tercile left motor neurons but only 5 of 23 smallest, and 0 of 21 smallest on the right. Synapse counts rise with motor-neuron size (log–log slope 0.74 left, 1.22 right), which cancels the rate model's own size scaling. In the A9 calibration, the first motor neurons recruited were largest-tercile coxa-stance neurons with 118–302 afferent synapses, and the recruitment-order index (Spearman of size against recruitment κ) was −0.15 to −0.26. Azevedo et al. 2020 (eLife 9:e56754) found the opposite order in flies: feedback PSPs are largest in slow motor neurons, and fast motor neurons are never driven to spike by feedback.

*Mechanism (ENGINEERED).*
- For each front-leg motor neuron j with n_j > 0 synapses from the 31 included afferents, the afferents' synapses onto j are multiplied by f_j = n_ref / n_j, with n_ref = 28 (the median over the 44 contacted motor neurons). Every contacted motor neuron therefore receives the same total afferent weight, 0.03 · κ · 28.
- κ multiplies all outgoing afferent synapses, as in A8.
- This changes 140 entries of the post × pre matrix (afferent columns × motor-neuron rows) and creates none.
- Central synapses, afferent → interneuron synapses (beyond κ), the frozen bridge (A7), the transduction and the read-out are unchanged.
- With f ≡ 1 the network equals the A8 κ network.

*κ\*₂M is fixed by an independent criterion before any coordination read-out.*
- Set-up and grid are exactly as A8: replicates 960–1023, DN drive off, no body, walking-clip stimulus (SHA-256 `c6ede5aa…` as in A9), r\* = 100 Hz, 4 hair-plate cells, κ ∈ {1, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.003, 0.001}.
- Criteria C1 and C2 are as A8, and κ\*₂M is the largest grid κ meeting both in ≥ 95% of replicates in all 4 cells.
- Recorded at κ\*₂M: the A8 effectiveness label.
- Recorded over the grid, per cell: the order index O_c. κ_rec,j is the smallest grid κ at which motor neuron j is active in ≥ 50% of replicates, and O_c = Spearman(s_j, κ_rec,j) over motor neurons recruited on the grid. The recruitment is "size-ordered" if O_c > 0 in all 4 cells, each with ≥ 10 recruited motor neurons.
- The confirmatory test runs only if κ\*₂M exists, the effectiveness label is "effective" or "mixed", and the recruitment is size-ordered. Otherwise the calibration is the stage's reported result.
- κ\*₂M, the factor table and the full calibration table are written to `data/paper2/stage2M_frozen.json`. Its SHA-256 is committed in an amendment before any test run.

*Test (confirmatory).* As A8's test (A1–A7: bilateral DNg100, position actuation, r\* = 100 Hz, 4 hair-plate cells, replicates 0–63, five conditions), with the stage 2M network at κ\*₂M. Read-outs, pass rules and the positive-verdict rule are unchanged (`experiment/paper2_stage2B_readout.py`, SHA-256 `9765fd82…`). A pass is described according to checks F1, F2 and F15–F20 of the stage 2M design.

*Descriptive only.*
- The κ scan: closed loop, scrambled and mismatched at every grid κ, replicates 0–15.
- A size-specificity control: f permuted within each leg, 4 fixed seeds, closed loop at κ\*₂M, replicates 0–15.
- No positive claim may rest on a κ other than κ\*₂M or on permuted factors.

*Accepted in advance.* The rule cannot create contacts, so right-leg smallest-tercile motor neurons cannot be recruited directly. Recruitment through interneurons stays scaled by κ alone. Azevedo et al. recorded tibia motor neurons under femoral chordotonal stimulation; as in A8, their constraint is applied to all front-leg motor neurons and afferent channels.

## 10. Implementation plan (new files only)

### 10.1 Files

| File | Contents |
|---|---|
| `experiment/paper2_stage2M_mechanism.py` | `afferent_mn_counts(W_counts, aff_rows, mn_rows)` → n_j. `equal_current_factors()` → {rows, n_j, f_j, n_ref}. `stage2M_network(net, kappa, factors, aff_rows)`: copies `net.W`; for CSR entries with column in A, multiplies data by κ, and additionally by f_j where the row j is in C; returns `dataclasses.replace(net, W=...)`. `permuted_factors(factors, seed)`, within leg |
| `experiment/paper2_stage2M_calibrate.py` | Reuses from `paper2_kappa_calibrate` (imported, unmodified): `GRID`, `CALIBRATION_REPLICATES`, `stimulus_states`, `afferent_current_sequence`, `motor_neuron_terciles`, `run_criteria`, `summarise`, `canonical_hash`. Has its own `_job` building `stage2M_network(..., dn_off)`, plus `order_index(table)`. Calls `paper2_stage2B_run.preflight()` and `paper2_kappa_run.kappa_preflight(path, "0e59ac6b…")`. Asserts the stimulus hash equals A9's. Writes `stage2M_frozen.json` and refuses to overwrite |
| `experiment/paper2_stage2M_run.py` | Arms `confirm`, `scan`, `permuted`. Copies the job structure of `paper2_kappa_run` with `_net` → `stage2M_network`. Preflight checks: A7 hash; A9 hash; 2M hash given on the command line; `git status` clean for the new modules and the 2M frozen file; and the status gate (§5.2). Records traces only |
| `experiment/paper2_stage2M_fchecks.py` | F1 (afferent share), F15 (open-loop MN set), F16 (rate-sum), F18 (amplified excluded), F19. Read-out-stage code; imports `paper2_stage2B_readout` and `leg_coordination` unchanged |
| `tests/test_paper2_stage2M.py` | See §10.2 |

### 10.2 Tests

1. **Factor table.** 44 contacted MNs (30 L, 14 R); n_ref = 28; 140 changed entries; f range [0.0927, 5.6]. Σ_{i∈A} W2M_ji = 0.84 κ for all j ∈ C (relative error < 1e-12).
2. **Identity.** With f ≡ 1, `stage2M_network` equals `kappa_network` for every grid κ (`indptr`, `indices` and `data` all equal). With κ = 1 and f ≡ 1, it equals `Network.W`.
3. **Locality.** Every entry outside (rows M × columns A) is identical to the κ network. The sparsity pattern and signs are unchanged. The cached base network is not mutated.
4. **Open-loop invariance.** A 0.3 s open-loop run on replicate 960 with the 2M network equals the base network bit for bit.
5. **Guards.** Replicates outside 960–1023, κ off the grid, overwriting, a wrong A9 hash and a stimulus-hash mismatch are all refused.
6. **Order index.** On synthetic tables it gives +1 for size order and −1 for reverse order. It is undefined below 10 recruited MNs. Applied to the A9 audit file, it reproduces the §2.2 values.
7. **AST ban.** The mechanism, calibration and runner modules do not import or call lag, phase, PLV or coherence functions (the pattern from `tests/test_paper2_kappa.py`).
8. **Runner preflight.** A wrong 2M hash, dirty guarded paths, and a confirm arm whose status is not "ready" are all refused.
9. **Permutation.** Each leg's multiset of factors is preserved; the result is seed-reproducible and differs from the identity.
10. **Frozen files untouched.** The A7 and A9 files and every `code_sha256` entry in the A9 file verify.

### 10.3 Order of work

1. Owner decisions (§12).
2. Commit this design and A10.
3. Write the mechanism module and tests 1–4, 9, 10.
4. Write the calibration and tests 5–7.
5. Run the calibration.
6. Commit the frozen file and A11 with its hash.
7. Write the runner and test 8.
8. Run confirm (if ready), scan and permuted.
9. Write the F-checks, then run the read-out and F-checks.
10. Write the results record `paper2_stage2M.md`.

### 10.4 Hashes verified today (19 November 2026)

- `stage2B_bridge_frozen.json`: `91bc54c3…fac25` (matches A7).
- `stage2Bk_kappa_frozen.json`: `0e59ac6b…160a` (matches A9).
- A9 `code_sha256` entries (`rate_stepper.py`, `proprio_transduction.py`, `flybody_tethered.py`, `paper2_cosim.py`, `paper2_kappa_calibrate.py`): all match.

## 11. Runtime estimate

Wall times on 9 workers are scaled from measured times of the same-sized jobs. The network size and number of non-zeros are unchanged, so per-run cost should be the same.

| Job | Runs | Basis | Estimate |
|---|---:|---|---:|
| Calibration | 3,072 | A9: 7,706 s | ~2 h 10 min |
| Confirm | 896 | A8 confirm: 2,076 s | ~35 min |
| Scan | 2,304 | A8 scan: 5,757 s | ~1 h 36 min |
| Permuted | 256 | 2.5 s/run (from scan) | ~11 min |
| F2 lesion (only if a cell passes) | 64 per cell | 2.3 s/run | ~2.5 min per cell |
| Tests | — | guess | 2–5 min |
| **Total** | | | **~4.5 h** (calibration only, if not ready: ~2 h 10 min) |

## 12. Decisions needing the owner

- **D1. Form of mechanism (a).** Equal-current (recommended: follows from the principle and adds no parameter), the exponent family s^−γ (extra fitted γ), or input-fraction normalisation.
- **D2. Gate on the confirmatory arm.** Run only if status is "ready" (recommended; a "feedback ineffective" κ\* would only repeat 2B-κ), or run regardless, as A8 did.
- **D3. Amplification.** Uncapped f (recommended), or capped f ≤ 1, which only scales down and never boosts the 5–27-synapse contacts. The capped variant could also be run as a second, descriptive calibration (+2 h 10 min).
- **D4. Scope of the Azevedo constraint.** All MNs and channels, as in A8 (recommended, since the latching pathway is hair plate → coxa), or only tibia MNs and the chordotonal channel, closer to Azevedo's preparation.
- **D5. Correction to the 2B-κ record.** Add a dated note to `paper2_stage2Bk.md`. 9A inhibition in the rate model is a near-veto, tonic and DNg100-driven on 2 hook-like afferents (via IN09A012_L), and absent on the others in open loop. It is not uniformly "weak".
- **D6. Defer (b) 9A gating** to a later stage that first splits claw and hook transduction and finds a verified PI magnitude.
- **D7. Stage name and file names** ("2M", `stage2M_*`).

## 13. Guesses and unverified items

- The connectivity-based "hook-like" identity of SNpp39/41/59 and "claw/club-like" identity of SNpp50/51 are inferences from 9A-input fractions, not a verified type mapping.
- The SNpp41 veto estimate uses nominal parameters (a = 1, θ = 7.5, r_max = 200). Replicate-specific values differ.
- The likely κ\*₂M range (0.05–0.2) and whether it will be effective are guesses. Indirect recruitment of a Tergotr. MN at κ = 0.1 in A9 suggests C1 may bind before small MNs are recruited.
- The equal-current derivation is my reasoning from the rate model's size scaling and Azevedo et al.'s observations. No source states it.
- The input-fraction convention (a″) has no source in the literature file.
- r\* = 100 Hz remains an assumption (A8).
- Runtimes are scaled from earlier runs, not measured for stage 2M.

## References (VERIFIED in the literature file)

- Azevedo AW et al. 2020. A size principle for recruitment of Drosophila leg motor neurons. eLife 9:e56754.
- Dallmann CJ et al. 2025. Selective presynaptic inhibition of leg proprioception in behaving Drosophila. Nature 647:445–453.
- Pratt BG et al. 2026. Proprioceptive limit detectors contribute to sensorimotor control of the Drosophila leg. Nature Communications 17.
