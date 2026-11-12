# Reproducing a connectome-derived walking rhythm on the male *Drosophila* CNS: validating the neural foundation of a persistent digital fly

**Paper 1 of the virtual *Drosophila* project** · Zayan ([@mzyazuan12](https://github.com/mzyazuan12)) · draft of 11 November 2026

Repository: [digital-fly-learns-to-play-my-game](https://github.com/mzyazuan12/digital-fly-learns-to-play-my-game). Every number in this paper comes from a file under `outputs/` that a command in [Reproducibility](#9-reproducibility) regenerates.

---

## Abstract

The long-term goal of this project is a persistent, embodied digital fruit fly: one individual with a body, senses, a nervous system grounded in measured anatomy, internal state, memory and learning, living continuously in a simulated world. Before any of that is trusted, the neural layer has to be shown to produce real biology and not artefacts of our own code. This paper reports that validation for walking, the first behaviour the body needs.

We take the published connectome-simulation result of Pugliese et al. (2025/2026), which found that stimulating the descending neuron DNg100 in a firing-rate model of the front-leg nerve-cord network produces a ~11 Hz rhythm in leg motor neurons through a small central pattern generator (CPG). We reproduce it in three steps, each with an explicit pass/fail gate. (1) Our port of the authors' rate equation reproduces their own MANC traces (mean correlation 0.99996). (2) On the authors' exact 4,310-neuron male-CNS (MaleCNS v1.0) subnetwork, our parameter sampler and size normalization reproduce every one of their 4 × 1,024 × 4,310 sampled neuron parameters to float32 precision, and our integrator, given their parameters, reproduces their published traces replicate by replicate. (3) Driving MaleCNS DNg100_R (body 10056) at the published CNS amplitude (I = 400) in 1,024 independently sampled networks produces rhythmic leg motor output in RESULT_FRACTION of replicates at RESULT_FREQ Hz, with the identified CPG interneurons E1, E2, I1 and I2 oscillating at the same frequency. Single-cell-type lesions, a sign- and class-preserving shuffled-wiring null, left/right/bilateral drive and an amplitude sweep characterise which parts of the result depend on the measured wiring.

Along the way we caught and corrected several errors that would have produced fake biology: a body-ID namespace collision between two connectomes, a wrong I2 cell type, a size-normalization denominator that silenced DNg100, and a stimulus amplitude that had been borrowed from the wrong dataset. We describe each, because the discipline that caught them is the main methodological contribution.

We make no claim about consciousness. We discuss what this result does and does not establish for the larger organism, and which properties of a digital organism could later be measured.

---

## 1. Introduction

### 1.1 The organism we are trying to build

The project's aim is a single persistent digital *Drosophila melanogaster*: a body (FlyBody/NeuroMechFly, simulated in MuJoCo), senses (vision, touch, proprioception), a nervous system whose topology is the measured male CNS connectome, internal drives and neuromodulatory state, memory and learning, all running as one continuous loop of perception, internal processing, action, consequence and memory. The organism is not scripted. There is no `go_to_fruit()` and no `play_shiritori()`; a game the fly later encounters, such as the word game Shiritori on a simulated monitor and keyboard, is one experience in its life, not its purpose.

The interesting scientific question is what organism-level behaviour can emerge from such an integrated system: whether one persistent individual can explore a world without a task, form stable preferences and spatial memories, acquire new skills through interaction, and carry them across unrelated environments. A further, much harder question is whether a system like this could ever have properties relevant to consciousness. We return to that in [Section 7](#7-on-consciousness).

### 1.2 Why validate the neural layer first

A connectome gives which neurons exist, which connect, how many synapses each connection has, and a predicted transmitter. It does not give membrane time constants, synaptic efficacy, receptor kinetics, neuromodulation or the live state of the animal. Any simulation fills those gaps with assumptions. A simulated fly that walks can therefore walk for the wrong reason: an engineered oscillator, a tuned gain, a mislabelled cell, or a normalization constant that happens to push activity into a convenient range. A digital organism built on a neural layer like that would be a puppet with a connectome painted on it.

So the project adopted a rule: no neural result is used to drive the body until it reproduces a published, independently derived biological result on the same data, with every source of difference between our implementation and the original either eliminated or measured.

### 1.3 The target result

Pugliese et al. (2025/2026) built a firing-rate model of the neurons controlling one front leg in the male adult nerve cord connectome (MANC), screened descending neurons for those whose activation produces rhythmic leg motor activity, and found DNg100 as the strongest driver. Synthetic pruning reduced the circuit to a core CPG of identified interneurons: excitatory E1 (IN17A001) and E2 (INXXX466), inhibitory I1 (IN16B036), with E3 (IN19B012) and I2 (IN19A007) also involved. They predicted that DNg100 drives walking and confirmed rhythmic leg movement with optogenetics in behaving flies. They then reran the model on three other connectomes, including the male CNS (mCNS, the dataset this project uses), and report that DNg100 produced rhythmic motor activity in all four.

That makes it an ideal first gate. It is (i) a causal claim about walking, the first behaviour the body needs; (ii) made on the same connectome we use; (iii) backed by open code (`smpuglie/Pugliese_2026`, commit `10e7661`) and open simulation data (Zenodo record 22260924); and (iv) independent of every assumption in our own codebase.

### 1.4 Contributions

1. A port of the authors' rate model that reproduces their MANC traces at r = 0.99996 and their mCNS sampled parameters to float32 precision.
2. Resolution of the last open unknown in the mCNS reproduction (the stimulus amplitude), from the paper's Methods and then confirmed against the authors' archived run configuration.
3. A 1,024-replicate reproduction of DNg100-driven rhythm on the authors' exact 4,310-neuron mCNS network with an independent random seed, plus a replicate-by-replicate trace comparison against the authors' own 1,024 published runs.
4. Perturbation arms (lesions, shuffled-wiring null, drive side, amplitude sweep) that the original mCNS analysis did not report.
5. A record of the identity, namespace and normalization errors caught on the way, and the gates that now prevent them.

---

## 2. Related work

**Connectomes.** Whole-brain (FlyWire; Dorkenwald et al. 2024; Schlegel et al. 2024), nerve-cord (MANC, Takemura et al. 2024, Marin et al. 2024, Cheong et al. 2024; FANC, Azevedo et al. 2024) and now whole-CNS (male CNS v1.0, Janelia FlyEM; Berg et al. 2025) reconstructions make connectome-constrained simulation possible across the entire fly nervous system. This project uses MaleCNS v1.0: 166,700 retained neurons and 25,582,938 directed edges (124,177,617 synaptic contacts) under our inclusion rule.

**Connectome-constrained simulation.** Shiu et al. (2024) simulated the whole FlyWire brain as leaky integrate-and-fire (LIF) neurons with uniform parameters and predicted sensorimotor pathways that were then confirmed experimentally. Lappalainen et al. (2024) trained connectome-constrained models of the visual system that predict neural responses. Pugliese et al. (2025/2026) used a firing-rate model of nerve-cord circuits to find the walking CPG reproduced here.

**Bodies.** NeuroMechFly (Lobato-Rios et al. 2022) and NeuroMechFly v2 / FlyGym (Wang-Chen et al. 2024) provide an anatomically detailed, physics-simulated fly body with sensory inputs; Vaxenburg et al. (2025) built a whole-body fly model in MuJoCo (Todorov et al. 2012) trained to walk and fly.

**Embodied connectome demonstrations.** In early 2026 the company Eon Systems publicised a FlyWire-based brain model coupled to a NeuroMechFly body in MuJoCo, showing walking, grooming and feeding-like behaviour. As of this writing we found press coverage and company material but no peer-reviewed paper, and coverage notes that brain activity was translated into higher-level body commands rather than driving individual muscles. We treat it as closely related prior work, not as a validated result.

**Position of this work.** This paper is not new biology. It is a validation study: it establishes that the neural substrate of a long-lived embodied organism reproduces a published causal walking result on the organism's own connectome, with every implementation choice checked against the original. What is new is the end-to-end check on mCNS (parameter-level and trace-level agreement, perturbation arms the original did not report) and the provenance discipline that makes later embodied results interpretable.

---

## 3. Data and identities

### 3.1 MaleCNS v1.0

Three official files define the topology: `connectome-weights-male-cns-v1.0-minconf-0.5.feather`, `body-annotations-male-cns-v1.0-minconf-0.5.feather` and `body-neurotransmitters-male-cns-v1.0.feather`, from male-cns.janelia.org. Synapse point clouds and SWC skeletons are used for anatomy checks only. Graph orientation is `W[pre, post]`; a spot check confirms the direction (MaleCNS 10045 → 800316 has 20 contacts; the reverse has none).

### 3.2 Identities, and the errors they hid

Cell identity is taken from the official annotation `type` column joined on `bodyId`, never from a derived table, an integer that happens to match, or a pooled type-level ROI total.

- **DNg100.** In MaleCNS, `type == "DNg100"` gives exactly two neurons: body 10045 (DNg100_L) and body 10056 (DNg100_R). Pugliese's MANC stimulus is MANC body 10093 (matrix index 31). In the MANC table, integer 10056 is a different cell, vMS16, a GABAergic neuron, and MaleCNS body 10093 is Am1. An earlier version of our pipeline joined across datasets on the integer and was archived as `cpg_mapping.INVALID_pre_namespace_fix.json`. Body IDs are now namespaced by dataset.
- **I2.** I2 is IN19A007 (six copies across T1–T3, predicted GABAergic), as in Pugliese et al. An earlier assignment used IN19B007, a different two-neuron type. It is now locked in `cpg_mapping.json` with provenance.
- **Leg and side.** Each CPG cell's segment and side come from its own synapses (PreSyn + PostSyn in `LegNp(T#)(L/R)`, confidence threshold 0.7, no fallback), never from soma position or type-level pooling. All six cells of each core type are assigned this way. An earlier mapping that used other signals is archived as `cpg_mapping.INVALID_pre_roi.json`.
- **Which DNg100 to drive.** MaleCNS 10056 has `somaSide = R`, but its curated MANC correspondence is MANC 10093 (the authors' stimulated cell) and its leg-neuropil innervation is on the left. It was pre-registered as the stimulated cell before the authors' configuration was available. That configuration later confirmed it ([Section 5.2](#52-the-authors-archived-run-confirms-the-stimulus)).

### 3.3 The authors' 4,310-neuron mCNS network

The authors ship their mCNS simulation network in the repository: `wTable_20260210_vncRoisOnly.csv` (4,310 neurons: 2,493 cholinergic, 1,011 GABAergic, 606 glutamatergic, 193 unclear, 4 histaminergic, 2 serotonergic, 1 blank) and `W_20260210_vncRoisOnly.csv` (signed synapse counts, 118,920 non-zero entries: 66,544 excitatory, 52,376 inhibitory). It contains 1,236 descending, 2,378 intrinsic, 328 ascending, 232 sensory and 130 motor neurons, with two copies of each core CPG type (one per front leg). Every row is in the simulation; transmitter is not a membership filter, and W is already signed.

We also rebuilt the population from MaleCNS v1.0 using the paper's stated selection rules (front-leg motor neurons, their premotor partners, descending inputs, a ≥5-synapse VNC floor). That gives 4,301 neurons, intersecting the shipped table in 4,295 (15 shipped-only, 6 rebuild-only). We report it as a methods audit only: the reproduction uses the authors' shipped table, because the size normalization depends on exactly which neurons are in it.

---

## 4. Model

### 4.1 Rate equation

Each neuron *i* has a firing rate *R*<sub>*i*</sub>(*t*) obeying the authors' half-rectified tanh rate model:

```text
τ_i dR_i/dt = −R_i + max(0, r_max,i · tanh( (a_i / r_max,i) · (Σ_j W̃_ij R_j + I_i(t) − θ_i) ))
```

with W̃ = 0.03·max(Wᵀ, 0) + 0.03·min(Wᵀ, 0) (postsynaptic × presynaptic, from signed synapse counts), and per-neuron parameters drawn from truncated normals (lower bound 0): τ ~ N(20 ms, 2 ms), *a* ~ N(1, 0.1), θ ~ N(7.5, 0.6), *r*<sub>max</sub> ~ N(200 Hz, 10 Hz). Each replicate is one independent draw of all 4 × 4,310 parameters, using the authors' sampler (`jax.random.split(PRNGKey(seed), 5)` and an inverse-CDF truncated normal). The stimulus is a tonic current *I* on the driven neuron from 20 ms to 1.999 s; simulations last 2 s, sampled every 1 ms. We integrate with SciPy's RK45 (Dormand–Prince) at the authors' tolerances (rtol 2×10⁻⁶, atol 5×10⁻⁹); the authors use Diffrax's Dopri5, the same method family.

### 4.2 Size normalization

Larger neurons need more input to reach threshold. The authors scale each neuron's gain and threshold by its volume relative to the median of the simulated population, *s*<sub>*i*</sub> = size<sub>*i*</sub> / nanmedian(size): *a*<sub>*i*</sub> ← *a*<sub>*i*</sub> / *s*<sub>*i*</sub> and θ<sub>*i*</sub> ← θ<sub>*i*</sub> · *s*<sub>*i*</sub>. For mCNS, size is the neuPrint `volume` property. Over the shipped 4,310 neurons the median is 998,603,428.5 (every row has a positive size). DNg100_R is 16.6 times the median, because its brain arbor is part of the reconstruction, so its mean threshold becomes about 124. This is why the CNS datasets need a larger DNg100 drive than MANC.

### 4.3 Stimulus amplitude

The authors' Methods state: "For DNg100 activation, I_stim = 250 in MANC, 150 in FANC, and 400 in the two CNS datasets". The MANC value matches the shipped `DNg100_Stim.yaml`, which cross-checks the sentence. We use I = 400 as `stimulus_source = authors_mcns`; any other amplitude is labelled `amplitude sweep`.

### 4.4 Read-outs (identical to the authors' notebooks)

- **Motor neurons:** wTable rows with a non-null `motor module` (129 neurons).
- **Active:** rate > 0.01 Hz at any time after 250 ms.
- **Motor rhythmicity score:** the authors' `compute_oscillation_score` (autocorrelation-peak based, prominence 0.05) on *R*[:, 250:], averaged over active motor neurons; a replicate is **rhythmic** if the score is ≥ 0.5, the authors' `oscillation_threshold`.
- **Frequency:** the authors' per-neuron frequency estimate averaged over active neurons, in cycles per sample, divided by *dt*.
- **CPG roles:** the same score on the active copies of E1, E2, E3, I1 and I2.

Published walking-frequency ranges are annotations only: no read-out restricts or selects a frequency band.

### 4.5 Experimental arms

All arms use the same 4,310-neuron network and the same per-replicate parameter draws (seed 1, replicate *k* of a 1,024-replicate block), so arms differ only in the intended manipulation.

| Arm | Drive | Manipulation | Replicates |
|---|---|---|---:|
| intact | 10056, I = 400 | none | 1,024 |
| E1 / E2 / E3 / I1 / I2 removed | 10056, I = 400 | both copies of the type removed from W (rows and columns), as the authors' `removeNeurons` | 64 each |
| shuffled wiring | 10056, I = 400 | postsynaptic targets permuted within DN-exc, DN-inh, interneuron-exc, interneuron-inh and MN groups (the authors' `shuffle_W` groups), new permutation per replicate | 64 |
| drive 10045 | 10045, I = 400 | other DNg100 | 64 |
| drive both | 10045 + 10056, I = 400 each | bilateral | 64 |
| amplitude sweep | 10056, I = 150 … 600 | none | 32 per level |

### 4.6 Numerical engineering

W is 0.6% dense, so integration uses a CSR sparse matrix product. On replicate 0 the sparse and dense integrations agree at r = 0.9999987 (max difference 0.055 Hz, at the level of adaptive-step error); a unit test checks agreement within integrator tolerance. The authors' oscillation score masks inactive neurons out of its average, so we evaluate it on active rows only, which gives identical values (checked to 10⁻⁷).

### 4.7 Gates

Every experiment directory is new (existing outputs are refused), records the git commit, data SHA-256 hashes and stimulus provenance, and preserves failures. Scientific permission flags default to false; a full-MaleCNS run, body coupling and lesions each require their prerequisite gate to have passed. Rates must be finite and within the firing-rate cap. Recruitment of the required populations is checked before any spectral analysis, and an FFT on an unrecruited population is refused.

---

<!-- RESULTS -->
