# virtual Drosophila

One persistent embodied fruit fly. MaleCNS v1.0 is the nervous-system
topology. FlyBody (Turaga lab, via FlyGym 2.x) is the body. MuJoCo is the
physics. Shiritori is **one experiment** you can later run on the same
individual. It is not the fly.

**Primary objective:** measure how much autonomous organism-level behavior
can emerge from an embodied model grounded in the real Drosophila nervous
system. Looking alive is not the metric. A hard-coded random-walk fly can
look more alive than a scientifically grounded one.

**Birth** means: instantiate a new persistent digital individual from
measured anatomy, initialize uncertain physiology explicitly, and let
neural, bodily and learned state evolve through experience. MaleCNS is a
chemically fixed EM reconstruction. It does not contain the specimen's
live mind, and this project does not claim to resurrect it.

```text
environment → biological receptors → MaleCNS brain + VNC
          → premotor / motor neurons → (today: FlyGym CPG)
          → FlyBody → MuJoCo → senses
```

There is no `world.reward` in autonomous mode. Metabolic state changes
neuromodulators; neuromodulators change identified circuits; they do not
call `find_food()` or `walk()`.

Default path: **no walking-bout timer, no walking_drive fallback, no
scripted groom/flight scheduler.** Those remain behind `FLY_LEGACY_SCAFFOLD=1`
for comparison only.

```sh
python birth_fly.py --individual fly_001
python -m sim3d.serve --connectome malecns
# http://127.0.0.1:8765/       organism (brain + body + Shiritori)
# http://127.0.0.1:8765/gait   physics lab, manual gait commands
# http://127.0.0.1:8765/desk   Shiritori desk
```

First experiment: DNp09 current should initiate walking through the CPG;
silencing DNp09 should stop it. If spontaneous walking does not emerge
without a timer, that is a result.

Next experiment: keep DNp09, and ask whether DNg100 / DNb08 / oDN1 and
the published VNC walking CPG (E1=IN17A001, E2=INXXX466, I1=IN16B036,
E5=INXXX464) exist and whether **sustained DNg100 current** produces
per-leg E1/E2/inhibition/MN oscillations. Joints still use FlyGym in
`MODE_ENGINEERED_CPG`. `MODE_NEURAL_CPG` records that motif and does not
actuate FlyBody yet.

```sh
.venv/bin/python -m pytest tests/test_neural_walk.py tests/test_walking_dn_investigator.py tests/test_dng100_cpg_rhythm.py tests/test_roi_innervation.py -q
.venv/bin/python scripts/cache_malecns_roi_innervation.py
.venv/bin/python -m experiment.dng100_cpg_rhythm --connectome malecns --stim left_vnc
```

Authors' MANC rate-ODE (conda `vnc-sim`, their Hydra entrypoint, not a reimplementation):

```sh
python src/run_hydra.py experiment=DNg100_Stim experiment.n_replicates=128 experiment.batch_size=16 paths=mac
```

## Body (optional)

The anatomically detailed body is FlyGym 2.x `FlyBody`. There is no
decorative-mesh fallback for walking. Mock unicycle trajectories are not
FlyBody results. The HybridTurningController is an engineered VNC/muscle
surrogate (`MODE_ENGINEERED_CPG`), not recovered motor circuitry.
`MODE_HYBRID_VNC`, `MODE_NEURAL_CPG`, and `MODE_NEURAL_MOTOR` exist so we
can replace FlyGym CPG outputs with identified VNC rhythm and then
motor-neuron activity without doing it all at once. `MODE_NEURAL_CPG`
does not move joints yet.

## MaleCNS files

Three official files (already in this checkout, or download from
https://male-cns.janelia.org/download/):

```
connectome-weights-male-cns-v1.0-minconf-0.5.feather
body-annotations-male-cns-v1.0-minconf-0.5.feather
body-neurotransmitters-male-cns-v1.0.feather
```

Synapse XYZ tables and SWC skeletons are optional and gitignored:

```
python scripts/download_malecns_anatomy.py
```

Do not download raw EM imagery for runtime simulation.

```sh
python -m training.milestone1   # load the full graph, one stimulation window
```

## Tests

```sh
python -m pytest tests -q --ignore=tests/test_connectome_full.py
python -m pytest tests/test_connectome_full.py -m full
```

## Layout

```
organism/     VirtualFly, birth, neuromodulation, MotorBridge, body
worlds/       living room, arena, stimulus, Shiritori workstation
flybrain/     MaleCNS loader, mixed LIF/graded dynamics, mushroom body
shiritori/    game rules used by an experiment / old letter curriculum
training/     letter-grid curriculum (does not bypass organism persistence)
sim3d/        organism view at / ; gait lab at /gait
tests/
ASSUMPTIONS.md
```

## Honesty

- The connectome is real. Membrane constants are not.
- Identified DN types (DNp09, DNg100, DNa02, …) are resolved from annotations.
  Mapping their rates onto the walking CPG is engineered. The published VNC
  walking CPG is traced, not executed.
- VNC premotor cells use graded/rate dynamics by literature, not because
  MaleCNS measured that for each cell.
- Learning is KC→MBON, not a global reward rule on every synapse.
- The miniature CI graph uses the same type names with mostly random
  extra edges. It is **not** wired for phototaxis.
- No internal variable is labeled consciousness.
- See `ASSUMPTIONS.md`.
