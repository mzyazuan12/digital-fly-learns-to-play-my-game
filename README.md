# virtual Drosophila

One persistent embodied fruit fly. MaleCNS v1.0 is the nervous-system
topology. FlyBody (Turaga lab, via FlyGym 2.x) is the body. MuJoCo is the
physics. Shiritori is **one experiment** you can later run on the same
individual. It is not the fly.

```text
environment → senses → MaleCNS → identified DNs → MotorBridge
          → FlyBody walking/flight controllers → MuJoCo → senses
```

There is no `world.reward` in autonomous mode. Internal state (arousal,
hunger, fatigue, …) modulates identified neurons; it does not call
`find_food()`. The brain never receives object coordinates, key identities,
or correct answers.

Status: FlyBody gait is verified. Default view is the organism: MaleCNS
activity on real soma coordinates, Google Neuroglancer EM, autonomous
locomotion, Shiritori on the desk. Walk/Stand buttons exist only at `/gait`.

```sh
python -m sim3d.serve --connectome malecns
# http://127.0.0.1:8765/       organism (brain + body + Shiritori)
# http://127.0.0.1:8765/gait   physics lab, manual gait commands
# http://127.0.0.1:8765/desk   Shiritori desk
```

## Body (optional)

The anatomically detailed body is FlyGym 2.x `FlyBody`. There is no
decorative-mesh fallback for walking. Mock unicycle trajectories are not
FlyBody results. The HybridTurningController is an engineered VNC/muscle
surrogate, not recovered motor circuitry.

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
organism/     VirtualFly, physiology, MotorBridge, body, persistence
worlds/       living room, arena, stimulus, Shiritori workstation
flybrain/     MaleCNS loader, LIF, plasticity
shiritori/    game rules used by an experiment / old letter curriculum
training/     letter-grid curriculum (does not bypass organism persistence)
sim3d/        MuJoCo gait viewer at / ; living room at /habitat (root motion off)
tests/
ASSUMPTIONS.md
```

## Honesty

- The connectome is real. The LIF constants are not.
- Identified DN types (DNp09, DNa02, …) are resolved from annotations.
  Mapping their rates onto the walking controller is engineered.
- The miniature CI graph uses the same type names with random recurrent
  edges. It is **not** wired for phototaxis.
- See `ASSUMPTIONS.md`.
# digital-fly-learns-to-play-my-game
