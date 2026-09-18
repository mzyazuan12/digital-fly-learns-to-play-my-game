"""Birth a persistent digital Drosophila individual.

Birth is not resurrection. MaleCNS is a chemically fixed EM reconstruction.
This procedure instantiates a NEW individual from that anatomy and lets
neural, bodily, neuromodulatory and learned state evolve from t = 0.

If the individual directory already exists, load it. Do not reconstruct
from defaults.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from organism.config import BIRTH_DEFINITION, MODEL_VERSION
from organism.fly import VirtualFly

ROOT = Path(__file__).resolve().parents[1]
INDIVIDUALS = ROOT / "individuals"


def birth_fly(
    individual_id: str,
    *,
    seed: int | None = None,
    connectome: str = "synthetic",
    directory: Path | None = None,
) -> dict:
    path = directory or (INDIVIDUALS / individual_id)
    existed = (path / "identity.json").exists() and (
        (path / "neural_state.npz").exists() or (path / "brain.npz").exists()
    )
    fly = VirtualFly.birth(
        individual_id,
        seed=seed,
        connectome=connectome,
        directory=path,
        legacy_scaffold=False,
    )
    report = {
        "individual_id": fly.identity.fly_id,
        "path": str(path),
        "seed": fly.identity.seed,
        "loaded_existing": existed,
        "model_version": MODEL_VERSION,
        "connectome_dataset": fly.identity.connectome_dataset,
        "neurons": fly.connectome.n,
        "edges": fly.connectome.n_edges,
        "motor_mode": fly.motor_mode.value,
        "motor_fidelity_level": fly.motor_fidelity_level,
        "policy": fly.policy.as_dict(),
        "legacy_scaffold": fly.legacy_scaffold,
        "consciousness_claimed": False,
        "birth_definition": BIRTH_DEFINITION,
        "files": sorted(p.name for p in path.iterdir()),
    }
    (path / "birth_report.json").write_text(json.dumps(report, indent=2) + "\n")
    validation = path / "dataset_validation.txt"
    if validation.exists():
        print(validation.read_text(), flush=True)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--individual", required=True, help="Persistent ID, e.g. fly_001")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--connectome", default="synthetic", choices=("synthetic", "malecns"))
    parser.add_argument("--directory", type=Path, default=None)
    args = parser.parse_args(argv)
    report = birth_fly(
        args.individual,
        seed=args.seed,
        connectome=args.connectome,
        directory=args.directory,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
