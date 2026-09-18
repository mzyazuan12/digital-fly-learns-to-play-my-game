"""Birth a persistent digital Drosophila individual.

Birth is not resurrection. MaleCNS is a chemically fixed EM reconstruction.
This procedure instantiates a NEW individual from that anatomy and lets
neural, bodily, neuromodulatory and learned state evolve from t = 0.
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
    seed: int = 0,
    connectome: str = "synthetic",
    directory: Path | None = None,
) -> dict:
    path = directory or (INDIVIDUALS / individual_id)
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
        "seed": seed,
        "model_version": MODEL_VERSION,
        "connectome_dataset": fly.identity.connectome_dataset,
        "neurons": fly.connectome.n,
        "edges": fly.connectome.n_edges,
        "motor_mode": fly.motor_mode.value,
        "legacy_scaffold": fly.legacy_scaffold,
        "consciousness_claimed": False,
        "birth_definition": BIRTH_DEFINITION,
        "files": sorted(p.name for p in path.iterdir()),
    }
    (path / "birth_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--individual", required=True, help="Persistent ID, e.g. fly_001")
    parser.add_argument("--seed", type=int, default=0)
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
