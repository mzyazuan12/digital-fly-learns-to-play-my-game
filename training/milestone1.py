"""Milestone 1: load the full MaleCNS graph, stimulate, propagate, stay in RAM.

This is the gate before any Shiritori training. Success means we are holding
the fly connectome in code, not that the fly can play.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from flybrain.loader import EXPECTED_CONTACTS, EXPECTED_EDGES, EXPECTED_RETAINED, load_connectome
from flybrain.network import LIFNetwork, LIFParams
from flybrain.neurons import shiu_coupling
from flybrain.populations import build_interface

ROOT = Path(__file__).resolve().parents[1]


def _rss_mb() -> float:
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024 * 1024) if usage > 10**8 else usage / 1024


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--window-ms", type=float, default=50.0)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "milestone1.json")
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    connectome = load_connectome(rebuild=args.rebuild, progress=True)
    stats = connectome.stats()
    print("graph statistics:")
    for k, v in stats.items():
        print(f"  {k:24s} {v:,}" if isinstance(v, (int, np.integer)) else f"  {k:24s} {v}")

    ok_n = stats["neurons"] == EXPECTED_RETAINED
    ok_e = stats["directed_edges"] == EXPECTED_EDGES
    ok_c = stats["synaptic_contacts"] == EXPECTED_CONTACTS
    print(f"expected neurons {EXPECTED_RETAINED:,}: {'YES' if ok_n else 'NO'}")
    print(f"expected edges   {EXPECTED_EDGES:,}: {'YES' if ok_e else 'NO — inspect importer'}")
    print(f"expected contacts {EXPECTED_CONTACTS:,}: {'YES' if ok_c else 'NO — inspect importer'}")

    interface = build_interface(connectome, seed=0)
    print("BCI interface (engineered, not biology):")
    for k, v in interface.notes.items():
        print(f"  {k}: {v}")

    params = LIFParams(dt=args.dt)
    net = LIFNetwork(connectome, params=params, seed=0)
    # Stimulate the first letter-A prefix slot.
    targets = interface.letter_prefix[0]  # slot 0, letter A
    net.inject(targets, 30.0)
    t_sim = time.perf_counter()
    counts = net.run_ms(args.window_ms)
    sim_s = time.perf_counter() - t_sim
    stimulated_spikes = int(counts[targets].sum())
    downstream = int(counts.sum()) - stimulated_spikes
    dn_spikes = int(counts[interface.descending_indices].sum())
    active = int(np.count_nonzero(counts))
    target_mask = np.zeros(connectome.n, dtype=bool)
    target_mask[targets] = True
    other_g = net.g[~target_mask]
    other_v = net.v[~target_mask]
    n_with_current = int(np.count_nonzero(np.abs(other_g) > 1e-8))
    n_depolarized = int(np.count_nonzero(np.abs(other_v - params.v_rest) > 1e-4))
    max_abs_g = float(np.max(np.abs(other_g))) if other_g.size else 0.0

    result = {
        "neurons": stats["neurons"],
        "directed_edges": stats["directed_edges"],
        "synaptic_contacts": stats["synaptic_contacts"],
        "matches_expected_neurons": ok_n,
        "matches_expected_edges": ok_e,
        "matches_expected_contacts": ok_c,
        "stimulated_neurons": int(targets.size),
        "stimulated_spikes": stimulated_spikes,
        "total_spikes": int(counts.sum()),
        "downstream_spikes": downstream,
        "active_neurons": active,
        "descending_spikes": dn_spikes,
        "downstream_neurons_with_synaptic_current": n_with_current,
        "downstream_neurons_off_rest": n_depolarized,
        "max_abs_downstream_g": max_abs_g,
        "window_ms": args.window_ms,
        "dt_ms": args.dt,
        "sim_wall_seconds": round(sim_s, 3),
        "import_and_sim_seconds": round(time.perf_counter() - t0, 3),
        "rss_mb": round(_rss_mb(), 1),
        "lif_coupling": shiu_coupling(params),
        "interface": interface.notes,
        "neural_dynamics_validated": False,
        "learning_demonstrated": False,
        "milestone1_ok": bool(
            ok_n
            and stimulated_spikes > 0
            and (downstream > 0 or n_with_current > 0)
            and _rss_mb() < 14000
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["milestone1_ok"]:
        return 1
    if not ok_n:
        return 2
    print("milestone 1 complete: full graph loaded and one stimulation window ran in-process.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
