"""Run the Pugliese et al. 2025 DNg100_Stim code as a positive control.

This is the authors' MANC T1 rate-ODE simulator, not MixedDynamicsNetwork
and not full MaleCNS. If this does not reproduce their reported rhythm,
do not compare our simulator yet.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = ROOT / "third_party" / "Pugliese_2026"
OUT = ROOT / "outputs" / "pugliese_dng100_control.json"
REPO = "https://github.com/smpuglie/Pugliese_2026.git"


def _ensure_repo() -> Path:
    if (THIRD_PARTY / "src" / "simulation" / "vnc_sim.py").exists():
        return THIRD_PARTY
    THIRD_PARTY.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["git", "clone", "--depth", "1", REPO, str(THIRD_PARTY)])
    return THIRD_PARTY


def _ensure_jax() -> None:
    try:
        import diffrax  # noqa: F401
        import jax  # noqa: F401
    except ImportError:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "jax>=0.6.2,<0.10",
                "diffrax==0.7.0",
                "hydra-core>=1.3,<2.0",
                "sparse>=0.17.0,<0.19",
                "h5py",
                "natsort",
            ]
        )


def run(*, n_replicates: int = 1, T: float = 2.0, stim_current: float = 250.0) -> dict:
    repo = _ensure_repo()
    _ensure_jax()
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.pop("XLA_FLAGS", None)
    sys.path.insert(0, str(repo))
    import jax
    import jax.numpy as jnp
    import numpy as np
    import pandas as pd
    from jax import random

    from src.simulation.vnc_sim import reweight_connectivity, run_single_simulation
    from src.utils.sim_utils import compute_oscillation_score, load_W, load_wTable, make_input, sample_trunc_normal, set_sizes

    w_path = repo / "data" / "manc t1 connectome data" / "W_20250813_DNtoMN_unsorted.csv"
    df_path = repo / "data" / "manc t1 connectome data" / "wTable_20250813_DNtoMN_unsorted_withModules.csv"
    if not w_path.exists() or not df_path.exists():
        return {
            "ok": False,
            "error": "authors' MANC T1 CSV files are missing after clone",
            "w_path": str(w_path),
            "df_path": str(df_path),
        }

    W = jnp.array(load_W(str(w_path)))
    table = load_wTable(str(df_path))
    n_neurons = int(W.shape[0])
    type_col = None
    for candidate in ("type", "cell_type", "class", "NT", "predictedNt"):
        if candidate in table.columns:
            type_col = candidate
            break
    type_series = table[type_col].astype(str) if type_col else pd.Series([""] * n_neurons)
    dng_rows = []
    for needle in ("DNg100", "BDN2"):
        hits = [i for i, name in enumerate(type_series.tolist()) if needle.lower() in str(name).lower()]
        if hits:
            dng_rows = hits
            break
    stim_idx = 31 if 31 < n_neurons else int(dng_rows[0] if dng_rows else 0)
    class_col = "class" if "class" in table.columns else None
    mn_mask = np.zeros(n_neurons, dtype=bool)
    if class_col:
        mn_mask = table[class_col].astype(str).str.contains(r"motor neuron", case=False, na=False).to_numpy()
    if not mn_mask.any() and "subclass" in table.columns:
        mn_mask = table["subclass"].astype(str).str.contains("motor", case=False, na=False).to_numpy()

    key = random.PRNGKey(1)
    keys = random.split(key, 5)
    tau = sample_trunc_normal(keys[0], 0.02, 0.002, (n_neurons,))
    a = sample_trunc_normal(keys[1], 1.0, 0.1, (n_neurons,))
    threshold = sample_trunc_normal(keys[2], 7.5, 0.6, (n_neurons,))
    fr_cap = sample_trunc_normal(keys[3], 200.0, 10.0, (n_neurons,))
    size_col = "size" if "size" in table.columns else "surf_area_um2"
    a, threshold = set_sizes(table[size_col].values, a[None, :], threshold[None, :])
    a = a[0]
    threshold = threshold[0]
    W_w = reweight_connectivity(W, 0.03, 0.03)
    inputs = make_input(n_neurons, jnp.array([stim_idx]), stim_current)
    dt = 0.001
    t_axis = jnp.arange(0.0, T + dt, dt)
    activity = run_single_simulation(
        W_w,
        tau,
        a,
        threshold,
        fr_cap,
        inputs,
        jnp.zeros(n_neurons),
        t_axis,
        T,
        dt,
        0.02,
        T - 0.001,
        2e-6,
        5e-9,
        keys[4],
    )
    activity = np.asarray(activity)
    mean_rate = activity.mean(axis=1)
    active_mask = mean_rate > 1.0
    if not active_mask.any():
        active_mask = mean_rate > 0.1
    osc_score, mean_hz = compute_oscillation_score(jnp.asarray(activity), jnp.asarray(active_mask), 0.05)
    e1_mask = type_series.eq("IN17A001").to_numpy()
    e2_mask = type_series.eq("INXXX466").to_numpy()
    i1_mask = type_series.eq("IN16B036").to_numpy()

    def _score(mask):
        if not mask.any():
            return None, None, 0
        sc, hz = compute_oscillation_score(
            jnp.asarray(activity[mask]),
            jnp.asarray(active_mask[mask]),
            0.05,
        )
        return float(sc), float(hz), int(mask.sum())

    e1_score, e1_hz, e1_n = _score(e1_mask)
    e2_score, e2_hz, e2_n = _score(e2_mask)
    i1_score, i1_hz, i1_n = _score(i1_mask)
    mn_score = mn_hz = None
    mn_n = int(mn_mask.sum())
    if mn_mask.any():
        mn_score, mn_hz, mn_n = _score(mn_mask)
    payload = {
        "ok": True,
        "source": "Pugliese et al. 2025 bioRxiv; github.com/smpuglie/Pugliese_2026",
        "experiment": "DNg100_Stim parameters on authors' JAX rate-ODE code",
        "n_neurons": n_neurons,
        "stim_index": int(stim_idx),
        "stim_type": str(type_series.iloc[int(stim_idx)]),
        "stim_body_id": int(table.iloc[int(stim_idx)]["bodyId"]) if "bodyId" in table.columns else None,
        "stim_current": stim_current,
        "dng100_rows_in_wtable": dng_rows[:8],
        "note": "Authors' yaml stimulates index 31 only; the second DNg100 is index 132 and is commented out.",
        "type_col": type_col,
        "T_s": T,
        "dt": dt,
        "exc_multiplier": 0.03,
        "inh_multiplier": 0.03,
        "rtol": 2e-6,
        "atol": 5e-9,
        "n_active": int(active_mask.sum()),
        "mean_rate_all": float(mean_rate.mean()),
        "mean_rate_active": float(mean_rate[active_mask].mean()) if active_mask.any() else 0.0,
        "oscillation_score_active": float(osc_score),
        "mean_frequency_active": float(mean_hz),
        "E1_n": e1_n,
        "E1_oscillation_score": e1_score,
        "E1_mean_frequency": e1_hz,
        "E2_n": e2_n,
        "E2_oscillation_score": e2_score,
        "E2_mean_frequency": e2_hz,
        "I1_n": i1_n,
        "I1_oscillation_score": i1_score,
        "I1_mean_frequency": i1_hz,
        "mn_n": mn_n,
        "mn_oscillation_score": mn_score,
        "mn_mean_frequency": mn_hz,
        "rhythm_reproduced": bool(
            float(osc_score) >= 0.5
            or (mn_score is not None and mn_score >= 0.5)
            or (e1_score is not None and e1_score >= 0.5)
        ),
        "jax_devices": [str(d) for d in jax.devices()],
        "repo": str(repo),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))
    payload["output"] = str(OUT)
    return payload


def main() -> int:
    result = run()
    print(json.dumps({k: result[k] for k in result if k != "output"}, indent=2, default=str)[:4000])
    print("wrote", result.get("output", OUT))
    if not result.get("ok"):
        return 1
    if result.get("rhythm_reproduced"):
        print("AUTHORS' SIMULATOR: rhythm score >= 0.5")
        return 0
    print("AUTHORS' SIMULATOR: did not reach oscillation_threshold 0.5 on this machine")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
