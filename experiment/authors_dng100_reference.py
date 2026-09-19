"""Run Pugliese et al. DNg100_Stim via their Hydra entrypoint, unchanged.

This is not MixedDynamicsNetwork and not a reimplementation of their ODE.
Requires the authors' `vnc-sim` environment (`python` on PATH).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from flybrain.neurons import PUGLIESE_CPG_MODEL
REPO = ROOT / "third_party" / "Pugliese_2026"
VNC_PYTHON = Path(os.environ.get("VNC_SIM_PYTHON", ROOT / ".mamba" / "envs" / "vnc-sim" / "bin" / "python"))
REFERENCE = ROOT / "reference" / "authors_dng100"
SHIPPED_FIGURES_DIR = REPO / "figures"
OUR_HYDRA_ROOT = ROOT / "outputs" / "pugliese_hydra"


def is_shipped_author_figure(path: Path) -> bool:
    """Figures checked in with the cloned Pugliese repo are not our run."""
    try:
        resolved = Path(path).resolve()
        shipped = SHIPPED_FIGURES_DIR.resolve()
        return resolved == shipped or shipped in resolved.parents
    except OSError:
        return False


def is_our_hydra_run(path: Path) -> bool:
    """True only for a directory we created under outputs/pugliese_hydra."""
    try:
        resolved = Path(path).resolve()
        root = OUR_HYDRA_ROOT.resolve()
    except OSError:
        return False
    if is_shipped_author_figure(resolved):
        return False
    if resolved != root and root not in resolved.parents:
        return False
    ckpt = resolved if resolved.name == "ckpt" else resolved / "ckpt"
    run_dir = ckpt.parent if ckpt.name == "ckpt" else resolved
    has_trace = bool(list(ckpt.glob("*_Rs.npz"))) if ckpt.exists() else False
    has_config = (run_dir / "logs" / "run_config.yaml").exists() or (run_dir / ".hydra" / "config.yaml").exists()
    return has_trace and has_config


def _python() -> Path:
    if VNC_PYTHON.exists():
        return VNC_PYTHON
    found = shutil.which("python")
    if found:
        return Path(found)
    raise SystemExit(
        "Authors' vnc-sim environment is missing. Create it from their repo:\n"
        "  conda env create -f third_party/Pugliese_2026/environment.yaml\n"
        "  conda activate vnc-sim && pip install -e third_party/Pugliese_2026"
    )


def run_hydra(
    *,
    n_replicates: int = 128,
    batch_size: int = 16,
    run_id: str = "authors_dng100",
) -> subprocess.CompletedProcess:
    python = _python()
    cmd = [
        str(python),
        "src/run_hydra.py",
        "experiment=DNg100_Stim",
        f"experiment.n_replicates={n_replicates}",
        f"experiment.batch_size={batch_size}",
        "paths=mac",
        f"run_id={run_id}",
    ]
    print("running:", " ".join(cmd), flush=True)
    print("python:", python, flush=True)
    subprocess.check_call([str(python), "--version"])
    return subprocess.run(cmd, cwd=str(REPO), check=True)


def _load_rates(ckpt_dir: Path):
    import sparse

    rs_path = ckpt_dir / "DNg100_Stim_Rs.npz"
    if not rs_path.exists():
        matches = list(ckpt_dir.glob("*_Rs.npz"))
        if not matches:
            raise FileNotFoundError(f"no *_Rs.npz in {ckpt_dir}")
        rs_path = matches[0]
    rates = np.asarray(sparse.load_npz(rs_path).todense())
    return rates, rs_path


def extract_reference(ckpt_dir: Path, config_path: Path | None = None) -> dict:
    ckpt_dir = Path(ckpt_dir)
    if is_shipped_author_figure(ckpt_dir):
        raise ValueError(
            "third_party/Pugliese_2026/figures are author-shipped reference "
            "outputs, not our reproduction. Invoke src/run_hydra.py "
            "experiment=DNg100_Stim with paths=mac and read traces from "
            f"{OUR_HYDRA_ROOT}."
        )
    if not is_our_hydra_run(ckpt_dir):
        raise ValueError(
            f"{ckpt_dir} is not a Hydra run under {OUR_HYDRA_ROOT} with "
            "logs/run_config.yaml (or .hydra/config.yaml) and a *_Rs.npz trace."
        )
    sys.path.insert(0, str(REPO))
    os.chdir(REPO)
    import pandas as pd
    import jax.numpy as jnp
    from src.utils.sim_utils import compute_oscillation_score, load_wTable

    rates, rs_path = _load_rates(ckpt_dir)
    # Authors save (n_stim, n_replicates, n_neurons, n_times)
    if rates.ndim == 4:
        mean_rates = rates.mean(axis=(0, 1))
    elif rates.ndim == 3:
        mean_rates = rates.mean(axis=0)
        if mean_rates.shape[0] != 4604 and mean_rates.shape[-1] == 4604:
            mean_rates = np.moveaxis(mean_rates, -1, 0)
    elif rates.ndim == 2:
        mean_rates = rates if rates.shape[0] <= rates.shape[1] else rates.T
    else:
        raise ValueError(f"unexpected rates shape {rates.shape}")
    if mean_rates.shape[0] != 4604 and 4604 in mean_rates.shape:
        mean_rates = np.moveaxis(mean_rates, list(mean_rates.shape).index(4604), 0)

    table = load_wTable(str(REPO / "data" / "manc t1 connectome data" / "wTable_20250813_DNtoMN_unsorted_withModules.csv"))
    type_series = table["type"].astype(str) if "type" in table.columns else pd.Series([""] * mean_rates.shape[0])
    class_col = "class" if "class" in table.columns else None
    mn_mask = np.zeros(mean_rates.shape[0], dtype=bool)
    if class_col:
        mn_mask = table[class_col].astype(str).str.contains(r"motor neuron", case=False, na=False).to_numpy()[: mean_rates.shape[0]]
    motor_rates = mean_rates[mn_mask] if mn_mask.any() else mean_rates[:0]
    mean_rate = mean_rates.mean(axis=1)
    active_mask = mean_rate > 1.0
    if not np.any(active_mask):
        active_mask = mean_rate > 0.1
    osc, hz = compute_oscillation_score(jnp.asarray(mean_rates), jnp.asarray(active_mask), 0.05)
    hz_cycles = float(hz)
    hz_real = hz_cycles / 0.001 if 0 < hz_cycles < 1 else hz_cycles

    def _pop(name: str, activity: np.ndarray, mask_active: np.ndarray):
        mask = type_series.eq(name).to_numpy()[: activity.shape[0]]
        if not mask.any():
            return {"n": 0, "oscillation_score": None, "frequency_cycles_per_sample": None, "frequency_hz": None}
        sc, f = compute_oscillation_score(jnp.asarray(activity[mask]), jnp.asarray(mask_active[mask]), 0.05)
        f = float(f)
        return {
            "n": int(mask.sum()),
            "oscillation_score": float(sc),
            "frequency_cycles_per_sample": f,
            "frequency_hz": f / 0.001 if 0 < f < 1 else f,
        }

    per_rep = []
    if rates.ndim == 4:
        n_rep = int(rates.shape[1])
        for r in range(n_rep):
            act = rates[0, r]
            mu = act.mean(axis=1)
            am = mu > 1.0
            if not np.any(am):
                am = mu > 0.1
            sc_e1 = _pop("IN17A001", act, am)
            per_rep.append(
                {
                    "n_active": int(am.sum()),
                    "E1": sc_e1,
                    "E2": _pop("INXXX466", act, am),
                    "I1": _pop("IN16B036", act, am),
                }
            )
    e1_osc = sum(1 for row in per_rep if (row["E1"].get("oscillation_score") or 0) >= 0.5)

    REFERENCE.mkdir(parents=True, exist_ok=True)
    np.save(REFERENCE / "rates.npy", mean_rates)
    np.save(REFERENCE / "motor_rates.npy", motor_rates)
    if config_path and config_path.exists():
        shutil.copy(config_path, REFERENCE / "config.yaml")
    metrics = {
        "source": "Pugliese_2026 src/run_hydra.py experiment=DNg100_Stim (authors' code, not a reimplementation)",
        "rates_shape_raw": list(rates.shape),
        "rates_shape_mean": list(mean_rates.shape),
        "n_replicates": int(rates.shape[1]) if rates.ndim == 4 else 1,
        "rs_path": str(rs_path),
        "n_active_mean_trace": int(active_mask.sum()),
        "mean_rate_all": float(mean_rate.mean()),
        "mean_rate_active": float(mean_rate[active_mask].mean()) if np.any(active_mask) else 0.0,
        "mean_trace_oscillation_score": float(osc),
        "mean_trace_frequency_hz": hz_real,
        "per_replicate_mean_oscillation_score": None,
        "per_replicate_mean_frequency_hz": float(
            np.mean([row["E1"]["frequency_hz"] for row in per_rep if row["E1"].get("frequency_hz")])
        )
        if per_rep
        else hz_real,
        "fraction_replicates_oscillation_ge_0.5": None,
        "fraction_replicates_E1_oscillation_ge_0.5": float(e1_osc / len(per_rep)) if per_rep else None,
        "published_walk_hz": [7.0, 15.0],
        "model_id": PUGLIESE_CPG_MODEL,
        "dynamics_model": PUGLIESE_CPG_MODEL,
        "includes_cell_size_normalization": True,
        "E1_mean_trace": _pop("IN17A001", mean_rates, active_mask),
        "E1_per_replicate_mean": float(np.mean([row["E1"]["oscillation_score"] for row in per_rep if row["E1"]["oscillation_score"] is not None])) if per_rep else None,
        "E2_per_replicate_mean": float(np.mean([row["E2"]["oscillation_score"] for row in per_rep if row["E2"]["oscillation_score"] is not None])) if per_rep else None,
        "I1_per_replicate_mean": float(np.mean([row["I1"]["oscillation_score"] for row in per_rep if row["I1"]["oscillation_score"] is not None])) if per_rep else None,
        "E1": _pop("IN17A001", mean_rates, active_mask),
        "E2": _pop("INXXX466", mean_rates, active_mask),
        "I1": _pop("IN16B036", mean_rates, active_mask),
        "I2": _pop("IN19A007", mean_rates, active_mask),
        "I2_note": (
            "I2 is IN19A007 (Pugliese et al. 2025 PMC13142387). "
            "IN19B007 is not I2."
        ),
        "mn_n": int(mn_mask.sum()),
        "rhythm_reproduced": bool(
            (e1_osc / len(per_rep) >= 0.5) if per_rep else float(osc) >= 0.5
        ),
        "shipped_author_figures": str(SHIPPED_FIGURES_DIR),
        "shipped_figures_count_as_reproduction": False,
        "our_run_dir": str(ckpt_dir.parent),
        "our_ckpt_dir": str(ckpt_dir),
        "note": (
            "Authors report frequency as 1/lag_samples. Multiply by 1000 (1/dt) for Hz. "
            "Do not clamp onto 7–15 Hz. Phase-averaging 128 traces can wash out rhythm; "
            "per-replicate scores are the implementation test. "
            "Cloned figures/DNg100_Stim_* directories are not this run."
        ),
    }
    if hz_real < 7:
        metrics["frequency_below_published_band"] = True
    (REFERENCE / "rhythm_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def latest_ckpt(run_id: str = "authors_dng100") -> Path:
    """Our Hydra ckpt only. Never third_party/Pugliese_2026/figures."""
    base = OUR_HYDRA_ROOT / "DNg100_Stim"
    matches = [p for p in base.glob(f"**/run_id={run_id}/ckpt") if is_our_hydra_run(p)]
    if matches:
        return matches[0]
    matches = [p for p in base.glob("**/ckpt") if is_our_hydra_run(p)]
    if not matches:
        raise FileNotFoundError(
            f"no Hydra ckpt under {base}. Shipped figures in {SHIPPED_FIGURES_DIR} "
            "do not count. Run: python src/run_hydra.py experiment=DNg100_Stim paths=mac"
        )
    return max(matches, key=lambda p: p.stat().st_mtime)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-replicates", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--run-id", default="authors_dng100")
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.extract_only:
        run_hydra(n_replicates=args.n_replicates, batch_size=args.batch_size, run_id=args.run_id)
    ckpt = latest_ckpt(args.run_id)
    config = ckpt.parent / "logs" / "run_config.yaml"
    metrics = extract_reference(ckpt, config if config.exists() else None)
    print(json.dumps({k: metrics[k] for k in metrics if k not in {"rs_path"}}, indent=2)[:4000])
    print("wrote", REFERENCE)
    return 0 if metrics.get("rhythm_reproduced") else 0


if __name__ == "__main__":
    raise SystemExit(main())
