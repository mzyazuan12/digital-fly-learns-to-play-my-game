"""Step 6: the ported rate ODE must reproduce the authors' MANC DNg100 run.

Feeds their MANC T1 W CSV, wTable sizes, and Hydra neuron_params.h5 into
this repository's half-tanh integrator. If E1/E2 rhythm fails here, the
MaleCNS transfer is a porting bug, not a connectome result.

This is not a Shiu-LIF matrix and not a MaleCNS run.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from experiment.pugliese_rate_malecns import (
    OSCILLATION_THRESHOLD,
    ROLE_TYPES,
    _e1_e2_rhythm,
    _jsonable,
    _pop_metrics,
    _try_authors_oscillation_score,
)
from flybrain.neurons import PUGLIESE_CPG_MODEL, PUGLIESE_RATE_MALECNS_V1
from flybrain.pugliese_rate import (
    PuglieseRateParams,
    integrate_rate,
    make_input,
    rates_are_valid,
    reweight_connectivity,
    sample_cell_parameters_block,
    set_sizes,
)
from organism.roi_innervation import (
    CORE_CPG_TYPES,
    PUGLIESE_MANC_STIM_BODY,
    PUGLIESE_MANC_STIM_INDEX,
    PUGLIESE_MANC_T1_TABLE,
)

ROOT = Path(__file__).resolve().parents[1]
MANC_W_PATH = (
    ROOT
    / "third_party"
    / "Pugliese_2026"
    / "data"
    / "manc t1 connectome data"
    / "W_20250813_DNtoMN_unsorted.csv"
)
HYDRA_CKPT = (
    ROOT
    / "outputs"
    / "pugliese_hydra"
    / "DNg100_Stim"
    / "default"
    / "run_id=validation_20260919"
    / "ckpt"
)
AUTHORS_METRICS = ROOT / "outputs" / "pugliese_reference_20260919" / "analysis_final" / "rhythm_metrics.json"


def load_authors_manc_w(path: Path = MANC_W_PATH) -> np.ndarray:
    """Anatomical signed W[pre, post] exactly as load_W reads the CSV."""
    return pd.read_csv(path).drop(columns="bodyId_pre").to_numpy(dtype=np.float64)


def load_authors_traces(ckpt: Path = HYDRA_CKPT) -> np.ndarray:
    rs_path = Path(ckpt) / "DNg100_Stim_Rs.npz"
    try:
        import sparse

        rates = np.asarray(sparse.load_npz(rs_path).todense(), dtype=np.float64)
    except Exception:
        loaded = np.load(rs_path, allow_pickle=True)
        rates = np.asarray(loaded[loaded.files[0]], dtype=np.float64)
    if rates.ndim != 4:
        raise ValueError(f"Expected authors rates (n_stim, n_rep, n, t), got {rates.shape}")
    return rates


def load_authors_neuron_params(ckpt: Path = HYDRA_CKPT) -> dict:
    import h5py

    path = Path(ckpt) / "neuron_params.h5"
    with h5py.File(path, "r") as handle:
        return {
            "W": np.asarray(handle["W"], dtype=np.float64),
            "a": np.asarray(handle["a"], dtype=np.float64),
            "threshold": np.asarray(handle["threshold"], dtype=np.float64),
            "tau": np.asarray(handle["tau"], dtype=np.float64),
            "fr_cap": np.asarray(handle["fr_cap"], dtype=np.float64),
            "input_currents": np.asarray(handle["input_currents"], dtype=np.float64),
        }


def manc_role_indices(table: pd.DataFrame) -> dict[str, np.ndarray]:
    types = table["type"].astype(str)
    idx = {role: np.flatnonzero(types.eq(typ).to_numpy()) for role, typ in ROLE_TYPES.items()}
    idx["MN"] = np.flatnonzero(table["class"].astype(str).eq("motor neuron").to_numpy())
    idx["DNg100"] = np.flatnonzero(table["bodyId"].to_numpy() == int(PUGLIESE_MANC_STIM_BODY))
    return idx


def _trace_agreement(ours: np.ndarray, authors: np.ndarray) -> dict:
    ours = np.asarray(ours, dtype=np.float64)
    authors = np.asarray(authors, dtype=np.float64)
    if ours.shape != authors.shape:
        raise ValueError(f"rate shape mismatch {ours.shape} vs {authors.shape}")
    err = ours - authors
    denom = np.linalg.norm(authors)
    return {
        "rmse": float(np.sqrt(np.mean(err * err))),
        "max_abs": float(np.max(np.abs(err))),
        "mae": float(np.mean(np.abs(err))),
        "relative_l2": None if denom == 0 else float(np.linalg.norm(err) / denom),
        "corrcoef": float(np.corrcoef(ours.ravel(), authors.ravel())[0, 1]) if ours.size else None,
    }


def run(out: Path, *, ckpt: Path = HYDRA_CKPT, n_replicates: int | None = None) -> dict:
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {out}")
    params = PuglieseRateParams()
    out.mkdir(parents=True, exist_ok=False)
    report = {
        "step": 6,
        "model_id": PUGLIESE_RATE_MALECNS_V1,
        "compared_to_model_id": PUGLIESE_CPG_MODEL,
        "source_dataset": "MANC_T1",
        "stimulated_manc_body_id": int(PUGLIESE_MANC_STIM_BODY),
        "source_matrix_index": int(PUGLIESE_MANC_STIM_INDEX),
        "is_pugliese_reproduction": True,
        "full_malecns_allowed": False,
        "status": "running",
        "port_ok": False,
        "rhythm_reproduced": False,
        "integrator": "scipy.integrate.solve_ivp RK45",
        "parameter_source": "authors neuron_params.h5 (already size-normalized)",
        "weight_source": "authors W CSV via pandas drop bodyId_pre (not a Shiu-LIF matrix)",
        "stimulus_source": "authors DNg100_Stim.yaml stimI=250 / Hydra input_currents",
        "note": (
            "If this port does not recover E1/E2 rhythm on the authors' MANC "
            "matrix, stop. Do not interpret a later MaleCNS failure as biology."
        ),
    }

    def save():
        (out / "report.json").write_text(json.dumps(_jsonable(report), indent=2, allow_nan=False) + "\n")

    save()
    try:
        print("loading authors MANC W / table / Hydra params", flush=True)
        W = load_authors_manc_w()
        table = pd.read_csv(PUGLIESE_MANC_T1_TABLE, index_col=0)
        hydra = load_authors_neuron_params(ckpt)
        authors_rates = load_authors_traces(ckpt)
        n_rep_available = int(authors_rates.shape[1])
        n_rep = n_rep_available if n_replicates is None else min(int(n_replicates), n_rep_available)
        if W.shape != hydra["W"].shape:
            raise ValueError(f"W CSV shape {W.shape} != h5 W {hydra['W'].shape}")
        w_abs = np.max(np.abs(W - hydra["W"]))
        report["W_csv_vs_h5_max_abs"] = float(w_abs)
        sizes = table["size"].to_numpy(dtype=np.float64)
        sampled = sample_cell_parameters_block(W.shape[0], n_rep, seed=1, params=params)
        a_hat, th_hat = set_sizes(sizes, sampled["gain"], sampled["threshold"])
        report["sampling"] = {
            "sampler": sampled["sampler"],
            "seed": 1,
            "tau_max_abs_vs_h5": float(np.max(np.abs(sampled["tau"][:n_rep] - hydra["tau"][:n_rep]))),
            "a_max_abs_vs_h5_after_set_sizes": float(np.max(np.abs(a_hat[:n_rep] - hydra["a"][:n_rep]))),
            "threshold_max_abs_vs_h5_after_set_sizes": float(
                np.max(np.abs(th_hat[:n_rep] - hydra["threshold"][:n_rep]))
            ),
            "fr_cap_max_abs_vs_h5": float(np.max(np.abs(sampled["fr_cap"][:n_rep] - hydra["fr_cap"][:n_rep]))),
        }
        weighted = reweight_connectivity(W, params.exc_multiplier, params.inh_multiplier)
        weighted_h5 = reweight_connectivity(hydra["W"], params.exc_multiplier, params.inh_multiplier)
        report["reweight_csv_vs_h5_max_abs"] = float(np.max(np.abs(weighted - weighted_h5)))
        inputs = hydra["input_currents"][0, 0]
        if float(inputs[PUGLIESE_MANC_STIM_INDEX]) != 250.0:
            raise RuntimeError(
                f"Hydra input at index {PUGLIESE_MANC_STIM_INDEX} is "
                f"{inputs[PUGLIESE_MANC_STIM_INDEX]}, expected 250"
            )
        rebuilt = make_input(W.shape[0], [PUGLIESE_MANC_STIM_INDEX], params.stim_amplitude)
        report["input_rebuild_max_abs"] = float(np.max(np.abs(rebuilt - inputs)))
        idx = manc_role_indices(table)
        replicate_rows = []
        agreements = []
        for rep in range(n_rep):
            print(f"MANC portcheck replicate {rep+1}/{n_rep} n={W.shape[0]}", flush=True)
            rates = integrate_rate(
                weighted,
                tau=hydra["tau"][rep],
                gain=hydra["a"][rep],
                threshold=hydra["threshold"][rep],
                fr_cap=hydra["fr_cap"][rep],
                inputs=inputs,
                params=params,
            )
            authors = authors_rates[0, rep]
            valid = rates_are_valid(rates)
            pops = {role: _pop_metrics(rates, mask, params.dt) for role, mask in idx.items() if mask.size}
            recruited = {
                role: bool((pops.get(role) or {}).get("recruited"))
                for role in ("DNg100", "E1", "E2", "I1", "I2", "MN")
            }
            agree = _trace_agreement(rates, authors)
            authors_pops = {
                role: _pop_metrics(authors, mask, params.dt) for role, mask in idx.items() if mask.size
            }
            row = {
                "replicate": rep,
                "rates_valid": valid,
                "required_recruited": recruited,
                "all_required_recruited": all(recruited.values()),
                "e1_e2_rhythm": bool(valid and all(recruited.values()) and _e1_e2_rhythm(pops)),
                "authors_e1_e2_rhythm": bool(_e1_e2_rhythm(authors_pops)),
                "populations": pops,
                "authors_populations": authors_pops,
                "trace_agreement": agree,
                "rate_min_hz": float(rates.min()),
                "rate_max_hz": float(rates.max()),
            }
            replicate_rows.append(row)
            agreements.append(agree)
            np.savez_compressed(
                out / f"replicate_{rep}.npz",
                model_id=PUGLIESE_RATE_MALECNS_V1,
                source_dataset="MANC_T1",
                rates_hz=rates,
                authors_rates_hz=authors,
                dt_s=params.dt,
                stim_index=np.int64(PUGLIESE_MANC_STIM_INDEX),
            )
        n_ok = sum(1 for row in replicate_rows if row["e1_e2_rhythm"])
        mean_corr = float(np.mean([a["corrcoef"] for a in agreements if a["corrcoef"] is not None]))
        report.update(
            {
                "n_neurons": int(W.shape[0]),
                "n_replicates": n_rep,
                "replicates": replicate_rows,
                "fraction_replicates_E1_E2_rhythm": float(n_ok / len(replicate_rows)),
                "mean_trace_corrcoef": mean_corr,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "authors_metrics_path": str(AUTHORS_METRICS.resolve()) if AUTHORS_METRICS.exists() else None,
            }
        )
        # Scientific pass: recovered E1/E2 rhythm under authors' score, and
        # trajectories stay correlated with the Hydra traces.
        report["rhythm_reproduced"] = bool(n_ok == len(replicate_rows))
        report["port_ok"] = bool(report["rhythm_reproduced"] and mean_corr >= 0.8)
        report["status"] = "manc_port_ok" if report["port_ok"] else "manc_port_failed"
        report["next_step"] = (
            "Port recovered the MANC DNg100 E1/E2 rhythm. MaleCNS transfer may proceed."
            if report["port_ok"]
            else (
                "Port did not recover the authors' MANC result. Debug the ODE, "
                "transpose/reweight, or integrator before touching MaleCNS."
            )
        )
        save()
        return report
    except Exception as exc:
        report.update(status="error", error=f"{type(exc).__name__}: {exc}", port_ok=False)
        save()
        raise


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs")
        / ("pugliese_rate_manc_portcheck_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")),
    )
    parser.add_argument("--ckpt", type=Path, default=HYDRA_CKPT)
    parser.add_argument("--n-replicates", type=int, default=None)
    args = parser.parse_args(argv)
    result = run(args.out, ckpt=args.ckpt, n_replicates=args.n_replicates)
    print(args.out.resolve())
    print("status", result["status"], "port_ok", result["port_ok"])
    return 0 if result["port_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
