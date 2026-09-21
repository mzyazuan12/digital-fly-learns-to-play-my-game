"""mCNS port check on the authors' exact 4,310-cell export.

Feeds W_20260210_vncRoisOnly.csv and wTable_20260210_vncRoisOnly.csv into
the same half-tanh integrator that recovered the MANC DNg100 rhythm.

I=250 is the MANC DNg100 amplitude. It is not a recovered mCNS stimI, so
this run is stimulus_source = "MANC amplitude control". It is not
stimulus_source = "authors_mcns" and it does not unlock the published
mCNS reproduction, the restricted-408 transfer, or connectivity dissection.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiment.pugliese_rate_malecns import (
    _e1_e2_rhythm,
    _jsonable,
    _pop_metrics,
)
from flybrain.authors_mcns import (
    AUTHORS_COMMIT,
    POPULATION_AUTHORS,
    POPULATION_RECONSTRUCTION,
    STIMULUS_AUTHORS_MCNS,
    STIMULUS_MANC_AMPLITUDE_CONTROL,
    W_PATH,
    WTABLE_PATH,
    export_summary,
    load_authors_W,
    load_authors_wtable,
)
from flybrain.malecns_volume import SIZE_NORM_PUGLIESE_SUBNETWORK
from flybrain.neurons import (
    MALECNS_DNG100_L,
    MALECNS_DNG100_R,
    PUGLIESE_CPG_STIM_AMPLITUDE,
    PUGLIESE_RATE_MALECNS_V1,
)
from flybrain.pugliese_rate import (
    PuglieseRateParams,
    integrate_rate,
    make_input,
    rates_are_valid,
    reweight_connectivity,
    sample_cell_parameters_block,
    set_sizes,
)
from organism.roi_innervation import CORE_CPG_TYPES

ROOT = Path(__file__).resolve().parents[1]


def role_indices(table) -> dict[str, np.ndarray]:
    types = table["type"].astype(str)
    idx = {role: np.flatnonzero(types.eq(typ).to_numpy()) for role, typ in CORE_CPG_TYPES.items()}
    idx["DNg100"] = np.flatnonzero(table["bodyId"].to_numpy() == int(MALECNS_DNG100_R))
    idx["DNg100_not_driven"] = np.flatnonzero(table["bodyId"].to_numpy() == int(MALECNS_DNG100_L))
    idx["MN"] = np.flatnonzero(table["class"].astype(str).eq("motor neuron").to_numpy())
    return idx


def _require_manc_port(path: Path) -> dict:
    port_path = Path(path)
    report_path = port_path / "report.json" if port_path.is_dir() else port_path
    if not report_path.exists():
        raise FileNotFoundError(
            "mCNS port check requires a passing MANC port check. "
            f"Missing {report_path}."
        )
    report = json.loads(report_path.read_text())
    if not report.get("port_ok"):
        raise RuntimeError(
            f"MANC port check has not passed ({report.get('status')}). "
            "Do not interpret an mCNS run until that port has."
        )
    return report


def preflight(summary: dict, *, stim_amplitude: float) -> dict:
    """Label the run before any integration. I=250 stays a MANC control."""
    if float(stim_amplitude) != float(PUGLIESE_CPG_STIM_AMPLITUDE):
        raise ValueError(
            "This runner only performs the I=250 MANC amplitude control. "
            f"Got {stim_amplitude}."
        )
    dng = summary["dng100"][str(MALECNS_DNG100_R)]
    other = summary["dng100"][str(MALECNS_DNG100_L)]
    return {
        "model_id": PUGLIESE_RATE_MALECNS_V1,
        "population": POPULATION_AUTHORS,
        "not_population": POPULATION_RECONSTRUCTION,
        "authors_commit": AUTHORS_COMMIT,
        "n_neurons": int(summary["n"]),
        "size_normalization_mode": SIZE_NORM_PUGLIESE_SUBNETWORK,
        "size_median": float(summary["size"]["nanmedian"]),
        "size_median_method": summary["size"]["method"],
        "missing_or_zero_sizes": int(summary["size"]["missing_or_nonfinite"])
        + int(summary["size"]["zero"]),
        "stimulated_bodyId": int(MALECNS_DNG100_R),
        "stimulated_instance": dng["instance"],
        "stimulated_matrix_index": int(dng["matrix_index"]),
        "stimulated_normalized_size": float(dng["normalized_size"]),
        "stimulated_mean_threshold": float(dng["mean_threshold_after_set_sizes"]),
        "not_stimulated_bodyId": int(MALECNS_DNG100_L),
        "not_stimulated_instance": other["instance"],
        "not_stimulated_mean_threshold": float(other["mean_threshold_after_set_sizes"]),
        "stim_amplitude": float(stim_amplitude),
        "stimulus_source": STIMULUS_MANC_AMPLITUDE_CONTROL,
        "stimulus_source_note": (
            "I=250 is the MANC DNg100_Stim amplitude. The paper says CNS "
            "datasets use a larger DNg100 input because brain arbors increase "
            "normalized DN size, and it reports DNg100-driven mCNS rhythms. "
            "The mCNS stimI itself is not in the cloned configs, so 250 is "
            "not stimulus_source authors_mcns."
        ),
        "is_authors_mcns_reproduction": False,
        "is_mcns_port_check": True,
        "full_malecns_allowed": False,
        "restricted_408_unlocked": False,
        "connectivity_dissection_unlocked": False,
        "mean_threshold_versus_stimulus": {
            "I": float(stim_amplitude),
            "mean_threshold": float(dng["mean_threshold_after_set_sizes"]),
            "I_exceeds_mean_threshold": bool(
                float(stim_amplitude) > float(dng["mean_threshold_after_set_sizes"])
            ),
        },
    }


def run(
    out: Path,
    *,
    manc_portcheck: Path,
    n_replicates: int = 1,
    seed: int = 1,
    stim_amplitude: float = PUGLIESE_CPG_STIM_AMPLITUDE,
    integrate: bool = True,
) -> dict:
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {out}")
    if int(n_replicates) < 1:
        raise ValueError("n_replicates must be positive")
    manc = _require_manc_port(manc_portcheck)
    summary = export_summary()
    header = preflight(summary, stim_amplitude=stim_amplitude)
    if header["stimulus_source"] == STIMULUS_AUTHORS_MCNS:
        raise RuntimeError("I=250 must not be labeled as the authors' mCNS stimulus")
    params = PuglieseRateParams()
    if params.stim_amplitude != float(stim_amplitude):
        params = PuglieseRateParams(stim_amplitude=float(stim_amplitude))
    out.mkdir(parents=True, exist_ok=False)
    report = {
        **header,
        "step": "mcns_port_check",
        "wtable": str(WTABLE_PATH),
        "W": str(W_PATH),
        "wtable_sha256": summary["wtable_sha256"],
        "W_sha256": summary["W_sha256"],
        "membership_rule": summary["membership_rule"],
        "consensusNt_counts": summary["consensusNt_counts"],
        "manc_portcheck": {
            "path": str(Path(manc_portcheck).resolve()),
            "port_ok": True,
            "status": manc.get("status"),
            "mean_trace_corrcoef": manc.get("mean_trace_corrcoef"),
        },
        "parameter_seed": int(seed),
        "n_replicates": int(n_replicates),
        "integrator": "scipy.integrate.solve_ivp RK45",
        "weight_source": "authors W CSV via pandas drop bodyId_pre; already signed",
        "status": "running",
        "rhythm_reproduced": False,
        "dng100_recruited": False,
    }

    def save():
        (out / "report.json").write_text(json.dumps(_jsonable(report), indent=2, allow_nan=False) + "\n")

    save()
    try:
        table = load_authors_wtable()
        matrix, _, _ = load_authors_W()
        sizes = table["size"].to_numpy(dtype=np.float64)
        drawn = sample_cell_parameters_block(matrix.shape[0], int(n_replicates), int(seed), params)
        gain, threshold = set_sizes(sizes, drawn["gain"], drawn["threshold"])
        stim_index = int(header["stimulated_matrix_index"])
        stim_threshold = threshold[:, stim_index]
        report["sampled_stimulated_threshold"] = {
            "replicate_0": float(stim_threshold[0]),
            "min": float(stim_threshold.min()),
            "max": float(stim_threshold.max()),
            "I_exceeds_sampled_threshold_replicate_0": bool(
                float(stim_amplitude) > float(stim_threshold[0])
            ),
        }
        report["parameter_sampler"] = drawn["sampler"]
        report["size_scale_matches_table_median"] = bool(
            abs((sizes[stim_index] / float(summary["size"]["nanmedian"])) - header["stimulated_normalized_size"])
            < 1e-9
        )
        if not integrate:
            report["status"] = "preflight_only"
            report["next_step"] = "Integration was not requested."
            save()
            return report
        weighted = reweight_connectivity(matrix, params.exc_multiplier, params.inh_multiplier)
        inputs = make_input(matrix.shape[0], [stim_index], float(stim_amplitude))
        if int(np.count_nonzero(inputs)) != 1 or float(inputs[stim_index]) != float(stim_amplitude):
            raise RuntimeError("Stimulus must hit only the authors' 10056 column")
        if float(inputs[int(summary["dng100"][str(MALECNS_DNG100_L)]["matrix_index"])]) != 0.0:
            raise RuntimeError("10045 must stay undriven")
        idx = role_indices(table)
        replicate_rows = []
        for rep in range(int(n_replicates)):
            print(
                f"mCNS port check replicate {rep + 1}/{n_replicates} "
                f"n={matrix.shape[0]} I={stim_amplitude} source={STIMULUS_MANC_AMPLITUDE_CONTROL}",
                flush=True,
            )
            rates = integrate_rate(
                weighted,
                tau=drawn["tau"][rep],
                gain=gain[rep],
                threshold=threshold[rep],
                fr_cap=drawn["fr_cap"][rep],
                inputs=inputs,
                params=params,
            )
            valid = rates_are_valid(rates)
            pops = {role: _pop_metrics(rates, mask, params.dt) for role, mask in idx.items() if mask.size}
            dng_trace = rates[stim_index]
            recruited = {
                role: bool((pops.get(role) or {}).get("recruited"))
                for role in ("DNg100", "E1", "E2", "I1", "I2", "MN")
            }
            row = {
                "replicate": rep,
                "rates_valid": valid,
                "populations": pops,
                "required_recruited": recruited,
                "dng100_10056_max_hz": float(dng_trace.max()),
                "dng100_10056_recruited": bool(np.any(dng_trace > 1e-8)),
                "dng100_10045_max_hz": float(rates[idx["DNg100_not_driven"]].max())
                if idx["DNg100_not_driven"].size
                else None,
                "all_required_recruited": all(recruited.values()),
                "e1_e2_rhythm": bool(valid and all(recruited.values()) and _e1_e2_rhythm(pops)),
                "rate_min_hz": float(rates.min()),
                "rate_max_hz": float(rates.max()),
            }
            replicate_rows.append(row)
            np.savez_compressed(
                out / f"replicate_{rep}.npz",
                model_id=PUGLIESE_RATE_MALECNS_V1,
                population=POPULATION_AUTHORS,
                stimulus_source=STIMULUS_MANC_AMPLITUDE_CONTROL,
                body_ids=table["bodyId"].to_numpy(dtype=np.int64),
                rates_hz=rates,
                dt_s=params.dt,
                stim_body=np.int64(MALECNS_DNG100_R),
                stim_index=np.int64(stim_index),
            )
        report["replicates"] = replicate_rows
        report["dng100_recruited"] = bool(all(row["dng100_10056_recruited"] for row in replicate_rows))
        n_ok = sum(1 for row in replicate_rows if row["e1_e2_rhythm"])
        report["fraction_replicates_E1_E2_rhythm"] = float(n_ok / len(replicate_rows))
        report["rhythm_reproduced"] = bool(n_ok == len(replicate_rows) and n_ok > 0)
        report["is_authors_mcns_reproduction"] = False
        report["stimulus_source"] = STIMULUS_MANC_AMPLITUDE_CONTROL
        if report["dng100_recruited"] and report["rhythm_reproduced"]:
            report["status"] = "mcns_port_I250_control_rhythm"
        elif report["dng100_recruited"]:
            report["status"] = "mcns_port_I250_control_dng100_recruited_no_e1_e2_rhythm"
        else:
            report["status"] = "mcns_port_I250_control_dng100_silent"
        report["next_step"] = (
            "I=250 on the authors' 4,310-cell export is a MANC amplitude control. "
            "The published mCNS reproduction stays locked until their mCNS DNg100 "
            "stimI is recovered. Restricted-408 transfer and connectivity "
            "dissection stay locked."
        )
        report["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        save()
        return report
    except Exception as exc:
        report.update(
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            rhythm_reproduced=False,
            is_authors_mcns_reproduction=False,
            stimulus_source=STIMULUS_MANC_AMPLITUDE_CONTROL,
        )
        save()
        raise


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs")
        / (
            "pugliese_rate_mcns_portcheck_I250_manc_amplitude_control_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        ),
    )
    parser.add_argument("--manc-portcheck", type=Path, required=True)
    parser.add_argument("--n-replicates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    result = run(
        args.out,
        manc_portcheck=args.manc_portcheck,
        n_replicates=args.n_replicates,
        seed=args.seed,
    )
    print(args.out.resolve())
    print(
        "status",
        result["status"],
        "stimulus_source",
        result["stimulus_source"],
        "dng100_recruited",
        result["dng100_recruited"],
        "rhythm_reproduced",
        result["rhythm_reproduced"],
    )
    return 0 if result["dng100_recruited"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
