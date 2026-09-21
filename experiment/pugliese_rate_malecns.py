"""Restricted MaleCNS rate runs. Not the authors' 4,310-cell mCNS reproduction.

The full-MaleCNS-median run is a labeled control
(pugliese_rate_malecns_full_dataset_size_normalization_control).
Pugliese set_sizes() uses the median of the simulation W_table. Their mCNS
table is the 4,310-cell front-leg subnetwork. Until that population is
reconstructed and its DNg100 rhythm is reproduced, this 408-cell runner
must not be called a Pugliese MaleCNS transfer and must not open a
connectivity dissection.

Stimulated cell remains MaleCNS 10056: somaSide R, LegNp innervation L,
curated mancBodyid 10093. 10045 is in the graph and is not driven.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiment.restricted_cpg import restricted_cpg_graph
from flybrain.loader import Connectome
from flybrain.neurons import (
    MALECNS_DNG100_R,
    PUGLIESE_CPG_MODEL,
    PUGLIESE_RATE_MALECNS_V1,
    SHIU_LIF_SANITY_MODEL,
)
from flybrain.malecns_volume import (
    FULL_DATASET_MEDIAN_CONTROL,
    SIZE_NORM_FULL_MALECNS,
    SIZE_NORM_MODES,
    SIZE_NORM_PUGLIESE_SUBNETWORK,
    SIZE_NORM_RESTRICTED_408,
    assert_frozen_reference_median,
    assert_volume_size_source,
)
from flybrain.pugliese_rate import (
    OSCILLATION_THRESHOLD,
    PuglieseRateParams,
    class_conditional_shuffle,
    integrate_rate,
    make_input,
    neuron_sizes_from_neuprint_cache,
    population_rate_metrics,
    rate_rhythmicity_score,
    rates_are_valid,
    reweight_connectivity,
    sample_cell_parameters_block,
    set_sizes,
    signed_weight_matrix,
)
from organism.roi_innervation import (
    CORE_CPG_TYPES,
    PUGLIESE_DNG100_STIM,
    PUGLIESE_MANC_STIM_BODY,
    PUGLIESE_MANC_T1_TABLE,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "outputs" / "pugliese_reference_20260919" / "analysis_final" / "rhythm_metrics.json"
DEFAULT_LIF = ROOT / "outputs" / "neural_validation_20260919T112522Z" / "report.json"
MANC_W_PATH = (
    ROOT
    / "third_party"
    / "Pugliese_2026"
    / "data"
    / "manc t1 connectome data"
    / "W_20250813_DNtoMN_unsorted.csv"
)

ROLE_TYPES = {"DNg100": "DNg100", **CORE_CPG_TYPES}
EDGE_FAMILIES = (
    ("DNg100", "E1"),
    ("DNg100", "E2"),
    ("DNg100", "E3"),
    ("E1", "E2"),
    ("E2", "E1"),
    ("E1", "I1"),
    ("E2", "I1"),
    ("E1", "I2"),
    ("E2", "I2"),
    ("I1", "E1"),
    ("I1", "E2"),
    ("I2", "E1"),
    ("I2", "E2"),
    ("E3", "E1"),
    ("E3", "E2"),
    ("I1", "I2"),
    ("I2", "I1"),
    ("E1", "MN"),
    ("E2", "MN"),
    ("E3", "MN"),
    ("I1", "MN"),
    ("I2", "MN"),
)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if np.isfinite(x) else None
    if isinstance(value, (np.bool_, bool)) or type(value) is bool:
        return bool(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if value is None:
        return None
    return value


def role_indices(graph: Connectome) -> dict[str, np.ndarray]:
    idx = {role: np.flatnonzero(graph.cell_type == typ) for role, typ in ROLE_TYPES.items()}
    idx["MN"] = np.flatnonzero(np.char.find(graph.superclass.astype(str), "motor") >= 0)
    return idx


def family_contacts(W: np.ndarray, pre_idx: np.ndarray, post_idx: np.ndarray) -> dict:
    if pre_idx.size == 0 or post_idx.size == 0:
        return {"n_pre": int(pre_idx.size), "n_post": int(post_idx.size), "signed_sum": 0.0, "abs_sum": 0.0, "n_edges": 0}
    block = W[np.ix_(pre_idx, post_idx)]
    return {
        "n_pre": int(pre_idx.size),
        "n_post": int(post_idx.size),
        "signed_sum": float(block.sum()),
        "abs_sum": float(np.abs(block).sum()),
        "n_edges": int(np.count_nonzero(block)),
    }


def malecns_edge_families(graph: Connectome, W: np.ndarray, *, dng100_body: int) -> dict:
    idx = role_indices(graph)
    dng = np.array([graph.index_of(dng100_body)], dtype=np.int64)
    groups = {**idx, "DNg100": dng}
    table = {}
    for pre_role, post_role in EDGE_FAMILIES:
        table[f"{pre_role}->{post_role}"] = family_contacts(W, groups[pre_role], groups[post_role])
    return table


def manc_edge_families(*, stim_body: int = PUGLIESE_MANC_STIM_BODY) -> dict:
    import pandas as pd

    table = pd.read_csv(PUGLIESE_MANC_T1_TABLE, index_col=0)
    W = pd.read_csv(MANC_W_PATH).drop(columns="bodyId_pre").to_numpy(dtype=np.float64)
    types = table["type"].astype(str)
    groups = {
        "DNg100": np.flatnonzero(table["bodyId"].to_numpy() == int(stim_body)),
        **{role: np.flatnonzero(types.eq(typ).to_numpy()) for role, typ in CORE_CPG_TYPES.items()},
        "MN": np.flatnonzero(table["class"].astype(str).eq("motor neuron").to_numpy()),
    }
    out = {
        "source_dataset": "MANC_T1",
        "stim_body_id": int(stim_body),
        "n_neurons": int(W.shape[0]),
        "families": {},
    }
    for pre_role, post_role in EDGE_FAMILIES:
        out["families"][f"{pre_role}->{post_role}"] = family_contacts(W, groups[pre_role], groups[post_role])
    return out


def compare_edge_families(malecns_families: dict, manc: dict) -> list[dict]:
    rows = []
    for key, male in malecns_families.items():
        other = (manc.get("families") or {}).get(key) or {}
        male_abs = float(male.get("abs_sum") or 0.0)
        manc_abs = float(other.get("abs_sum") or 0.0)
        rows.append(
            {
                "edge_family": key,
                "MANC_abs_sum": manc_abs,
                "MaleCNS_abs_sum": male_abs,
                "ratio_MaleCNS_over_MANC": None if manc_abs == 0 else male_abs / manc_abs,
                "MANC_signed_sum": other.get("signed_sum"),
                "MaleCNS_signed_sum": male.get("signed_sum"),
                "MANC_n_edges": other.get("n_edges"),
                "MaleCNS_n_edges": male.get("n_edges"),
            }
        )
    return rows


def _load_existing_reference(path: Path) -> dict:
    metrics = json.loads(Path(path).read_text())
    if metrics.get("model_id") != PUGLIESE_CPG_MODEL:
        raise ValueError(f"Experiment A must be {PUGLIESE_CPG_MODEL}, got {metrics.get('model_id')}")
    if int(metrics.get("source_body_id") or 0) != int(PUGLIESE_DNG100_STIM["source_body_id"]):
        raise ValueError("Experiment A reference is not MANC body 10093")
    return metrics


def _load_existing_lif(path: Path) -> dict:
    report = json.loads(Path(path).read_text())
    if report.get("model_id") != SHIU_LIF_SANITY_MODEL:
        raise ValueError(f"Experiment C must be {SHIU_LIF_SANITY_MODEL}")
    return report


_authors_score_fn = False


def _try_authors_oscillation_score(activity: np.ndarray, active_mask: np.ndarray):
    """Authors' score when their repo is importable. Not required for the transfer ODE."""
    global _authors_score_fn
    if _authors_score_fn is False:
        repo = ROOT / "third_party" / "Pugliese_2026"
        fn = None
        if (repo / "src" / "utils" / "sim_utils.py").exists():
            if str(repo) not in sys.path:
                sys.path.insert(0, str(repo))
            try:
                from src.utils.sim_utils import compute_oscillation_score as fn  # type: ignore
            except Exception:
                fn = None
        _authors_score_fn = fn
    if _authors_score_fn is None:
        return None
    try:
        import jax.numpy as jnp
    except Exception:
        return None
    score, freq = _authors_score_fn(jnp.asarray(activity), jnp.asarray(active_mask), 0.05)
    freq = float(freq)
    return {
        "oscillation_score": float(score) if np.isfinite(float(score)) else None,
        "frequency_cycles_per_sample": freq if np.isfinite(freq) else None,
        "frequency_hz": (freq / PUGLIESE_RATE_PARAMS_DT) if np.isfinite(freq) and freq > 0 else None,
    }


PUGLIESE_RATE_PARAMS_DT = PuglieseRateParams().dt


def _pop_metrics(rates: np.ndarray, mask: np.ndarray, dt_s: float) -> dict:
    sub = rates[mask] if mask.size else rates[:0]
    recruited = np.any(sub > 1e-8, axis=1) if sub.size else np.zeros(0, dtype=bool)
    row = population_rate_metrics(sub if sub.size else np.zeros((0, rates.shape[1])), dt_s, recruited)
    authors = None
    if sub.size and np.any(recruited):
        authors = _try_authors_oscillation_score(sub, recruited)
    row["authors_oscillation"] = authors
    mean_trace = sub[recruited].mean(axis=0) if np.any(recruited) else np.zeros(rates.shape[1])
    row["mean_trace_rhythmicity"] = rate_rhythmicity_score(mean_trace, dt_s)
    return row


def _e1_e2_rhythm(pops: dict) -> bool:
    def _score(role: str) -> float:
        authors = (pops.get(role) or {}).get("authors_oscillation") or {}
        if authors.get("oscillation_score") is not None:
            return float(authors["oscillation_score"])
        return float((pops.get(role) or {}).get("rhythmicity_score") or 0.0)

    return bool(_score("E1") >= OSCILLATION_THRESHOLD and _score("E2") >= OSCILLATION_THRESHOLD)


def simulate_graph(
    graph: Connectome,
    *,
    stim_body: int,
    seed: int,
    n_replicates: int,
    shuffle: bool,
    params: PuglieseRateParams,
    size_norm: str,
    reference_median: float | None = None,
) -> dict:
    W = signed_weight_matrix(graph)
    print("loading neuPrint Neuron.size cache", flush=True)
    size_info = neuron_sizes_from_neuprint_cache(graph.neuron_ids)
    assert_volume_size_source(size_info["source"])
    circuit_median = float(np.nanmedian(size_info["sizes"]))
    full_median = float(size_info["median_volume_reference"])
    if size_norm == SIZE_NORM_FULL_MALECNS:
        median = full_median
    elif size_norm == SIZE_NORM_RESTRICTED_408:
        median = circuit_median
    elif size_norm == SIZE_NORM_PUGLIESE_SUBNETWORK:
        if reference_median is None:
            raise RuntimeError("Pugliese simulation-population median is not available")
        assert_frozen_reference_median(float(reference_median), float(reference_median))
        median = float(reference_median)
    else:
        raise ValueError(f"Unknown size normalization mode {size_norm}")
    print(
        f"size_norm={size_norm} missing={size_info['n_missing']} "
        f"denominator={median} full_median={full_median} circuit_median={circuit_median}",
        flush=True,
    )
    rng = np.random.default_rng(seed)
    W_sim = W
    if shuffle:
        is_dn = np.array(["descending" in str(s).lower() for s in graph.superclass], dtype=bool)
        is_mn = np.array(["motor" in str(s).lower() for s in graph.superclass], dtype=bool)
        is_exc = graph.neurotransmitter.astype(str) == "acetylcholine"
        W_sim = class_conditional_shuffle(W, is_dn=is_dn, is_mn=is_mn, is_exc=is_exc, rng=rng)
    weighted = reweight_connectivity(W_sim, params.exc_multiplier, params.inh_multiplier)
    stim_idx = graph.index_of(stim_body)
    inputs = make_input(graph.n, [stim_idx], params.stim_amplitude)
    idx = role_indices(graph)
    drawn = sample_cell_parameters_block(graph.n, n_replicates, seed, params)
    gain, threshold = set_sizes(
        size_info["sizes"],
        drawn["gain"],
        drawn["threshold"],
        median_size=median,
    )
    replicates = []
    for rep in range(n_replicates):
        print(f"replicate {rep+1}/{n_replicates} integrating {graph.n} cells", flush=True)
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
        required = ("DNg100", "E1", "E2", "I1", "I2", "MN")
        recruited = {role: bool((pops.get(role) or {}).get("recruited")) for role in required}
        row = {
            "replicate": rep,
            "rates_valid": valid,
            "populations": pops,
            "required_recruited": recruited,
            "all_required_recruited": all(recruited.values()),
            "e1_e2_rhythm": bool(valid and all(recruited.values()) and _e1_e2_rhythm(pops)),
            "rate_min_hz": float(rates.min()),
            "rate_max_hz": float(rates.max()),
        }
        replicates.append({"row": row, "rates": rates})
    return {
        "W": W,
        "W_sim": W_sim,
        "size_info": size_info,
        "size_norm": size_norm,
        "median_used": median,
        "circuit_median": circuit_median,
        "full_dataset_median": full_median,
        "parameter_sampler": drawn["sampler"],
        "replicates": replicates,
        "stim_index": int(stim_idx),
        "indices": {k: v.astype(int).tolist() for k, v in idx.items()},
    }


def run(
    out: Path,
    *,
    stim_body: int = MALECNS_DNG100_R,
    seed: int = 1,
    n_replicates: int = 8,
    shuffle: bool = False,
    reference_metrics: Path = DEFAULT_REFERENCE,
    lif_report: Path = DEFAULT_LIF,
    manc_portcheck: Path | None = None,
    size_norm: str = SIZE_NORM_FULL_MALECNS,
    reference_median: float | None = None,
) -> dict:
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {out}")
    if stim_body != MALECNS_DNG100_R:
        raise ValueError(
            f"Primary comparison stimulates MaleCNS {MALECNS_DNG100_R} "
            f"(mancBodyid {PUGLIESE_MANC_STIM_BODY}, left LegNp). Got {stim_body}."
        )
    if size_norm not in SIZE_NORM_MODES:
        raise ValueError(f"Unknown size normalization mode {size_norm}")
    if size_norm == SIZE_NORM_PUGLIESE_SUBNETWORK:
        raise RuntimeError(
            "Restricted-408 rate test is locked. Pugliese's denominator is the "
            "4,310-cell mCNS simulation population, and that reproduction has not "
            "passed. Do not substitute the 408-cell median or the full-MaleCNS median."
        )
    if size_norm == SIZE_NORM_FULL_MALECNS:
        experiment_label = FULL_DATASET_MEDIAN_CONTROL
    else:
        experiment_label = "pugliese_rate_malecns_restricted_408_median_control"
    params = PuglieseRateParams()
    if params.stim_amplitude == 40.0 or params.model_id == SHIU_LIF_SANITY_MODEL:
        raise RuntimeError("Rate transfer must not use Shiu LIF stimulus or model id")
    out.mkdir(parents=True, exist_ok=False)
    letter = "D" if shuffle else "B"
    report = {
        "model_id": PUGLIESE_RATE_MALECNS_V1,
        "dynamics_model": PUGLIESE_RATE_MALECNS_V1,
        "not_shiu_lif": True,
        "includes_cell_size_normalization": True,
        "experiment_letter": letter,
        "source_dataset": "MaleCNS_v1.0",
        "stimulated_malecns_body_ids": [int(stim_body)],
        "pugliese_homolog": dict(PUGLIESE_DNG100_STIM),
        "experiment_label": experiment_label,
        "size_normalization_mode": size_norm,
        "is_authors_mcns_reproduction": False,
        "is_pugliese_reproduction": False,
        "full_malecns_allowed": False,
        "allow_lesions": False,
        "valid_for_rhythm_analysis": False,
        "rhythm_reproduced": False,
        "status": "running",
        "seed": int(seed),
        "n_replicates": int(n_replicates),
        "shuffle": bool(shuffle),
        "integrator": "scipy.integrate.solve_ivp RK45",
        "rtol": params.rtol,
        "atol": params.atol,
        "stim_amplitude": params.stim_amplitude,
        "stimulus_source": "transfer assumption",
        "stimulus_source_note": (
            "No mCNS DNg100 stimI yaml is in the cloned Pugliese repo. MANC "
            "DNg100_Stim uses 250; FANC uses 150. The paper says CNS datasets "
            "needed a larger DNg100 input because brain arbors increase DN "
            "normalized size. Amplitude 250 is therefore a transfer assumption, "
            "not a recovered mCNS convention."
        ),
        "weight_transformation": "anatomical count × NT sign, then W_rate = signed_W.T × 0.03/0.03",
        "rhythm_criterion": "authors compute_oscillation_score >= 0.5 for both E1 and E2; published 7-15 Hz is an annotation",
        "note": (
            f"{experiment_label}. Restricted 408-cell MaleCNS graph with the "
            "ported half-tanh ODE. This is not the authors' 4,310-cell mCNS "
            "simulation and not a connectivity discrepancy. Not a Shiu-LIF retune. "
            "Graph keeps both DNg100 cells; only 10056 is driven."
        ),
    }

    def save():
        (out / "report.json").write_text(json.dumps(_jsonable(report), indent=2, allow_nan=False) + "\n")

    save()
    try:
        experiment_a = _load_existing_reference(reference_metrics)
        experiment_c = _load_existing_lif(lif_report) if Path(lif_report).exists() else None
        report["experiment_A"] = {
            "model_id": PUGLIESE_CPG_MODEL,
            "path": str(Path(reference_metrics).resolve()),
            "rhythm_reproduced": bool(experiment_a.get("rhythm_reproduced")),
            "source_body_id": experiment_a.get("source_body_id"),
            "source_matrix_index": experiment_a.get("source_matrix_index"),
        }
        if experiment_c is not None:
            cond = (experiment_c.get("conditions") or {}).get(str(stim_body), {})
            report["experiment_C"] = {
                "model_id": SHIU_LIF_SANITY_MODEL,
                "path": str(Path(lif_report).resolve()),
                "valid_for_rhythm_analysis": bool(cond.get("valid_for_rhythm_analysis")),
                "label": "restricted MaleCNS / shiu_lif_sanity_v1 connectivity-response phenotype, not a CPG lesion result",
            }
        if not experiment_a.get("rhythm_reproduced"):
            raise RuntimeError("Pugliese MANC reference has not passed; refusing MaleCNS transfer")
        if manc_portcheck is None:
            raise RuntimeError(
                "MaleCNS transfer requires a passing MANC port check (step 6). "
                "Run python -m experiment.pugliese_rate_manc_portcheck and pass --manc-portcheck."
            )
        port_path = Path(manc_portcheck)
        port_report = json.loads((port_path / "report.json" if port_path.is_dir() else port_path).read_text())
        if not port_report.get("port_ok"):
            raise RuntimeError(
                f"MANC port check failed ({port_report.get('status')}). "
                "A MaleCNS result would not be interpretable as connectome biology."
            )
        report["experiment_manc_portcheck"] = {
            "path": str(port_path.resolve()),
            "port_ok": True,
            "status": port_report.get("status"),
            "mean_trace_corrcoef": port_report.get("mean_trace_corrcoef"),
        }
        graph = restricted_cpg_graph()
        dng = sorted(int(x) for x in graph.neuron_ids[graph.cell_type == "DNg100"])
        if dng != [10045, 10056]:
            raise RuntimeError(f"Transfer graph must keep both DNg100 cells, got {dng}")
        if int(stim_body) not in set(graph.neuron_ids.astype(int)):
            raise RuntimeError(f"Stimulated body {stim_body} is not in the restricted graph")
        print(f"restricted graph n={graph.n} edges={graph.n_edges} stim={stim_body} dng={dng}", flush=True)
        if graph.n > 5000:
            raise RuntimeError("Refusing unexpectedly large graph; full MaleCNS is locked")
        sim = simulate_graph(
            graph,
            stim_body=int(stim_body),
            seed=seed,
            n_replicates=n_replicates,
            shuffle=shuffle,
            params=params,
            size_norm=size_norm,
            reference_median=reference_median,
        )
        report["n_neurons"] = graph.n
        report["n_edges"] = graph.n_edges
        report["subset"] = graph.report
        report["size_normalization"] = {
            "source": sim["size_info"]["source"],
            "dataset_version": sim["size_info"]["dataset_version"],
            "cache_path": sim["size_info"]["cache_path"],
            "n_missing": sim["size_info"]["n_missing"],
            "missing_malecns_body_ids": sim["size_info"]["missing_malecns_body_ids"],
            "not_neuprint_volume_property": False,
            "median_volume_reference": float(sim["median_used"]),
            "full_dataset_median": float(sim["full_dataset_median"]),
            "median_of_restricted_circuit": float(sim["circuit_median"]),
            "size_normalization_mode": size_norm,
            "denominator": (
                "full neuPrint male-cns:v1.0 Neuron.size median. Control, not the "
                "authors' 4,310-cell W_table median."
                if size_norm == SIZE_NORM_FULL_MALECNS
                else "median of this 408-cell circuit. Control, not the authors' W_table."
            ),
            "parameter_sampler": sim["parameter_sampler"],
        }
        report["graph_dng100_body_ids"] = dng
        report["stimulated_malecns_body_ids"] = [int(stim_body)]
        manc = manc_edge_families()
        male_families = malecns_edge_families(graph, sim["W"], dng100_body=int(stim_body))
        report["anatomy_comparison"] = {
            "note": (
                "Signed synapse-count families between MANC T1 DNg100 10093 and "
                "MaleCNS DNg100_R 10056 restricted circuits. MANC T1 contains two "
                "copies of each CPG type; this MaleCNS restriction includes all six "
                "T1/T2/T3 copies. Ratios are therefore not per-copy matched. Not a dynamics result."
            ),
            "rows": compare_edge_families(male_families, manc),
            "MANC": manc,
        }
        git = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        report["git_commit"] = git
        replicate_rows = []
        for item in sim["replicates"]:
            replicate_rows.append(item["row"])
            np.savez_compressed(
                out / f"replicate_{item['row']['replicate']}.npz",
                model_id=PUGLIESE_RATE_MALECNS_V1,
                source_dataset="MaleCNS_v1.0",
                malecns_body_ids=graph.neuron_ids,
                cell_type=graph.cell_type.astype(str),
                rates_hz=item["rates"],
                dt_s=params.dt,
                stim_body=np.int64(stim_body),
                shuffle=np.bool_(shuffle),
            )
        report["replicates"] = replicate_rows
        n_ok = sum(1 for row in replicate_rows if row["e1_e2_rhythm"])
        report["fraction_replicates_E1_E2_rhythm"] = float(n_ok / len(replicate_rows)) if replicate_rows else 0.0
        report["valid_for_rhythm_analysis"] = bool(all(row["rates_valid"] for row in replicate_rows))
        report["rhythm_reproduced"] = bool(n_ok / len(replicate_rows) >= 0.5) if replicate_rows else False
        report["is_authors_mcns_reproduction"] = False
        if size_norm == SIZE_NORM_FULL_MALECNS:
            report["status"] = (
                "full_malecns_median_control_rhythm"
                if report["rhythm_reproduced"]
                else "full_malecns_median_control_silent"
            )
            report["next_step"] = (
                "Full-MaleCNS-median control. Not a Pugliese mCNS reproduction and "
                "not a connectivity discrepancy. Next: reconstruct the authors' "
                "4,310-cell mCNS population, take its Neuron.size median, and "
                "reproduce their mCNS DNg100 rhythm before any restricted-408 "
                "rate test. Do not retune Shiu LIF."
            )
        else:
            report["status"] = (
                "restricted_408_median_control_rhythm"
                if report["rhythm_reproduced"]
                else "restricted_408_median_control_silent"
            )
            report["next_step"] = (
                "408-cell median control. Not the authors' denominator. Do not "
                "promote it because it changes DNg100 excitability. The reference "
                "median has to come from the 4,310-cell mCNS population and stay "
                "frozen for every restricted or lesioned run."
            )
        np.savez_compressed(
            out / "weights.npz",
            model_id=PUGLIESE_RATE_MALECNS_V1,
            malecns_body_ids=graph.neuron_ids,
            W_signed=sim["W"],
            sizes=sim["size_info"]["sizes"],
            median_volume_reference=np.float64(sim["median_used"]),
            normalized_size=sim["size_info"]["sizes"] / sim["median_used"],
        )
        _write_preview_figure(out, sim, params.dt, stim_body)
        save()
        return report
    except Exception as exc:
        report.update(status="error", error=f"{type(exc).__name__}: {exc}", rhythm_reproduced=False)
        save()
        raise


def _write_preview_figure(out: Path, sim: dict, dt_s: float, stim_body: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    item = sim["replicates"][0]
    rates = item["rates"]
    idx = {k: np.asarray(v, dtype=int) for k, v in sim["indices"].items()}
    t = np.arange(rates.shape[1]) * dt_s * 1000.0
    fig, axes = plt.subplots(6, 1, figsize=(10, 9), sharex=True)
    series = [
        ("DNg100", idx.get("DNg100")),
        ("E1", idx.get("E1")),
        ("E2", idx.get("E2")),
        ("I1", idx.get("I1")),
        ("I2", idx.get("I2")),
        ("MN", idx.get("MN")),
    ]
    for ax, (title, mask) in zip(axes, series):
        y = rates[mask].mean(axis=0) if mask is not None and mask.size else np.zeros_like(t)
        ax.plot(t, y, color="black", lw=0.8)
        ax.set_ylabel(title, fontsize=8)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("time (ms)")
    fig.suptitle(f"{PUGLIESE_RATE_MALECNS_V1}  DNg100_R {stim_body}  (not {SHIU_LIF_SANITY_MODEL})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "rates.png", dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs") / ("pugliese_rate_malecns_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")),
    )
    parser.add_argument("--stim-body", type=int, default=MALECNS_DNG100_R)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-replicates", type=int, default=8)
    parser.add_argument("--shuffle", action="store_true", help="Experiment D: class-conditional topology null")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--lif-report", type=Path, default=DEFAULT_LIF)
    parser.add_argument(
        "--manc-portcheck",
        type=Path,
        required=True,
        help="Directory or report.json from experiment.pugliese_rate_manc_portcheck (step 6)",
    )
    parser.add_argument(
        "--size-norm",
        choices=list(SIZE_NORM_MODES),
        default=SIZE_NORM_FULL_MALECNS,
        help="Denominator for set_sizes. Default is the full-MaleCNS median control.",
    )
    args = parser.parse_args(argv)
    result = run(
        args.out,
        stim_body=args.stim_body,
        seed=args.seed,
        n_replicates=args.n_replicates,
        shuffle=args.shuffle,
        reference_metrics=args.reference,
        lif_report=args.lif_report,
        manc_portcheck=args.manc_portcheck,
        size_norm=args.size_norm,
    )
    print(args.out.resolve())
    print("status", result["status"], "rhythm_reproduced", result["rhythm_reproduced"])
    return 0 if result["rhythm_reproduced"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
