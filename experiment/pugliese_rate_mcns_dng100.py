"""DNg100 experiment on the authors' exact 4,310-cell mCNS export.

Network: W_20260210_vncRoisOnly.csv / wTable_20260210_vncRoisOnly.csv
(Pugliese_2026 commit 10e7661). Dynamics: the half-tanh rate ODE whose MANC
port matched the authors' traces (mean corrcoef 0.99996). Size
normalization: nanmedian of the simulated table (set_sizes()).

Stimulus: I=400, the published CNS-dataset DNg100 drive (see
flybrain.authors_mcns.AUTHORS_MCNS_STIM_CITATION). Other amplitudes are
allowed only as a labeled "amplitude sweep", never as authors_mcns.

Arms:
  intact             DNg100 drive, full network
  silence:<roles>    same drive, named CPG roles removed from W (rows and
                     columns), as vnc_sim does for experiment.removeNeurons
  shuffle            same drive, class-conditional column shuffle (null)

Read-outs are the paper's: motor neurons are wTable rows with a "motor
module"; a MN is active if r > 0.01 at any time after 250 ms; the motor
rhythmicity score is the authors' compute_oscillation_score on R[:, 250:].
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUTHORS_REPO = ROOT / "third_party" / "Pugliese_2026"
TRANSIENT_SAMPLES = 250  # notebooks: R[..., 250:]
ACTIVE_RATE = 0.01  # notebooks: np.max(R, 1) > 0.01
SCORE_THRESHOLD = 0.5  # configs: oscillation_threshold

_W = None  # worker globals
_PARAMS = None


def _worker_init(weighted: np.ndarray, params_kwargs: dict) -> None:
    global _W, _PARAMS
    from flybrain.pugliese_rate import PuglieseRateParams

    _W = weighted
    _PARAMS = PuglieseRateParams(**params_kwargs)


def _worker_run(job: dict) -> dict:
    from flybrain.pugliese_rate import integrate_rate

    t0 = time.time()
    W = _W
    if job.get("remove") is not None and len(job["remove"]):
        W = W.copy()
        W[job["remove"], :] = 0.0
        W[:, job["remove"]] = 0.0
    if job.get("shuffle_seed") is not None:
        from flybrain.pugliese_rate import class_conditional_shuffle

        # _W is post x pre (reweight_connectivity transposes). The authors'
        # shuffle_W permutes postsynaptic columns of W[pre, post], so shuffle
        # the transpose and transpose back.
        W = class_conditional_shuffle(
            np.ascontiguousarray(W.T),
            is_dn=job["is_dn"],
            is_mn=job["is_mn"],
            is_exc=job["is_exc"],
            rng=np.random.default_rng(job["shuffle_seed"]),
        ).T
    from scipy.sparse import csr_matrix

    # W is 0.6% dense; CSR matvec gives the same sums ~100x faster than a
    # single-threaded dense product. Verified against dense traces in tests.
    rates = integrate_rate(
        csr_matrix(W),
        tau=job["tau"],
        gain=job["gain"],
        threshold=job["threshold"],
        fr_cap=job["fr_cap"],
        inputs=job["inputs"],
        params=_PARAMS,
    )
    return {"rates": rates.astype(np.float32), "seconds": time.time() - t0, "replicate": job["replicate"]}


def _authors_score():
    if str(AUTHORS_REPO) not in sys.path:
        sys.path.insert(0, str(AUTHORS_REPO))
    from src.utils.sim_utils import compute_oscillation_score  # type: ignore

    return compute_oscillation_score


def paper_metrics(rates: np.ndarray, mn_idx: np.ndarray, dt_s: float, score_fn) -> dict:
    """Figure 2 read-outs exactly as the authors' notebooks compute them."""
    import jax.numpy as jnp

    n = rates.shape[0]
    is_mn = np.zeros(n, dtype=bool)
    is_mn[mn_idx] = True
    past = rates[:, TRANSIENT_SAMPLES:]
    active_mn_past = is_mn & (past.max(axis=1) > ACTIVE_RATE)
    active_mn_any = is_mn & (rates.max(axis=1) > ACTIVE_RATE)
    # compute_oscillation_score masks inactive rows to 0 / NaN and averages
    # over active rows only, so scoring just the active rows is identical
    # and avoids vmapping over all 4,310 neurons.
    if active_mn_past.any():
        sub = past[active_mn_past]
        score, freq = score_fn(jnp.asarray(sub), jnp.ones(sub.shape[0], dtype=bool))
        score = float(score)
        freq = float(freq)
    else:
        score, freq = 0.0, float("nan")
    return {
        "n_active_total": int((rates.max(axis=1) > ACTIVE_RATE).sum()),
        "n_mn_active": int(active_mn_any.sum()),
        "n_mn_active_past_transient": int(active_mn_past.sum()),
        "mn_rhythmicity_score": score if np.isfinite(score) else 0.0,
        "mn_frequency_hz": (freq / dt_s) if np.isfinite(freq) and freq > 0 else None,
        "rhythmic": bool(np.isfinite(score) and score >= SCORE_THRESHOLD),
        "max_rate_hz": float(rates.max()),
        "min_rate_hz": float(rates.min()),
    }


def role_metrics(rates: np.ndarray, idx: np.ndarray, dt_s: float, score_fn) -> dict:
    import jax.numpy as jnp

    sub = rates[idx]
    past = sub[:, TRANSIENT_SAMPLES:]
    active = past.max(axis=1) > ACTIVE_RATE
    out = {
        "n": int(idx.size),
        "n_active_past_transient": int(active.sum()),
        "max_rate_hz": float(sub.max()) if sub.size else 0.0,
        "mean_rate_hz_past_transient": float(past.mean()) if sub.size else 0.0,
        "score": 0.0,
        "frequency_hz": None,
    }
    if active.any():
        score, freq = score_fn(jnp.asarray(past), jnp.asarray(active))
        out["score"] = float(score) if np.isfinite(float(score)) else 0.0
        f = float(freq)
        out["frequency_hz"] = (f / dt_s) if np.isfinite(f) and f > 0 else None
    return out


def summarize(rows: list[dict]) -> dict:
    scores = np.array([r["paper"]["mn_rhythmicity_score"] for r in rows])
    nmn = np.array([r["paper"]["n_mn_active_past_transient"] for r in rows])
    freqs = np.array(
        [r["paper"]["mn_frequency_hz"] for r in rows if r["paper"]["mn_frequency_hz"] is not None]
    )
    with_mn = nmn > 0
    rhythmic = np.array([r["paper"]["rhythmic"] for r in rows])
    rhythmic_freqs = np.array(
        [
            r["paper"]["mn_frequency_hz"]
            for r in rows
            if r["paper"]["rhythmic"] and r["paper"]["mn_frequency_hz"] is not None
        ]
    )

    def q(a):
        if a.size == 0:
            return None
        return {
            "median": float(np.median(a)),
            "q25": float(np.percentile(a, 25)),
            "q75": float(np.percentile(a, 75)),
            "min": float(a.min()),
            "max": float(a.max()),
            "mean": float(a.mean()),
        }

    roles = {}
    for role in rows[0]["roles"]:
        rs = np.array([r["roles"][role]["score"] for r in rows])
        act = np.array([r["roles"][role]["n_active_past_transient"] > 0 for r in rows])
        roles[role] = {
            "fraction_replicates_active": float(act.mean()),
            "score": q(rs),
            "fraction_score_ge_0.5": float((rs >= SCORE_THRESHOLD).mean()),
        }
    return {
        "n_replicates": len(rows),
        "fraction_replicates_any_mn_active": float(with_mn.mean()),
        "fraction_replicates_rhythmic": float(rhythmic.mean()),
        "fraction_rhythmic_among_mn_active": float(rhythmic[with_mn].mean()) if with_mn.any() else 0.0,
        "mn_score_all": q(scores),
        "mn_score_mn_active": q(scores[with_mn]),
        "n_mn_active_past_transient": q(nmn.astype(float)),
        "mn_frequency_hz_all": q(freqs),
        "mn_frequency_hz_rhythmic": q(rhythmic_freqs),
        "dng100_driven_recruited_all": bool(all(r["dng100_driven_max_hz"] > 1e-8 for r in rows)),
        "rates_valid_all": bool(all(r["rates_valid"] for r in rows)),
        "seconds_per_replicate_median": float(np.median([r["seconds"] for r in rows])),
        "roles": roles,
    }


def run(
    out: Path,
    *,
    manc_portcheck: Path,
    seed: int = 1,
    param_block: int = 1024,
    replicate_indices: list[int] | None = None,
    stim_bodies: tuple[int, ...] = (10056,),
    stim_amplitude: float | None = None,
    silence_roles: tuple[str, ...] = (),
    shuffle: bool = False,
    workers: int | None = None,
    save_traces: int = 8,
) -> dict:
    from experiment.pugliese_rate_mcns_portcheck import _require_manc_port
    from experiment.pugliese_rate_malecns import _jsonable
    from flybrain.authors_mcns import (
        AUTHORS_COMMIT,
        AUTHORS_MCNS_DNG100_STIM_I,
        AUTHORS_MCNS_STIM_CITATION,
        POPULATION_AUTHORS,
        STIMULUS_AMPLITUDE_SWEEP,
        STIMULUS_AUTHORS_MCNS,
        W_PATH,
        WTABLE_PATH,
        export_summary,
        load_authors_W,
        load_authors_wtable,
    )
    from flybrain.malecns_volume import SIZE_NORM_PUGLIESE_SUBNETWORK
    from flybrain.neurons import MALECNS_DNG100_L, MALECNS_DNG100_R, PUGLIESE_RATE_MALECNS_V1
    from flybrain.pugliese_rate import (
        PuglieseRateParams,
        rates_are_valid,
        reweight_connectivity,
        sample_cell_parameters_block,
        set_sizes,
    )
    from organism.roi_innervation import CORE_CPG_TYPES

    out = Path(out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {out}")
    amp = AUTHORS_MCNS_DNG100_STIM_I if stim_amplitude is None else float(stim_amplitude)
    source = STIMULUS_AUTHORS_MCNS if amp == AUTHORS_MCNS_DNG100_STIM_I else STIMULUS_AMPLITUDE_SWEEP
    allowed = {int(MALECNS_DNG100_R), int(MALECNS_DNG100_L)}
    if not stim_bodies or any(int(b) not in allowed for b in stim_bodies):
        raise ValueError(f"stim_bodies must be a subset of {sorted(allowed)}")
    for role in silence_roles:
        if role not in CORE_CPG_TYPES:
            raise ValueError(f"Unknown CPG role {role}; known {sorted(CORE_CPG_TYPES)}")
    if replicate_indices is None:
        replicate_indices = list(range(64))
    if any(i < 0 or i >= param_block for i in replicate_indices):
        raise ValueError("replicate index outside the parameter block")

    manc = _require_manc_port(manc_portcheck)
    summary = export_summary()
    table = load_authors_wtable()
    body = table["bodyId"].to_numpy(dtype=np.int64)
    stim_idx = [int(np.flatnonzero(body == int(b))[0]) for b in stim_bodies]
    mn_idx = np.flatnonzero(table["motor module"].notna().to_numpy())
    types = table["type"].astype(str)
    role_idx = {r: np.flatnonzero(types.eq(t).to_numpy()) for r, t in CORE_CPG_TYPES.items()}
    role_idx["DNg100"] = np.flatnonzero(types.eq("DNg100").to_numpy())
    cls = table["class"].astype(str).str.lower()
    is_dn = cls.str.contains("descending").to_numpy()
    is_mn = np.zeros(len(table), dtype=bool)
    is_mn[mn_idx] = True
    is_exc = table["consensusNt"].astype(str).eq("acetylcholine").to_numpy()
    remove = np.concatenate([role_idx[r] for r in silence_roles]).astype(int) if silence_roles else np.zeros(0, int)
    if np.intersect1d(remove, stim_idx).size:
        raise ValueError("Cannot silence the stimulated neuron")

    params = PuglieseRateParams(stim_amplitude=amp)
    arm = "intact"
    if silence_roles:
        arm = "silence:" + "+".join(silence_roles)
    if shuffle:
        arm = "shuffle" if arm == "intact" else arm + "+shuffle"

    out.mkdir(parents=True, exist_ok=False)
    report = {
        "model_id": PUGLIESE_RATE_MALECNS_V1,
        "experiment": "mcns_dng100_authors_export",
        "arm": arm,
        "population": POPULATION_AUTHORS,
        "authors_commit": AUTHORS_COMMIT,
        "wtable": str(WTABLE_PATH.relative_to(ROOT)),
        "W": str(W_PATH.relative_to(ROOT)),
        "wtable_sha256": summary["wtable_sha256"],
        "W_sha256": summary["W_sha256"],
        "n_neurons": int(summary["n"]),
        "n_motor_neurons": int(mn_idx.size),
        "size_normalization_mode": SIZE_NORM_PUGLIESE_SUBNETWORK,
        "size_median": float(summary["size"]["nanmedian"]),
        "stim_amplitude": amp,
        "stimulus_source": source,
        "stimulus_citation": AUTHORS_MCNS_STIM_CITATION if source == STIMULUS_AUTHORS_MCNS else None,
        "stimulated_bodyIds": [int(b) for b in stim_bodies],
        "stimulated_matrix_indices": stim_idx,
        "silenced_roles": list(silence_roles),
        "silenced_matrix_indices": remove.tolist(),
        "silenced_bodyIds": body[remove].tolist(),
        "shuffle": bool(shuffle),
        "parameter_seed": int(seed),
        "parameter_block": int(param_block),
        "replicate_indices": [int(i) for i in replicate_indices],
        "params": {k: getattr(params, k) for k in params.__dataclass_fields__},
        "integrator": "scipy.integrate.solve_ivp RK45, authors rtol/atol, dense output at dt",
        "readout": {
            "motor_neurons": "wTable rows with non-null 'motor module'",
            "transient_samples_dropped": TRANSIENT_SAMPLES,
            "active_rate": ACTIVE_RATE,
            "score": "authors src.utils.sim_utils.compute_oscillation_score(R[:,250:], active MN mask)",
            "rhythmic_threshold": SCORE_THRESHOLD,
        },
        "manc_portcheck": {
            "path": str(Path(manc_portcheck)),
            "status": manc.get("status"),
            "mean_trace_corrcoef": manc.get("mean_trace_corrcoef"),
        },
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    try:
        report["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
    except Exception:
        report["git_commit"] = None

    def save():
        (out / "report.json").write_text(json.dumps(_jsonable(report), indent=2, allow_nan=False) + "\n")

    save()
    try:
        sizes = table["size"].to_numpy(dtype=np.float64)
        drawn = sample_cell_parameters_block(sizes.shape[0], int(param_block), int(seed), params)
        gain, threshold = set_sizes(sizes, drawn["gain"], drawn["threshold"])
        report["parameter_sampler"] = drawn["sampler"]
        report["stimulated_threshold_by_replicate"] = {
            str(i): [float(threshold[i, s]) for s in stim_idx] for i in replicate_indices
        }
        matrix, _, _ = load_authors_W()
        weighted = reweight_connectivity(matrix, params.exc_multiplier, params.inh_multiplier)
        inputs = np.zeros(matrix.shape[0], dtype=np.float64)
        inputs[stim_idx] = amp
        if int(np.count_nonzero(inputs)) != len(stim_idx):
            raise RuntimeError("stimulus must hit only the named DNg100 columns")
        jobs = []
        for k, i in enumerate(replicate_indices):
            jobs.append(
                {
                    "replicate": int(i),
                    "tau": drawn["tau"][i],
                    "gain": gain[i],
                    "threshold": threshold[i],
                    "fr_cap": drawn["fr_cap"][i],
                    "inputs": inputs,
                    "remove": remove,
                    "shuffle_seed": (int(seed) * 100003 + int(i)) if shuffle else None,
                    "is_dn": is_dn,
                    "is_mn": is_mn,
                    "is_exc": is_exc,
                }
            )
        n_workers = workers or max(1, (os.cpu_count() or 2) - 1)
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[var] = "1"
        score_fn = _authors_score()
        params_kwargs = {k: getattr(params, k) for k in params.__dataclass_fields__}
        rows = []
        traces = {}
        with ProcessPoolExecutor(
            max_workers=n_workers, initializer=_worker_init, initargs=(weighted, params_kwargs)
        ) as pool:
            for res in pool.map(_worker_run, jobs):
                rates = res["rates"].astype(np.float64)
                i = res["replicate"]
                row = {
                    "replicate": i,
                    "seconds": res["seconds"],
                    "rates_valid": rates_are_valid(rates),
                    "paper": paper_metrics(rates, mn_idx, params.dt, score_fn),
                    "roles": {r: role_metrics(rates, ix, params.dt, score_fn) for r, ix in role_idx.items()},
                    "dng100_driven_max_hz": float(rates[stim_idx].max()),
                    "dng100_by_body_max_hz": {
                        str(int(body[j])): float(rates[j].max()) for j in role_idx["DNg100"]
                    },
                }
                rows.append(row)
                if len(traces) < save_traces:
                    traces[i] = res["rates"]
                p = row["paper"]
                print(
                    f"[{arm}] rep {i}: {res['seconds']:.1f}s  MNs={p['n_mn_active_past_transient']} "
                    f"score={p['mn_rhythmicity_score']:.3f} f={p['mn_frequency_hz']}",
                    flush=True,
                )
        report["replicates"] = rows
        report["summary"] = summarize(rows)
        np.savez_compressed(
            out / "traces.npz",
            body_ids=body,
            replicate=np.array(sorted(traces)),
            rates_hz=np.stack([traces[k] for k in sorted(traces)]),
            dt_s=params.dt,
            stim_index=np.array(stim_idx),
            mn_index=mn_idx,
        )
        s = report["summary"]
        report["status"] = (
            "rhythmic" if s["fraction_replicates_rhythmic"] >= 0.5 else
            ("mn_active_not_rhythmic" if s["fraction_replicates_any_mn_active"] > 0 else "no_motor_output")
        )
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        return report
    except Exception as exc:
        report.update(status="error", error=f"{type(exc).__name__}: {exc}")
        save()
        raise


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--manc-portcheck", type=Path, default=ROOT / "outputs" / "pugliese_rate_manc_portcheck_20260919T133800Z")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--param-block", type=int, default=1024)
    p.add_argument("--replicates", type=str, default="0:64", help="start:stop or comma list, indices into the block")
    p.add_argument("--stim-bodies", type=str, default="10056")
    p.add_argument("--stim-amplitude", type=float, default=None)
    p.add_argument("--silence", type=str, default="", help="comma list of CPG roles, e.g. E1,E2,I1")
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--workers", type=int, default=None)
    a = p.parse_args(argv)
    if ":" in a.replicates:
        lo, hi = (int(x) for x in a.replicates.split(":"))
        reps = list(range(lo, hi))
    else:
        reps = [int(x) for x in a.replicates.split(",") if x]
    bodies = tuple(int(x) for x in a.stim_bodies.split(",") if x)
    silence = tuple(x for x in a.silence.split(",") if x)
    out = a.out
    if out is None:
        tag = "+".join(map(str, bodies))
        amp = "I400" if a.stim_amplitude is None else f"I{a.stim_amplitude:g}"
        arm = ("silence_" + "_".join(silence)) if silence else ("shuffle" if a.shuffle else "intact")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = ROOT / "outputs" / f"mcns_dng100_{amp}_{tag}_{arm}_{stamp}"
    r = run(
        out,
        manc_portcheck=a.manc_portcheck,
        seed=a.seed,
        param_block=a.param_block,
        replicate_indices=reps,
        stim_bodies=bodies,
        stim_amplitude=a.stim_amplitude,
        silence_roles=silence,
        shuffle=a.shuffle,
        workers=a.workers,
    )
    print(out)
    print(json.dumps({k: r["summary"][k] for k in ("fraction_replicates_rhythmic", "fraction_replicates_any_mn_active", "mn_score_mn_active", "mn_frequency_hz_rhythmic")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
