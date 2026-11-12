"""Stage 2B runner (design §6.3, §8 row 9; amendments A1-A7; plan step 9).

Runs the pre-registered conditions on test replicates of the seed-1 block with
the frozen bridge, and writes per-run traces. It computes **no** phase, lag,
coherence or coordination read-out (enforced by tests/test_paper2_stage2B_run.py).
Read-outs are plan step 10.

Refuses to start unless:
  - SHA-256 of data/paper2/stage2B_bridge_frozen.json equals the hash frozen
    by amendment A7;
  - the bridge, co-simulation and stage-2B data files are clean in git;
  - every code and data hash recorded in the frozen file matches the current file;
  - the output directory does not exist yet;
  - no requested replicate is a calibration replicate (960-1023).

Arms (the frozen file is position-actuated; amendment A7):

  primary   position actuation, r* = 100 Hz. Per replicate: open loop and
            deafferented once, plus closed loop, scrambled and mismatched for
            each of the 4 hair-plate cells (A1.5) = 14 runs.
  R2_r50    r* = 50 Hz. Per replicate: closed loop, scrambled and mismatched
            for each of the 4 cells = 12 runs. Open loop and deafferented do
            not use afferents, so they are identical to the primary's runs and
            are not repeated.
  R2_r150   as R2_r50 with r* = 150 Hz (A5).
  R1_torque torque actuation. Needs its own torque-calibrated frozen bridge,
            which does not exist yet, so the runner refuses it. Once
            calibrated, it needs 13 runs per replicate: the 12 feedback runs plus
            deafferented once, because the body differs. Its neural trace still
            equals open loop.

Robustness arms run every factorial cell. Choosing cells after seeing the
primary result would be a selection.

Within a (replicate, cell) job the closed loop is written to disk before its
scrambled run starts. The scrambled run uses that closed loop's every-tick body
state, held in memory, with the frozen seed rule.

Per run, rates are saved at 1 ms for the motor neurons, the core CPG cells (E1,
E2, E3, I1, I2; both copies), both DNg100 and the 31 included afferents. Body
state (angle, velocity, actuator force, command; both front legs) and afferent
input current are saved too. Failures are written and kept.

  .venv/bin/python -m experiment.paper2_stage2B_run --arm primary --workers 9
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FROZEN_PATH = ROOT / "data" / "paper2" / "stage2B_bridge_frozen.json"
EXPECTED_FROZEN_SHA256 = "91bc54c33dd1c1e60199ece5f6a0c17f947393dc9022963e4d221b6d456fac25"  # amendment A7
GUARDED_PATHS = (
    "flybrain/rate_stepper.py",
    "organism/flybody_tethered.py",
    "organism/mn_joint_bridge.py",
    "organism/proprio_transduction.py",
    "experiment/paper2_cosim.py",
    "experiment/paper2_calibrate_2B.py",
    "data/paper2/stage2B_bridge_frozen.json",
    "data/paper2/stage2B_sensory_table.json",
    "data/paper2/stage2B_sensory_norm.json",
    "data/paper2/stage2B_mn_joint_table.json",
    "data/paper2/stage2B_body_transfer.json",
)
CALIBRATION_REPLICATES = range(960, 1024)
PRIMARY_REPLICATES = tuple(range(0, 64))  # protocol: >= 64 replicates
PARAM_BLOCK = 1024
CORE_CPG_TYPES = {"E1": "IN17A001", "E2": "INXXX466", "E3": "IN19B012", "I1": "IN16B036", "I2": "IN19A007"}
FEEDBACK_ORDER = ("closed_loop", "scrambled", "mismatched")  # closed loop first: scrambled needs it
ARMS = {
    "primary": {"actuation": "position", "r_star": "primary", "base": ("open_loop", "deafferented")},
    "R2_r50": {"actuation": "position", "r_star": "R2_lower", "base": ()},
    "R2_r150": {"actuation": "position", "r_star": "R2_upper", "base": ()},
    "R1_torque": {"actuation": "torque", "r_star": "primary", "base": ("deafferented",)},
}

_W: dict = {}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cell_name(cell: dict) -> str:
    return f"{cell['hair_plate_coxa']}-{cell['hair_plate_trochanter']}"


# ----------------------------------------------------------------------------- gates
def git_dirty(paths=GUARDED_PATHS) -> list[str]:
    out = subprocess.check_output(["git", "status", "--porcelain", "--", *paths], text=True, cwd=ROOT)
    return [line for line in out.splitlines() if line.strip()]


def preflight(frozen_path: Path = FROZEN_PATH, expected_sha256: str = EXPECTED_FROZEN_SHA256,
              dirty_lines: list[str] | None = None) -> dict:
    """All refusals that do not depend on the output path. Returns the frozen document."""
    frozen_path = Path(frozen_path)
    got = sha256(frozen_path)
    if got != expected_sha256:
        raise RuntimeError(f"frozen bridge SHA-256 {got} != {expected_sha256} (amendment A7); refusing to run")
    dirty = git_dirty() if dirty_lines is None else dirty_lines
    if dirty:
        raise RuntimeError("bridge/co-simulation files are not clean in git; refusing to run:\n" + "\n".join(dirty))
    doc = json.loads(frozen_path.read_text())
    for rel, h in doc["code_sha256"].items():
        if sha256(ROOT / rel) != h:
            raise RuntimeError(f"{rel} differs from the code frozen with the bridge; refusing to run")
    from flybrain.authors_mcns import W_PATH, WTABLE_PATH

    data = ROOT / "data" / "paper2"
    clip = Path(__import__("flygym_demo").__file__).resolve().parent / "ball_flybody_data" / "assets" / "ball_flybody_clip.npz"
    paths = {"wTable": WTABLE_PATH, "W": W_PATH, "sensory_table": data / "stage2B_sensory_table.json",
             "sensory_norm": data / "stage2B_sensory_norm.json", "mn_joint_table": data / "stage2B_mn_joint_table.json",
             "body_transfer": data / "stage2B_body_transfer.json", "walking_clip": clip}
    for key, h in doc["data_sha256"].items():
        if sha256(paths[key]) != h:
            raise RuntimeError(f"data file {key} differs from the one frozen with the bridge; refusing to run")
    return doc


def check_replicates(reps) -> tuple[int, ...]:
    reps = tuple(int(r) for r in reps)
    for r in reps:
        if r in CALIBRATION_REPLICATES:
            raise ValueError(f"replicate {r} is a calibration replicate (960-1023)")
        if not 0 <= r < PARAM_BLOCK:
            raise ValueError(f"replicate {r} is outside the seed-1 block")
    return reps


def arm_settings(arm: str, frozen: dict) -> dict:
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {sorted(ARMS)}")
    spec = ARMS[arm]
    if spec["actuation"] != frozen["bridge_params"]["mode"]:
        raise RuntimeError(
            f"arm {arm} needs a {spec['actuation']}-calibrated frozen bridge; the frozen bridge is "
            f"{frozen['bridge_params']['mode']} (R1 torque calibration has not been run)"
        )
    r = {"primary": frozen["sensory"]["r_star_hz"], "R2_lower": frozen["sensory"]["R2_r_star_hz"]["lower"],
         "R2_upper": frozen["sensory"]["R2_r_star_hz"]["upper"]}[spec["r_star"]]
    if r is None:
        raise RuntimeError(f"arm {arm}: r* is not set in the frozen bridge")
    return {"arm": arm, "actuation": spec["actuation"], "r_star_hz": float(r), "base": spec["base"]}


# ----------------------------------------------------------------------------- recording rows
def recorded_rows() -> dict:
    """Motor neurons, core CPG cells, both DNg100 and the included afferents."""
    from flybrain.authors_mcns import load_authors_wtable

    t = load_authors_wtable()
    st = json.loads((ROOT / "data" / "paper2" / "stage2B_sensory_table.json").read_text())
    types = t["type"].astype(str).to_numpy()
    side = t["somaSide"].astype(str).to_numpy()
    rows, labels = [], []
    for i in np.flatnonzero(t["motor module"].notna().to_numpy()):
        rows.append(int(i))
        labels.append(f"MN|{t['motor module'].iloc[i]}|{side[i]}|{types[i]}")
    for role, typ in CORE_CPG_TYPES.items():
        for i in np.flatnonzero(types == typ):
            rows.append(int(i))
            labels.append(f"CPG|{role}|{side[i]}")
    for i in np.flatnonzero(types == "DNg100"):
        rows.append(int(i))
        labels.append(f"DN|DNg100|{t['instance'].iloc[i]}")
    for a in st["afferents"]:
        if a["included_2B"]:
            rows.append(int(a["row"]))
            labels.append(f"AFF|{a['leg']}|{a['channel']}|{a['type']}")
    if len(set(rows)) != len(rows):
        raise RuntimeError("recorded rows overlap")
    body_ids = t["bodyId"].to_numpy(np.int64)[rows]
    return {"rows": np.array(rows, dtype=np.int64), "labels": labels, "bodyIds": body_ids}


# ----------------------------------------------------------------------------- worker
def _setup(settings: dict, frozen: dict) -> None:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    if _W.get("key") == (settings["arm"], frozen["content_sha256"]):
        return
    from experiment.paper2_cosim import Network
    from organism.flybody_tethered import TetheredFlyBody
    from organism.mn_joint_bridge import BridgeParams, MNJointBridge

    _W.clear()
    _W["key"] = (settings["arm"], frozen["content_sha256"])
    _W["net"] = Network.load(frozen["parameter_block"]["seed"], frozen["parameter_block"]["n_replicates"])
    _W["body"] = TetheredFlyBody(settings["actuation"])
    _W["bridge"] = MNJointBridge()
    _W["params"] = BridgeParams.from_dict(frozen["bridge_params"])
    _W["rec"] = recorded_rows()
    _W["transducers"] = {}


def _transducer(cell: dict, r_star: float):
    from organism.proprio_transduction import ProprioTransducer

    key = (cell_name(cell), r_star)
    if key not in _W["transducers"]:
        _W["transducers"][key] = ProprioTransducer(cell, r_star_hz=r_star)
    return _W["transducers"][key]


def _save(path: Path, res: dict, meta: dict) -> None:
    from flybrain.pugliese_rate import RATE_MAX_HZ

    rec = _W["rec"]
    full = res["rates_hz"]
    meta["rates_valid"] = bool(np.all(np.isfinite(full)) and full.min() >= 0.0 and full.max() <= RATE_MAX_HZ)
    meta["max_rate_hz"] = float(np.nanmax(full))
    arrays = {"rates_hz": full[rec["rows"]], "rows": rec["rows"], "bodyIds": rec["bodyIds"],
              "labels": np.array(rec["labels"]), "dt_s": np.array(res["dt_s"])}
    for k in ("angle", "velocity", "actuator_force", "command", "afferent_rows", "afferent_current"):
        if k in res:
            arrays[k] = res[k]
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **arrays)
    os.replace(tmp, path)


def _run_one(out_dir: Path, settings: dict, frozen: dict, rep: int, condition: str, cell: dict | None,
             surrogate=None, keep_state=False):
    from experiment.paper2_cosim import RunConfig, run

    name = f"rep{rep:03d}_{condition}" + (f"_{cell_name(cell)}" if cell else "")
    meta = {"arm": settings["arm"], "condition": condition, "cell": cell, "replicate": rep,
            "r_star_hz": settings["r_star_hz"] if cell else None, "actuation": settings["actuation"],
            "frozen_sha256": EXPECTED_FROZEN_SHA256, "file": f"{name}.npz"}
    t0 = time.time()
    try:
        cfg = RunConfig(condition, rep, seed=frozen["parameter_block"]["seed"], T=frozen["integrator"]["T_s"],
                        hair_plate_directions=cell, r_star_hz=settings["r_star_hz"],
                        record_state_every_tick=keep_state)
        res = run(cfg, net=_W["net"], params=_W["params"], body=_W["body"], bridge=_W["bridge"],
                  transducer=_transducer(cell, settings["r_star_hz"]) if cell else None,
                  surrogate_source=surrogate)
        meta["seconds"] = round(time.time() - t0, 2)
        meta["status"] = "ok"
        _save(out_dir / "runs" / f"{name}.npz", res, meta)
        return meta, res.get("state_every_tick")
    except Exception as exc:  # failures are kept, not dropped
        meta.update(status="error", seconds=round(time.time() - t0, 2), error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc())
        (out_dir / "runs" / f"{name}.error.json").write_text(json.dumps(meta, indent=1))
        return meta, None


def job(args) -> list[dict]:
    """One job: the base conditions of a replicate, or the feedback conditions of a
    (replicate, cell), with the closed loop written before its scrambled run."""
    out_dir, settings, frozen, rep, cell, conditions = args
    _setup(settings, frozen)
    metas = []
    if cell is None:
        for c in conditions:
            metas.append(_run_one(out_dir, settings, frozen, rep, c, None)[0])
        return metas
    state = None
    for c in FEEDBACK_ORDER:
        if c not in conditions:
            continue
        if c == "scrambled":
            if state is None:
                metas.append({"condition": c, "replicate": rep, "cell": cell, "status": "error",
                              "error": "closed loop of this replicate/cell missing or failed"})
                continue
            metas.append(_run_one(out_dir, settings, frozen, rep, c, cell, surrogate=state)[0])
        else:
            meta, st = _run_one(out_dir, settings, frozen, rep, c, cell, keep_state=(c == "closed_loop"))
            if c == "closed_loop":
                state = st if meta["status"] == "ok" else None
            metas.append(meta)
    return metas


def plan_jobs(out_dir: Path, settings: dict, frozen: dict, reps, conditions, cells) -> list:
    jobs = []
    base = [c for c in settings["base"] if c in conditions]
    fb = [c for c in FEEDBACK_ORDER if c in conditions]
    if "scrambled" in fb and "closed_loop" not in fb:
        raise ValueError("scrambled needs closed_loop in the same run")
    for rep in reps:
        if base:
            jobs.append((out_dir, settings, frozen, rep, None, tuple(base)))
        for cell in cells:
            if fb:
                jobs.append((out_dir, settings, frozen, rep, cell, tuple(fb)))
    return jobs


# ----------------------------------------------------------------------------- main
def run_arm(arm: str, *, replicates=PRIMARY_REPLICATES, out: Path | None = None, workers: int = 9,
            conditions=None, cells=None, frozen_path: Path = FROZEN_PATH, dirty_lines=None) -> dict:
    from experiment.paper2_cosim import CONDITIONS
    from organism.proprio_transduction import FACTORIAL

    reps = check_replicates(replicates)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(out) if out is not None else ROOT / "outputs" / f"paper2_stage2B_{arm}_{stamp}"
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    frozen = preflight(frozen_path, dirty_lines=dirty_lines)
    settings = arm_settings(arm, frozen)
    conditions = tuple(CONDITIONS) if conditions is None else tuple(conditions)
    cells = [dict(c) for c in FACTORIAL] if cells is None else [dict(c) for c in cells]
    jobs = plan_jobs(out, settings, frozen, reps, conditions, cells)
    (out / "runs").mkdir(parents=True)
    report = {
        "what": "Stage 2B runs (plan step 9); traces only, no read-outs",
        "arm": settings, "replicates": list(reps), "conditions": list(conditions),
        "cells": cells, "frozen_sha256": EXPECTED_FROZEN_SHA256, "frozen_status": frozen["status"],
        "runner_sha256": sha256(Path(__file__)),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip(),
        "runner_git_status": subprocess.check_output(["git", "status", "--porcelain", "--", "experiment/paper2_stage2B_run.py"],
                                                     text=True, cwd=ROOT).strip(),
        "recorded": "rates at 1 ms for MNs, core CPG (E1,E2,E3,I1,I2), DNg100, included afferents; body state; afferent current",
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "n_jobs": len(jobs), "runs": [],
    }
    rp = out / "report.json"
    rp.write_text(json.dumps(report, indent=1))
    t0 = time.time()
    if workers <= 0:
        for j in jobs:
            report["runs"].extend(job(j))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(job, j) for j in jobs]
            for i, f in enumerate(as_completed(futs)):
                try:
                    report["runs"].extend(f.result())
                except Exception as exc:  # a crashed worker is a failure, kept
                    report["runs"].append({"status": "error", "error": f"job crashed: {type(exc).__name__}: {exc}"})
                if i % 9 == 0:
                    report["elapsed_s"] = round(time.time() - t0, 1)
                    rp.write_text(json.dumps(report, indent=1))
    report["runs"].sort(key=lambda m: (m.get("replicate", -1), str(m.get("cell")), m.get("condition", "")))
    report["n_runs"] = len(report["runs"])
    report["n_ok"] = sum(m.get("status") == "ok" for m in report["runs"])
    report["n_failed"] = report["n_runs"] - report["n_ok"]
    report["n_rates_invalid"] = sum(m.get("status") == "ok" and not m.get("rates_valid") for m in report["runs"])
    report["elapsed_s"] = round(time.time() - t0, 1)
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["status"] = "finished"
    rp.write_text(json.dumps(report, indent=1))
    report["out"] = str(out)
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, choices=sorted(ARMS))
    p.add_argument("--replicates", type=str, default="0:64", help="start:stop (calibration replicates refused)")
    p.add_argument("--workers", type=int, default=9)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    lo, hi = (int(x) for x in a.replicates.split(":"))
    r = run_arm(a.arm, replicates=range(lo, hi), out=a.out, workers=a.workers)
    print(json.dumps({k: r[k] for k in ("out", "n_runs", "n_ok", "n_failed", "n_rates_invalid", "elapsed_s")}, indent=1))
    return 0 if r["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
