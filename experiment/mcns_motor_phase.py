"""Phase structure of DNg100-driven motor output in the authors' 1,024 mCNS runs.

For each replicate: active motor neurons (r > 0.01 after 250 ms), their
soma side, and the phase of each motor module's mean trace relative to the
coxa-swing module, from the cross-correlation peak within one period.
Phase is in cycles: 0 = in phase, 0.5 = antiphase. Also records which copy
(soma side) of each CPG type is active.

Reads data/zenodo_pugliese_22260924 (see experiment.authors_mcns_zenodo_compare).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRANSIENT = 250
REFERENCE = "coxa swing"


def module_phase(a: np.ndarray, b: np.ndarray, period: float) -> float | None:
    """Lag of b relative to a, in cycles in [0, 1), from the xcorr peak."""
    a = a - a.mean()
    b = b - b.mean()
    if a.std() < 1e-9 or b.std() < 1e-9 or not np.isfinite(period) or period < 2:
        return None
    n = a.size
    max_lag = int(round(period))
    lags = np.arange(0, max_lag + 1)
    cc = np.array([np.dot(a[: n - k], b[k:]) / (n - k) for k in lags])
    k = int(lags[np.argmax(cc)])
    return float((k / period) % 1.0)


def run(out: Path) -> dict:
    import jax.numpy as jnp
    import pandas as pd

    from experiment.authors_mcns_zenodo_compare import stream_replicates
    from experiment.pugliese_rate_mcns_dng100 import _authors_score
    from flybrain.authors_mcns import load_authors_wtable
    from organism.roi_innervation import CORE_CPG_TYPES

    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    table = load_authors_wtable()
    module = table["motor module"]
    mn = np.flatnonzero(module.notna().to_numpy())
    side = table["somaSide"].astype(str).to_numpy()
    types = table["type"].astype(str)
    score_fn = _authors_score()
    rows = []
    for rep, R in stream_replicates():
        past = R[:, TRANSIENT:].astype(np.float64)
        act = [i for i in mn if past[i].max() > 0.01]
        if not act:
            rows.append({"replicate": rep, "n_active": 0})
            continue
        _, freq = score_fn(jnp.asarray(past[act]), jnp.ones(len(act), dtype=bool))
        freq = float(freq)
        period = 1.0 / freq if np.isfinite(freq) and freq > 0 else np.nan
        mods = {}
        for i in act:
            x = past[i]
            z = (x - x.mean()) / (x.std() + 1e-12)
            mods.setdefault(str(module.iloc[i]), []).append(z)
        mean = {m: np.mean(v, axis=0) for m, v in mods.items()}
        phases = {}
        if REFERENCE in mean:
            for m, tr in mean.items():
                if m != REFERENCE:
                    phases[m] = module_phase(mean[REFERENCE], tr, period)
        cpg_side = {}
        for role, typ in CORE_CPG_TYPES.items():
            idx = np.flatnonzero(types.eq(typ).to_numpy())
            cpg_side[role] = sorted({side[i] for i in idx if past[i].max() > 0.01})
        rows.append(
            {
                "replicate": rep,
                "n_active": len(act),
                "active_mn_sides": sorted({side[i] for i in act}),
                "modules_active": {m: len(v) for m, v in mods.items()},
                "period_samples": None if not np.isfinite(period) else float(period),
                "phase_vs_coxa_swing": phases,
                "cpg_active_sides": cpg_side,
            }
        )
    # summary
    summary: dict = {"n_replicates": len(rows)}
    sides = [tuple(r.get("active_mn_sides", [])) for r in rows]
    summary["active_mn_sides_counts"] = {"+".join(k) or "none": sides.count(k) for k in set(sides)}
    mod_counts: dict = {}
    for r in rows:
        for m in r.get("modules_active", {}):
            mod_counts[m] = mod_counts.get(m, 0) + 1
    summary["fraction_replicates_module_active"] = {m: c / len(rows) for m, c in sorted(mod_counts.items())}
    ph: dict = {}
    for r in rows:
        for m, v in r.get("phase_vs_coxa_swing", {}).items():
            if v is not None:
                ph.setdefault(m, []).append(v)
    circ = {}
    for m, v in ph.items():
        ang = 2 * np.pi * np.asarray(v)
        c = np.mean(np.exp(1j * ang))
        circ[m] = {
            "n": len(v),
            "circular_mean_cycles": float((np.angle(c) / (2 * np.pi)) % 1.0),
            "resultant_length": float(np.abs(c)),
        }
    summary["phase_vs_coxa_swing"] = circ
    cs: dict = {}
    for r in rows:
        for role, s in r.get("cpg_active_sides", {}).items():
            key = "+".join(s) or "none"
            cs.setdefault(role, {}).setdefault(key, 0)
            cs[role][key] += 1
    summary["cpg_active_sides_counts"] = cs
    report = {
        "source": "Zenodo 22260924 DNg100_Stim_IMAC_vncOnly run 33241780 (authors' 1,024 replicates)",
        "reference_module": REFERENCE,
        "phase_definition": "cross-correlation peak lag of module mean z-trace vs coxa swing, in cycles [0,1)",
        "summary": summary,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "replicates.json").write_text(json.dumps(rows) + "\n")
    return report


if __name__ == "__main__":
    o = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "outputs" / (
        "mcns_motor_phase_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    r = run(o)
    print(o)
    print(json.dumps(r["summary"], indent=1))
