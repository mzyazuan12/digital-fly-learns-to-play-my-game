"""Figures for Paper 1 (neural foundation). Reads saved outputs only.

    .venv/bin/python scripts/make_paper1_figures.py \
        --battery outputs/mcns_dng100_battery_20261111 \
        --compare outputs/authors_mcns_zenodo_compare_20261111 \
        --out docs/papers/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OURS = "#2a78d6"  # categorical slot 1
AUTH = "#eb6834"  # slot 2
AQUA = "#1baf7a"  # slot 3
YELL = "#eda100"  # slot 4
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#8a8985"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "legend.frameon": False,
    }
)

WTABLE = ROOT / "third_party" / "Pugliese_2026" / "data" / "imac t1 connectome data" / "wTable_20260210_vncRoisOnly.csv"
ROLES = {"E1": "IN17A001", "E2": "INXXX466", "I1": "IN16B036", "I2": "IN19A007"}


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def fig_traces(battery: Path, out: Path) -> None:
    table = pd.read_csv(WTABLE, index_col=0)
    d = np.load(battery / "intact_10056" / "traces.npz")
    reps = list(d["replicate"])
    R = d["rates_hz"][reps.index(0)].astype(float)
    dt = float(d["dt_s"])
    t = np.arange(R.shape[1]) * dt
    win = t <= 0.5
    mn = d["mn_index"]
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 6.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.6, 2.6]})
    ax = axes[0]
    ax.plot(t[win], R[9, win], color=NEUTRAL, lw=2)
    ax.set_ylabel("rate (Hz)")
    ax.set_title("DNg100_R (body 10056), driven at I = 400", loc="left", fontsize=9, color=INK)
    ax = axes[1]
    for (role, typ), color in zip(ROLES.items(), (OURS, AUTH, AQUA, YELL)):
        idx = np.flatnonzero(table["type"].eq(typ).to_numpy())
        k = idx[np.argmax(R[idx].max(axis=1))]
        ax.plot(t[win], R[k, win], color=color, lw=1.6, label=f"{role} ({typ}, body {int(table['bodyId'].iloc[k])})")
    ax.set_ylabel("rate (Hz)")
    ax.set_title("Core CPG interneurons (active copy of each type)", loc="left", fontsize=9, color=INK)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax = axes[2]
    active = [i for i in mn if R[i, 250:].max() > 0.01]
    active = sorted(active, key=lambda i: str(table["motor module"].iloc[i]))
    span = 1.1 * max(R[i, win].max() for i in active) if active else 1
    for j, i in enumerate(active):
        ax.plot(t[win], R[i, win] + j * span, color=INK2, lw=1.0)
        ax.text(0.505, j * span + 0.2 * span, str(table["motor module"].iloc[i]), fontsize=6, color=INK2, va="bottom")
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    ax.set_title(f"Active leg motor neurons (n = {len(active)}), offset for clarity", loc="left", fontsize=9, color=INK)
    fig.align_ylabels()
    fig.savefig(out / "fig1_traces_rep0.png")
    plt.close(fig)


def fig_authors_overlay(compare: Path, out: Path, rep: int = 954) -> None:
    table = pd.read_csv(WTABLE, index_col=0)
    f = compare / f"trace_{rep}.npz"
    if not f.exists():
        rep = 0
        f = compare / f"trace_{rep}.npz"
    d = np.load(f)
    A = d["authors"].astype(float)
    O = d["ours"].astype(float)
    mn = d["mn_index"]
    t = np.arange(A.shape[1]) * 0.001
    win = t <= 0.5
    active = [i for i in mn if max(A[i, 250:].max(), O[i, 250:].max()) > 0.01]
    active = sorted(active, key=lambda i: str(table["motor module"].iloc[i]))
    report = load(compare / "report.json")
    row = next(r for r in report["traces"] if r["replicate"] == rep)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    span = 1.1 * max(max(A[i, win].max(), O[i, win].max()) for i in active)
    for j, i in enumerate(active):
        ax.plot(t[win], A[i, win] + j * span, color=AUTH, lw=2.4, label="authors (Zenodo run 33241780)" if j == 0 else None)
        ax.plot(t[win], O[i, win] + j * span, color=OURS, lw=1.0, label="this work, same parameters" if j == 0 else None)
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    ax.set_title(
        f"Replicate {rep}: active motor neurons, r = {row['corrcoef_active']:.6f} over all active cells",
        loc="left",
        fontsize=9,
    )
    ax.legend(loc="upper right", fontsize=7)
    fig.savefig(out / "fig2_authors_overlay.png")
    plt.close(fig)


def fig_population(battery: Path, compare: Path, out: Path) -> None:
    ours = load(battery / "intact_10056" / "report.json")["replicates"]
    auth = load(compare / "authors_replicates.json")
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.6))
    so = np.array([r["paper"]["mn_rhythmicity_score"] for r in ours])
    sa = np.array([r["paper"]["mn_rhythmicity_score"] for r in auth])
    bins = np.linspace(0, 1.0001, 21)
    ax = axes[0]
    ax.hist(sa, bins=bins, color=AUTH, alpha=0.55, label=f"authors, seed 626 (n={sa.size})")
    ax.hist(so, bins=bins, histtype="step", color=OURS, lw=2, label=f"this work, seed 1 (n={so.size})")
    ax.set_xlabel("motor rhythmicity score")
    ax.set_ylabel("replicates")
    ax.axvline(0.5, color=INK2, lw=0.8, ls=":")
    no = np.array([r["paper"]["n_mn_active_past_transient"] for r in ours])
    na = np.array([r["paper"]["n_mn_active_past_transient"] for r in auth])
    ax = axes[1]
    ks = np.arange(0, max(no.max(), na.max()) + 1)
    w = 0.4
    ax.bar(ks - w / 2, [(na == k).sum() for k in ks], width=w, color=AUTH, alpha=0.75)
    ax.bar(ks + w / 2, [(no == k).sum() for k in ks], width=w, color=OURS)
    ax.set_xlabel("# motor neurons active")
    fo = np.array([r["paper"]["mn_frequency_hz"] for r in ours if r["paper"]["rhythmic"] and r["paper"]["mn_frequency_hz"]])
    fa = np.array([r["paper"]["mn_frequency_hz"] for r in auth if r["paper"]["rhythmic"] and r["paper"]["mn_frequency_hz"]])
    fb = np.linspace(min(fo.min(), fa.min()) - 0.5, max(fo.max(), fa.max()) + 0.5, 25)
    ax = axes[2]
    ax.hist(fa, bins=fb, color=AUTH, alpha=0.55)
    ax.hist(fo, bins=fb, histtype="step", color=OURS, lw=2)
    ax.set_xlabel("MN rhythm frequency (Hz)")
    handles = [
        plt.Line2D([], [], color=AUTH, lw=6, alpha=0.6, label=f"authors, seed 626 (n={sa.size})"),
        plt.Line2D([], [], color=OURS, lw=2, label=f"this work, seed 1 (n={so.size})"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08), fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig3_population.png")
    plt.close(fig)


ARM_LABELS = [
    ("intact_10056", "intact"),
    ("silence_E1_10056", "E1 removed"),
    ("silence_E2_10056", "E2 removed"),
    ("silence_E3_10056", "E3 removed"),
    ("silence_I1_10056", "I1 removed"),
    ("silence_I2_10056", "I2 removed"),
    ("shuffle_10056", "shuffled wiring (null)"),
    ("intact_10045", "drive 10045 instead"),
    ("intact_both", "drive both DNg100"),
]


def fig_perturbations(battery: Path, out: Path) -> list[dict]:
    rows = []
    for arm, label in ARM_LABELS:
        p = battery / arm / "report.json"
        if not p.exists():
            continue
        r = load(p)
        s = r["summary"]
        reps = r["replicates"]
        nmn = np.array([x["paper"]["n_mn_active_past_transient"] for x in reps])
        rows.append(
            {
                "arm": arm,
                "label": label,
                "n": s["n_replicates"],
                "fraction_rhythmic": s["fraction_replicates_rhythmic"],
                "fraction_any_mn": s["fraction_replicates_any_mn_active"],
                "median_score": (s["mn_score_all"] or {}).get("median"),
                "median_n_mn": float(np.median(nmn)),
                "median_freq": (s["mn_frequency_hz_rhythmic"] or {}).get("median"),
            }
        )
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, key, xl in (
        (axes[0], "fraction_rhythmic", "fraction of replicates rhythmic (score ≥ 0.5)"),
        (axes[1], "median_n_mn", "median # motor neurons active"),
    ):
        vals = [r[key] for r in rows]
        colors = [OURS if r["arm"] == "intact_10056" else NEUTRAL for r in rows]
        ax.barh(y, vals, color=colors, height=0.62)
        for yy, v in zip(y, vals):
            ax.text(v, yy, f" {v:.2f}" if key == "fraction_rhythmic" else f" {v:.0f}", va="center", fontsize=7, color=INK)
        ax.set_xlabel(xl, fontsize=8)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [f"{r['label']} (n={r['n']})" for r in rows], fontsize=8)
    axes[0].set_xlim(0, 1.15)
    fig.tight_layout()
    fig.savefig(out / "fig4_perturbations.png")
    plt.close(fig)
    return rows


def fig_dose(battery: Path, out: Path) -> list[dict]:
    pts = []
    for p in sorted(battery.glob("sweep_I*_10056/report.json")):
        r = load(p)
        pts.append((r["stim_amplitude"], r))
    p400 = battery / "intact_10056" / "report.json"
    if p400.exists():
        pts.append((400.0, load(p400)))
    pts.sort(key=lambda x: x[0])
    rows = []
    for amp, r in pts:
        s = r["summary"]
        f = s["mn_frequency_hz_rhythmic"] or {}
        rows.append(
            {
                "I": amp,
                "n": s["n_replicates"],
                "fraction_rhythmic": s["fraction_replicates_rhythmic"],
                "fraction_any_mn": s["fraction_replicates_any_mn_active"],
                "f_median": f.get("median"),
                "f_q25": f.get("q25"),
                "f_q75": f.get("q75"),
                "n_mn_median": (s["n_mn_active_past_transient"] or {}).get("median"),
            }
        )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    I = [r["I"] for r in rows]
    ax = axes[0]
    ax.plot(I, [r["fraction_rhythmic"] for r in rows], "-o", color=OURS, lw=2, ms=5)
    ax.set_xlabel("DNg100 drive I (a.u.)")
    ax.set_ylabel("fraction rhythmic")
    ax.set_ylim(-0.03, 1.05)
    ax.axvline(400, color=INK2, lw=0.8, ls=":")
    ax.text(405, 0.05, "published\nCNS drive", fontsize=7, color=INK2)
    ax = axes[1]
    ok = [r for r in rows if r["f_median"] is not None]
    ax.errorbar(
        [r["I"] for r in ok],
        [r["f_median"] for r in ok],
        yerr=[[r["f_median"] - r["f_q25"] for r in ok], [r["f_q75"] - r["f_median"] for r in ok]],
        fmt="-o",
        color=OURS,
        lw=2,
        ms=5,
        capsize=2,
    )
    ax.set_xlabel("DNg100 drive I (a.u.)")
    ax.set_ylabel("rhythm frequency (Hz), median ± IQR")
    ax.axvline(400, color=INK2, lw=0.8, ls=":")
    fig.tight_layout()
    fig.savefig(out / "fig5_dose_response.png")
    plt.close(fig)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--battery", type=Path, required=True)
    ap.add_argument("--compare", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "papers" / "figures")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fig_traces(a.battery, a.out)
    fig_authors_overlay(a.compare, a.out)
    fig_population(a.battery, a.compare, a.out)
    pert = fig_perturbations(a.battery, a.out)
    dose = fig_dose(a.battery, a.out)
    (a.out / "figure_data.json").write_text(json.dumps({"perturbations": pert, "dose_response": dose}, indent=2) + "\n")
    print(json.dumps({"perturbations": pert, "dose_response": dose}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
