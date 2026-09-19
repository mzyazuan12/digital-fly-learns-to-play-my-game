"""Pugliese half-tanh rate ODE, transcribed for MaleCNS transfer.

This is model_id=pugliese_rate_malecns_v1. It is not Shiu LIF and not the
authors' MANC Hydra run (pugliese_cpg_v1).

Equation (Pugliese_2026 src/simulation/vnc_sim.py::rate_equation_half_tanh):

    I(t) = inputs * 1[pulse_start <= t <= pulse_end]
    total = I + W_weighted @ R
    g = max( rmax * tanh((a / rmax) * (total - theta)), 0 )
    dR/dt = (g - R) / tau

W_weighted is the transpose of signed synapse counts, with excitatory and
inhibitory multipliers applied separately (default 0.03 / 0.03).

Gain a is divided by median-normalized neuron size; threshold is multiplied
by the same factor (src/utils/sim_utils.py::set_sizes). Amplitude 250 is
this rate-ODE convention. Do not copy it into shiu_lif_sanity_v1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import truncnorm

from flybrain.loader import Connectome, DEFAULT_DATA
from flybrain.neurons import (
    PUGLIESE_ATOL,
    PUGLIESE_CPG_STIM_AMPLITUDE,
    PUGLIESE_DT_S,
    PUGLIESE_EXC_MULTIPLIER,
    PUGLIESE_FRCAP_MEAN_HZ,
    PUGLIESE_FRCAP_STD_HZ,
    PUGLIESE_GAIN_MEAN,
    PUGLIESE_GAIN_STD,
    PUGLIESE_INH_MULTIPLIER,
    PUGLIESE_PULSE_END_S,
    PUGLIESE_PULSE_START_S,
    PUGLIESE_RATE_MALECNS_V1,
    PUGLIESE_RTOL,
    PUGLIESE_T_S,
    PUGLIESE_TAU_MEAN_S,
    PUGLIESE_TAU_STD_S,
    PUGLIESE_THRESHOLD_MEAN,
    PUGLIESE_THRESHOLD_STD,
)

SWC_DIR = DEFAULT_DATA / "skeletons-swc"
RATE_MAX_HZ = 1000.0
OSCILLATION_THRESHOLD = 0.5


@dataclass(frozen=True)
class PuglieseRateParams:
    """Published rate-ODE constants. Units are seconds and Hz, not millivolts."""

    tau_mean: float = PUGLIESE_TAU_MEAN_S
    tau_std: float = PUGLIESE_TAU_STD_S
    gain_mean: float = PUGLIESE_GAIN_MEAN
    gain_std: float = PUGLIESE_GAIN_STD
    threshold_mean: float = PUGLIESE_THRESHOLD_MEAN
    threshold_std: float = PUGLIESE_THRESHOLD_STD
    frcap_mean: float = PUGLIESE_FRCAP_MEAN_HZ
    frcap_std: float = PUGLIESE_FRCAP_STD_HZ
    exc_multiplier: float = PUGLIESE_EXC_MULTIPLIER
    inh_multiplier: float = PUGLIESE_INH_MULTIPLIER
    stim_amplitude: float = PUGLIESE_CPG_STIM_AMPLITUDE
    dt: float = PUGLIESE_DT_S
    T: float = PUGLIESE_T_S
    pulse_start: float = PUGLIESE_PULSE_START_S
    pulse_end: float = PUGLIESE_PULSE_END_S
    rtol: float = PUGLIESE_RTOL
    atol: float = PUGLIESE_ATOL

    @property
    def model_id(self) -> str:
        return PUGLIESE_RATE_MALECNS_V1

    def __post_init__(self) -> None:
        if self.stim_amplitude == 40.0:
            raise ValueError("Rate-ODE stimulus 250 is not the Shiu 40 mV drive.")
        if self.dt <= 0 or self.T <= 0:
            raise ValueError("dt and T must be positive.")
        if self.tau_mean <= 0 or self.frcap_mean <= 0:
            raise ValueError("tau and firing-rate cap means must be positive.")


def sample_trunc_normal(rng: np.random.Generator, mean: float, stdev: float, shape) -> np.ndarray:
    """Truncated-normal samples, lower bound 0. Same family as the authors' JAX helper."""
    if stdev < 1e-10:
        return np.full(shape, max(mean, 0.0), dtype=np.float64)
    upper = min(mean + min(100.0 * stdev, 1e6), 1e10)
    a = (0.0 - mean) / stdev
    b = (upper - mean) / stdev
    return np.asarray(truncnorm.rvs(a, b, loc=mean, scale=stdev, size=shape, random_state=rng), dtype=np.float64)


def set_sizes(sizes, gain, threshold):
    """Divide gain and multiply threshold by median-normalized size.

    Transcription of Pugliese_2026 src/utils/sim_utils.py::set_sizes.
    ``gain`` / ``threshold`` may be (n,) or (n_rep, n).
    """
    sizes = np.asarray(sizes, dtype=np.float64).copy()
    if sizes.ndim != 1:
        raise ValueError("sizes must be a 1-D per-neuron vector")
    norm = float(np.nanmedian(sizes))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Median neuron size must be a positive finite number")
    sizes[np.isnan(sizes)] = norm
    sizes[sizes == 0] = norm
    scale = sizes / norm
    gain = np.asarray(gain, dtype=np.float64)
    threshold = np.asarray(threshold, dtype=np.float64)
    if gain.ndim == 1:
        return gain / scale, threshold * scale
    return gain / scale[None, :], threshold * scale[None, :]


def reweight_connectivity(W: np.ndarray, exc_mult: float, inh_mult: float) -> np.ndarray:
    """Transpose signed W[pre, post] and apply excitatory/inhibitory multipliers.

    Transcription of Pugliese_2026 src/simulation/vnc_sim.py::reweight_connectivity.
    The ODE then uses (W_weighted @ R) as postsynaptic input.
    """
    wt = np.transpose(np.asarray(W, dtype=np.float64))
    return exc_mult * np.maximum(wt, 0.0) + inh_mult * np.minimum(wt, 0.0)


def signed_weight_matrix(connectome: Connectome) -> np.ndarray:
    """Dense W[pre, post] = anatomical synapse count × presynaptic NT sign."""
    n = connectome.n
    W = np.zeros((n, n), dtype=np.float64)
    pre = np.repeat(np.arange(n, dtype=np.int64), np.diff(connectome.pre_ptr))
    W[pre, connectome.post.astype(np.int64)] = (
        connectome.anatomical.astype(np.float64) * connectome.sign.astype(np.float64)
    )
    return W


def rate_equation_half_tanh(
    t: float,
    R: np.ndarray,
    *,
    inputs: np.ndarray,
    pulse_start: float,
    pulse_end: float,
    tau: np.ndarray,
    weighted_W: np.ndarray,
    threshold: np.ndarray,
    gain: np.ndarray,
    fr_cap: np.ndarray,
) -> np.ndarray:
    """Numpy transcription of rate_equation_half_tanh."""
    pulse_active = (t >= pulse_start) and (t <= pulse_end)
    I = inputs if pulse_active else np.zeros_like(inputs)
    total_input = I + weighted_W @ R
    activation = np.maximum(fr_cap * np.tanh((gain / fr_cap) * (total_input - threshold)), 0.0)
    return (activation - R) / tau


def rates_are_valid(rates: np.ndarray) -> bool:
    """Rate-model numerical gate. Not a millivolt voltage bound."""
    x = np.asarray(rates, dtype=np.float64)
    if x.size == 0:
        return True
    if not np.all(np.isfinite(x)):
        return False
    return bool(float(np.min(x)) >= -1e-9 and float(np.max(x)) <= RATE_MAX_HZ + 1e-6)


def swc_frustum_volume(path: Path) -> float:
    """Volume of truncated cones along an official MaleCNS SWC skeleton.

    The released annotation feather has no neuPrint ``size`` / volume column.
    Pugliese normalize MANC and mCNS by neuron volume. These coarse NeuTu
    skeletons are the official centerlines in this checkout; the absolute
    unit is not the MANC voxel-count property, but median-normalized ratios
    inside one simulated population are what set_sizes uses.
    """
    xyz = []
    radius = []
    parent = []
    with Path(path).open() as handle:
        for line in handle:
            if not line or line[0] == "#":
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            xyz.append((float(parts[2]), float(parts[3]), float(parts[4])))
            radius.append(float(parts[5]))
            parent.append(int(parts[6]))
    if not xyz:
        return float("nan")
    xyz = np.asarray(xyz, dtype=np.float64)
    radius = np.asarray(radius, dtype=np.float64)
    parent = np.asarray(parent, dtype=np.int64)
    volume = 0.0
    for i, p in enumerate(parent):
        if p < 1:
            continue
        j = p - 1
        if j < 0 or j >= len(xyz):
            continue
        h = float(np.linalg.norm(xyz[i] - xyz[j]))
        r1 = radius[i]
        r2 = radius[j]
        volume += (np.pi * h / 3.0) * (r1 * r1 + r1 * r2 + r2 * r2)
    return float(volume) if volume > 0 else float("nan")


def neuron_sizes_from_swc(body_ids, swc_dir: Path = SWC_DIR) -> dict:
    """Per-body SWC volume plus provenance. Missing files are NaN (median-filled later)."""
    sizes = np.full(len(body_ids), np.nan, dtype=np.float64)
    missing = []
    for i, body in enumerate(body_ids):
        path = Path(swc_dir) / f"{int(body)}.swc"
        if not path.exists():
            missing.append(int(body))
            continue
        sizes[i] = swc_frustum_volume(path)
        if not np.isfinite(sizes[i]) or sizes[i] <= 0:
            missing.append(int(body))
            sizes[i] = np.nan
    return {
        "sizes": sizes,
        "source": "MaleCNS SWC frustum volume (NeuTu coarse skeletons)",
        "swc_dir": str(Path(swc_dir).resolve()),
        "n_missing": len(missing),
        "missing_malecns_body_ids": missing,
        "not_neuprint_volume_property": True,
    }


def make_input(n: int, stim_indices, amplitude: float) -> np.ndarray:
    I = np.zeros(n, dtype=np.float64)
    idx = np.asarray(stim_indices, dtype=np.int64)
    if idx.size:
        I[idx] = float(amplitude)
    return I


def sample_cell_parameters(n: int, rng: np.random.Generator, params: PuglieseRateParams) -> dict:
    tau = sample_trunc_normal(rng, params.tau_mean, params.tau_std, n)
    gain = sample_trunc_normal(rng, params.gain_mean, params.gain_std, n)
    threshold = sample_trunc_normal(rng, params.threshold_mean, params.threshold_std, n)
    fr_cap = sample_trunc_normal(rng, params.frcap_mean, params.frcap_std, n)
    return {"tau": tau, "gain": gain, "threshold": threshold, "fr_cap": fr_cap}


def integrate_rate(
    weighted_W: np.ndarray,
    *,
    tau: np.ndarray,
    gain: np.ndarray,
    threshold: np.ndarray,
    fr_cap: np.ndarray,
    inputs: np.ndarray,
    params: PuglieseRateParams,
) -> np.ndarray:
    """RK45 (Dopri5 family) with the authors' rtol/atol. Not bit-identical to JAX/diffrax."""
    n = weighted_W.shape[0]
    t_axis = np.arange(0.0, params.T + params.dt / 2.0, params.dt, dtype=np.float64)
    R0 = np.zeros(n, dtype=np.float64)

    def rhs(t, R):
        return rate_equation_half_tanh(
            t,
            R,
            inputs=inputs,
            pulse_start=params.pulse_start,
            pulse_end=params.pulse_end,
            tau=tau,
            weighted_W=weighted_W,
            threshold=threshold,
            gain=gain,
            fr_cap=fr_cap,
        )

    sol = solve_ivp(
        rhs,
        (0.0, params.T),
        R0,
        method="RK45",
        t_eval=t_axis,
        rtol=params.rtol,
        atol=params.atol,
        vectorized=False,
    )
    if not sol.success:
        raise RuntimeError(f"Rate ODE failed: {sol.message}")
    rates = np.clip(np.where(np.isfinite(sol.y), sol.y, 0.0), 0.0, RATE_MAX_HZ)
    return rates  # (n_neurons, n_times)


def shuffle_weight_columns(W: np.ndarray, idxs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One shared permutation of selected postsynaptic columns. Authors' shuffle_W."""
    idxs = np.asarray(idxs, dtype=np.int64)
    if idxs.size < 2:
        return W
    out = np.array(W, copy=True)
    perm = rng.permutation(idxs.size)
    out[:, idxs] = out[:, idxs][:, perm]
    return out


def class_conditional_shuffle(
    W: np.ndarray,
    *,
    is_dn: np.ndarray,
    is_mn: np.ndarray,
    is_exc: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sign- and class-conditional topology null, following authors' full_shuffle groups."""
    is_in = ~(is_dn | is_mn)
    groups = (
        np.flatnonzero(is_dn & is_exc),
        np.flatnonzero(is_dn & ~is_exc),
        np.flatnonzero(is_in & is_exc),
        np.flatnonzero(is_in & ~is_exc),
        np.flatnonzero(is_mn),
    )
    out = np.array(W, copy=True)
    for idxs in groups:
        out = shuffle_weight_columns(out, idxs, rng)
    return out


def population_rate_metrics(rates: np.ndarray, dt_s: float, recruited: np.ndarray | None = None) -> dict:
    """Census plus FFT/autocorr on the mean recruited trace. No walking-band clamp."""
    from organism.cpg_rhythm import rhythmicity_score

    rates = np.asarray(rates, dtype=np.float64)
    if rates.ndim != 2:
        raise ValueError("rates must be (n_neurons, n_times)")
    n, t = rates.shape
    mean_over_time = rates.mean(axis=1)
    if recruited is None:
        recruited = np.any(rates > 1e-8, axis=1)
    legacy_active = mean_over_time > 1.0
    if not np.any(legacy_active):
        legacy_active = mean_over_time > 0.1
    mean_trace = rates[recruited].mean(axis=0) if np.any(recruited) else np.zeros(t)
    scored = rhythmicity_score(mean_trace, dt_s * 1000.0)
    return {
        "n": int(n),
        "mean_activity": float(rates.mean()) if rates.size else 0.0,
        "active_neuron_count": int(np.count_nonzero(recruited)),
        "scoring_active_neuron_count": int(np.count_nonzero(legacy_active)),
        "mean_firing_rate_hz": float(mean_over_time.mean()) if n else 0.0,
        "max_firing_rate_hz": float(rates.max()) if rates.size else 0.0,
        "recruited": bool(np.any(recruited)),
        "rhythmicity": scored,
        "dominant_frequency": scored.get("dominant_frequency"),
        "rhythmicity_score": scored.get("score"),
        "fft_executed": bool(scored.get("fft_executed")),
    }
