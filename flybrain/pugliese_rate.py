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
by the same factor (src/utils/sim_utils.py::set_sizes). For MaleCNS the
denominator is the full neuPrint Neuron.size median, not the 408-cell
circuit median. Amplitude 250 is a rate-ODE unit, not millivolts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from flybrain.loader import Connectome, DEFAULT_DATA
from flybrain.malecns_volume import (
    VOLUME_SOURCE,
    assert_volume_size_source,
    sizes_for_body_ids,
)
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


def sample_trunc_normal(rng: np.random.Generator, mean: float, stdev: float, shape, lower_bound: float = 0.0) -> np.ndarray:
    """Truncated-normal via inverse CDF. Matches Pugliese sim_utils math, not JAX Threefry draws."""
    shape = tuple(np.atleast_1d(shape).tolist()) if np.ndim(shape) else (int(shape),)
    if stdev < 1e-10 or not np.isfinite(stdev) or not np.isfinite(mean):
        return np.full(shape, max(mean, 0.0) if np.isfinite(mean) else 0.0, dtype=np.float64)
    upper = min(mean + min(100.0 * stdev, 1e6), 1e10)
    a = np.clip((lower_bound - mean) / stdev, -10.0, 10.0)
    b = np.clip((upper - mean) / stdev, -10.0, 10.0)
    from scipy.stats import norm

    cdf_a = float(np.clip(norm.cdf(a), 1e-10, 1.0 - 1e-10))
    cdf_b = float(np.clip(norm.cdf(b), 1e-10, 1.0 - 1e-10))
    cdf_b = max(cdf_b, cdf_a + 1e-10)
    u = rng.uniform(cdf_a, cdf_b, size=shape)
    z = norm.ppf(np.clip(u, 1e-10, 1.0 - 1e-10))
    return np.clip(mean + stdev * z, -1e10, 1e10).astype(np.float64)


def sample_trunc_normal_jax(key, mean: float, stdev: float, shape, lower_bound: float = 0.0):
    """Authors' JAX inverse-CDF truncated normal. Requires jax."""
    import jax
    import jax.numpy as jnp

    shape = tuple(shape)
    invalid = (~jnp.isfinite(mean)) | (~jnp.isfinite(stdev)) | (~jnp.isfinite(lower_bound)) | (stdev < 0)

    def handle_invalid():
        return jnp.zeros(shape)

    def handle_zero():
        return jnp.maximum(mean, 0.0) * jnp.ones(shape)

    def handle_normal():
        upper_bound = jnp.minimum(mean + jnp.minimum(100 * stdev, 1e6), 1e10)
        a = jnp.clip((lower_bound - mean) / stdev, -10.0, 10.0)
        b = jnp.clip((upper_bound - mean) / stdev, -10.0, 10.0)
        cdf_a = jnp.clip(jax.scipy.stats.norm.cdf(a), 1e-10, 1.0 - 1e-10)
        cdf_b = jnp.clip(jax.scipy.stats.norm.cdf(b), 1e-10, 1.0 - 1e-10)
        cdf_b = jnp.maximum(cdf_b, cdf_a + 1e-10)
        u = jax.random.uniform(key, shape=shape, minval=cdf_a, maxval=cdf_b)
        z = jax.scipy.stats.norm.ppf(jnp.clip(u, 1e-10, 1.0 - 1e-10))
        return jnp.clip(mean + stdev * z, -1e10, 1e10)

    return jax.lax.cond(
        invalid,
        handle_invalid,
        lambda: jax.lax.cond(stdev < 1e-10, handle_zero, handle_normal),
    )


def set_sizes(sizes, gain, threshold, *, median_size: float | None = None):
    """Divide gain and multiply threshold by median-normalized size.

    Transcription of Pugliese_2026 src/utils/sim_utils.py::set_sizes.
    ``gain`` / ``threshold`` may be (n,) or (n_rep, n).

    Pass ``median_size`` to normalize by the modeled-dataset median rather
    than nanmedian of the vector in hand. Missing/zero sizes are replaced
    with that same median, then a /= s, theta *= s.
    """
    sizes = np.asarray(sizes, dtype=np.float64).copy()
    if sizes.ndim != 1:
        raise ValueError("sizes must be a 1-D per-neuron vector")
    if median_size is None:
        norm = float(np.nanmedian(sizes))
    else:
        norm = float(median_size)
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
    np.add.at(
        W,
        (pre, connectome.post.astype(np.int64)),
        connectome.anatomical.astype(np.float64) * connectome.sign.astype(np.float64),
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
    """SWC frustum volume. Not a Pugliese size source. Do not pass to set_sizes."""
    raise ValueError(
        "Pugliese MANC/mCNS size is neuPrint neuron volume. "
        "Do not derive size from SWC skeletons, synapse count, cable length, "
        "soma size, or partner count."
    )


def _swc_frustum_volume_for_tests(body_ids, swc_dir: Path = SWC_DIR) -> dict:
    """Kept only so tests can show SWC is the wrong size source."""
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


def rate_rhythmicity_score(trace: np.ndarray, dt_s: float) -> dict:
    """Unrestricted FFT/autocorr on a firing-rate trace.

    Does not use millivolt guardrails. Those apply only to shiu_lif_sanity_v1.
    Published 7–15 Hz is an annotation, not a peak-selection prior.
    """
    x = np.asarray(trace, dtype=np.float64)
    out = {
        "n": int(x.size),
        "mean": float(x.mean()) if x.size else 0.0,
        "std": float(x.std()) if x.size else 0.0,
        "min": float(x.min()) if x.size else 0.0,
        "max": float(x.max()) if x.size else 0.0,
        "score": None,
        "rhythmicity_score": None,
        "dominant_frequency": None,
        "fft_hz": None,
        "autocorrelation_peak": None,
        "fft_executed": False,
        "in_published_walk_band": False,
        "model_id": PUGLIESE_RATE_MALECNS_V1,
        "voltage_gate_applied": False,
    }
    if x.size < 16 or not np.all(np.isfinite(x)) or dt_s <= 0:
        return out
    mean = out["mean"]
    std = out["std"]
    if std < 1e-12:
        out["score"] = 0.0
        out["rhythmicity_score"] = 0.0
        out["autocorrelation_peak"] = 0.0
        return out
    out["fft_executed"] = True
    freqs = np.fft.rfftfreq(x.size, d=dt_s)
    spec = np.abs(np.fft.rfft(x - mean)) ** 2
    if spec.size > 1:
        peak_i = 1 + int(np.argmax(spec[1:]))
        out["fft_hz"] = float(freqs[peak_i])
        out["dominant_frequency"] = out["fft_hz"]
    xc = x - mean
    ac = np.correlate(xc, xc, mode="full")[x.size - 1 :]
    if ac[0] > 0:
        ac = ac / ac[0]
    lag_max = x.size // 3
    interior = ac[1:lag_max]
    if interior.size >= 3:
        peaks = np.flatnonzero((interior[1:-1] > interior[:-2]) & (interior[1:-1] >= interior[2:])) + 2
        if peaks.size:
            lag = int(peaks[np.argmax(ac[peaks])])
            score = float(max(0.0, ac[lag]))
            hz = 1.0 / (lag * dt_s)
            out["score"] = score
            out["rhythmicity_score"] = score
            out["autocorrelation_peak"] = score
            out["autocorr_hz"] = hz
            out["in_published_walk_band"] = bool(7.0 <= hz <= 15.0)
    return out


def population_rate_metrics(rates: np.ndarray, dt_s: float, recruited: np.ndarray | None = None) -> dict:
    """Census plus FFT/autocorr on the mean recruited trace. No walking-band clamp."""
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
    scored = rate_rhythmicity_score(mean_trace, dt_s)
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
