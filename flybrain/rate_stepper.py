"""Fixed-step RK4 for the half-tanh rate model, for lockstep co-simulation.

Same right-hand side as flybrain.pugliese_rate.rate_equation_half_tanh, but
advanced one step at a time so input computed during the run (sensory
current from the body) can enter. Stage 2B design §2: h = 0.1 ms, one
neural step per MuJoCo step, sensory current held over each step, state
not clipped during integration (integrate_rate clips only its output).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RateStepper:
    W: object  # post x pre, CSR or dense (reweight_connectivity output)
    tau: np.ndarray
    gain: np.ndarray
    threshold: np.ndarray
    fr_cap: np.ndarray
    dn_input: np.ndarray  # tonic drive vector (e.g. 400 at DNg100)
    h: float = 1e-4
    pulse_start: float = 0.02
    pulse_end: float = 1.999

    def __post_init__(self):
        n = self.W.shape[0]
        self.R = np.zeros(n, dtype=np.float64)
        self.k = 0  # step count; time is k*h, never accumulated (no onset drift)
        self.t = 0.0
        self._k = self.gain / self.fr_cap

    def _f(self, I, R: np.ndarray, I_sens: np.ndarray | None) -> np.ndarray:
        total = self.W @ R + I
        if I_sens is not None:
            total = total + I_sens
        act = np.maximum(self.fr_cap * np.tanh(self._k * (total - self.threshold)), 0.0)
        return (act - R) / self.tau

    def step(self, I_sens: np.ndarray | None = None) -> np.ndarray:
        h, t, R = self.h, self.t, self.R
        # Inputs are held over the step (evaluated at its start), so the DN
        # pulse switches on exactly at a step boundary, like I_sens.
        I = self.dn_input if (self.pulse_start <= t + 1e-12 <= self.pulse_end) else 0.0
        k1 = self._f(I, R, I_sens)
        k2 = self._f(I, R + h / 2 * k1, I_sens)
        k3 = self._f(I, R + h / 2 * k2, I_sens)
        k4 = self._f(I, R + h * k3, I_sens)
        self.R = R + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.k += 1
        self.t = self.k * h
        return self.R

    def run(self, T: float = 2.0, record_every: int = 10, sensory=None) -> np.ndarray:
        """Integrate to T; record clipped rates every `record_every` steps (1 ms at h=0.1 ms).

        sensory: optional callable (t, R) -> I_sens, called once per step.
        """
        from flybrain.pugliese_rate import RATE_MAX_HZ

        n_steps = int(round(T / self.h))
        out = [np.clip(self.R.copy(), 0.0, RATE_MAX_HZ)]
        for k in range(1, n_steps + 1):
            I_sens = sensory(self.t, self.R) if sensory is not None else None
            self.step(I_sens)
            if k % record_every == 0:
                out.append(np.clip(self.R.copy(), 0.0, RATE_MAX_HZ))
        return np.stack(out, axis=1)
