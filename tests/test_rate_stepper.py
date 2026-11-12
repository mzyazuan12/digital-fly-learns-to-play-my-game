"""RK4 stepper: hook identity, determinism, agreement with integrate_rate on a toy net."""

import numpy as np
from scipy.sparse import csr_matrix

from flybrain.pugliese_rate import PuglieseRateParams, integrate_rate, reweight_connectivity
from flybrain.rate_stepper import RateStepper


def _toy(n=40, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 30, (n, n)) * (rng.random((n, n)) < 0.1)
    p = PuglieseRateParams(stim_amplitude=400.0, T=0.4)
    w = csr_matrix(reweight_connectivity(W, p.exc_multiplier, p.inh_multiplier))
    kw = dict(
        tau=np.full(n, 0.02),
        gain=np.ones(n),
        threshold=np.full(n, 7.5),
        fr_cap=np.full(n, 200.0),
    )
    inp = np.where(np.arange(n) == 0, 400.0, 0.0)
    return w, p, kw, inp


def test_zero_hook_is_bit_identical_and_deterministic():
    w, p, kw, inp = _toy()
    a = RateStepper(W=w, dn_input=inp, **kw).run(T=0.4)
    b = RateStepper(W=w, dn_input=inp, **kw).run(T=0.4, sensory=lambda t, R: np.zeros_like(R))
    c = RateStepper(W=w, dn_input=inp, **kw).run(T=0.4)
    assert np.array_equal(a, b)
    assert np.array_equal(a, c)


def test_matches_integrate_rate():
    w, p, kw, inp = _toy()
    p_ref = PuglieseRateParams(stim_amplitude=400.0, T=0.4, rtol=1e-9, atol=1e-11)
    ref = integrate_rate(w, inputs=inp, params=p_ref, **kw)
    st = RateStepper(W=w, dn_input=inp, **kw).run(T=0.4)
    assert ref.shape == st.shape
    assert ref.max() > 1.0
    np.testing.assert_allclose(st, ref, atol=0.05, rtol=1e-3)


def test_sensory_input_changes_the_target_neuron():
    w, p, kw, inp = _toy()
    base = RateStepper(W=w, dn_input=inp, **kw).run(T=0.4)
    I = np.zeros(w.shape[0])
    I[5] = 50.0
    driven = RateStepper(W=w, dn_input=inp, **kw).run(T=0.4, sensory=lambda t, R: I)
    assert driven[5].max() > base[5].max() + 1.0
