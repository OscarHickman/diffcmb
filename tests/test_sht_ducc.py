"""Validation for the matrix-free ducc0 SHT (Phase 1.5).

Checks, at small lmax so this runs cheaply on a CPU-only runner:
  A) `HealpixSHT.synthesis_full` agrees with `healpy.alm2map`.
  B) `masked_synthesis_tf`'s custom gradient agrees with a finite-difference
     gradient of a random real-valued loss (Wirtinger derivative, complex
     alm input -> real map output).
  C) The masked adjoint identity <Synthesis(a), m> == <a, w * conj(AdjointSynthesis(m))>
     used to derive the gradient in sht_ducc.py holds to float64 precision.

These three checks are what Phase 1.5 in ROADMAP.md calls "validate the
wrapped op against the dense path... at lmax=50" — done here against
healpy (the reference implementation already used everywhere else in this
codebase) rather than the dense `sph` matrix directly, since healpy's
`alm2map`/`map2alm` are themselves validated against that dense path
historically and are much cheaper to construct at test time.
"""
import numpy as np
import pytest

try:
    import healpy as hp
    HAS_HP = True
except ImportError:
    HAS_HP = False

try:
    import ducc0
    HAS_DUCC0 = True
except ImportError:
    HAS_DUCC0 = False

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

pytestmark = pytest.mark.skipif(
    not (HAS_HP and HAS_DUCC0 and HAS_TF),
    reason="healpy, ducc0 and tensorflow are all required for sht_ducc tests",
)

if HAS_HP and HAS_DUCC0 and HAS_TF:
    from diffcmb.sht_ducc import HealpixSHT, masked_synthesis_tf

NSIDE = 16
LMAX = 15


def _make_sht(seed=0, mask_frac=0.3):
    rng = np.random.default_rng(seed)
    npix = hp.nside2npix(NSIDE)
    unmasked = np.where(rng.random(npix) > mask_frac)[0]
    sht = HealpixSHT(nside=NSIDE, lmax=LMAX, unmasked_idx=unmasked, nthreads=2)
    return sht, rng


def _random_real_alm(sht, rng):
    n_alm = sht.n_alm
    alm = rng.standard_normal(n_alm) + 1j * rng.standard_normal(n_alm)
    _, ms = hp.Alm.getlm(LMAX - 1, i=np.arange(n_alm))
    alm[ms == 0] = alm[ms == 0].real
    return alm.astype(np.complex128), ms


def test_synthesis_matches_healpy():
    sht, rng = _make_sht()
    alm, _ = _random_real_alm(sht, rng)
    full_hp = hp.alm2map(alm, NSIDE, lmax=LMAX - 1, mmax=LMAX - 1)
    full_ducc = sht.synthesis_full(alm)
    assert np.max(np.abs(full_ducc - full_hp)) < 1e-10


def test_adjoint_identity():
    sht, rng = _make_sht(seed=1)
    alm, ms = _random_real_alm(sht, rng)
    m = rng.standard_normal(len(sht.unmasked_idx))

    lhs = float(np.sum(sht.masked_synthesis(alm) * m))
    g = sht.masked_adjoint_synthesis(m)
    w = np.where(ms == 0, 1.0, 2.0)
    rhs = float(np.sum(w * (np.conj(alm) * g).real))

    assert abs(lhs - rhs) / abs(lhs) < 1e-10


def test_gradient_matches_finite_differences():
    sht, rng = _make_sht(seed=2)
    alm, ms = _random_real_alm(sht, rng)
    weights = rng.standard_normal(len(sht.unmasked_idx))

    # Build alm from two independently-watched REAL leaves via tf.complex,
    # matching how this op is actually used in model.py/samplers.py
    # (splittosingularalm_tf constructs alm the same way). tf.complex(re, im)'s
    # registered gradient extracts Re/Im of the upstream cotangent directly
    # (no conjugation) — a different, but equally valid, convention from
    # watching a raw complex tensor as a Wirtinger leaf (which TF's
    # GradientTape *does* conjugate). Testing through tf.complex is the
    # convention this op's backward pass (sht_ducc.py) is actually derived
    # for and is what production code exercises.
    re_tf = tf.constant(alm.real, dtype=tf.float64)
    im_tf = tf.constant(alm.imag, dtype=tf.float64)
    w_tf = tf.constant(weights, dtype=tf.float64)

    with tf.GradientTape() as tape:
        tape.watch([re_tf, im_tf])
        alm_tf = tf.complex(re_tf, im_tf)
        out = masked_synthesis_tf(alm_tf, sht)
        loss = tf.reduce_sum(out * w_tf)
    grad_re, grad_im = tape.gradient(loss, [re_tf, im_tf])
    grad_re, grad_im = grad_re.numpy(), grad_im.numpy()

    eps = 1e-6
    n_check = min(20, sht.n_alm)
    rng_idx = rng.choice(sht.n_alm, size=n_check, replace=False)
    for i in rng_idx:
        d = np.zeros(sht.n_alm, dtype=np.complex128)
        d[i] = eps
        lp = float(np.sum(sht.masked_synthesis(alm + d) * weights))
        lm = float(np.sum(sht.masked_synthesis(alm - d) * weights))
        deriv_re_fd = (lp - lm) / (2 * eps)
        assert abs(grad_re[i] - deriv_re_fd) < 1e-4

        if ms[i] == 0:
            continue
        d = np.zeros(sht.n_alm, dtype=np.complex128)
        d[i] = 1j * eps
        lp = float(np.sum(sht.masked_synthesis(alm + d) * weights))
        lm = float(np.sum(sht.masked_synthesis(alm - d) * weights))
        deriv_im_fd = (lp - lm) / (2 * eps)
        assert abs(grad_im[i] - deriv_im_fd) < 1e-4
