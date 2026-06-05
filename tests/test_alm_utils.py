"""Tests for alm_utils — covering both pure-Python and TF paths."""
import numpy as np
import pytest

# ── helpers ──────────────────────────────────────────────────────────────────

def _has_tf():
    try:
        import tensorflow as tf  # noqa: F401
        return True
    except Exception:
        return False


def _has_healpy():
    try:
        import healpy as hp  # noqa: F401
        return True
    except Exception:
        return False


# ── splittosingularalm (numpy reference) ─────────────────────────────────────

def test_splittosingularalm_roundtrip():
    """Real and imaginary alm vectors survive a split→combine round-trip."""
    from src.cmb.alm_utils import singulartosplitalm, splittosingularalm

    rng = np.random.default_rng(0)
    lmax = 6
    len_alm = lmax * (lmax + 1) // 2
    alm = rng.standard_normal(len_alm) + 1j * rng.standard_normal(len_alm)
    # zero out monopole/dipole and m=1 imaginary parts (as the model does)
    alm[:3] = 0
    alm = np.array([complex(a.real if m == 0 or m == 1 else a.real,
                            0 if m == 0 or m == 1 else a.imag)
                    for l in range(lmax) for m, a in enumerate(
                        [alm[l*(l+1)//2+mm] for mm in range(l+1)])])

    real_parts, imag_parts = singulartosplitalm(alm)
    real_alm = real_parts[3:]  # skip L=0,1
    imag_alm = [alm[l*(l+1)//2+m].imag
                for l in range(lmax) for m in range(l+1)
                if m >= 2 and l >= 2]

    rebuilt = splittosingularalm(real_alm, imag_alm, lmax)
    assert len(rebuilt) == len_alm
    for i in range(len_alm):
        assert abs(rebuilt[i].real - alm[i].real) < 1e-12
        assert abs(rebuilt[i].imag - alm[i].imag) < 1e-12


def test_splittosingularalm_monopole_dipole_zero():
    """L=0 and L=1 entries are always zeroed out."""
    from src.cmb.alm_utils import splittosingularalm

    rng = np.random.default_rng(1)
    lmax = 5
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = sum(l - 1 for l in range(2, lmax))
    realalm = rng.standard_normal(n_real).tolist()
    imagalm = rng.standard_normal(n_imag).tolist()
    alm = splittosingularalm(realalm, imagalm, lmax)
    # L=0 (index 0), L=1 m=0 (index 1), L=1 m=1 (index 2) must be zero
    assert alm[0] == complex(0, 0)
    assert alm[1] == complex(0, 0)
    assert alm[2] == complex(0, 0)


def test_splittosingularalm_m01_imaginary_zero():
    """For L≥2, m=0 and m=1 entries must have zero imaginary part."""
    from src.cmb.alm_utils import splittosingularalm

    rng = np.random.default_rng(2)
    lmax = 6
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = sum(l - 1 for l in range(2, lmax))
    realalm = rng.standard_normal(n_real).tolist()
    imagalm = rng.standard_normal(n_imag).tolist()
    alm = splittosingularalm(realalm, imagalm, lmax)
    for L in range(2, lmax):
        for m in [0, 1]:
            idx = L * (L + 1) // 2 + m
            assert alm[idx].imag == 0.0, f"L={L} m={m} should have zero imaginary"


@pytest.mark.skipif(not _has_tf(), reason="TensorFlow not available")
def test_splittosingularalm_tf_matches_numpy():
    """TF scatter_nd implementation must match the numpy reference exactly."""
    import tensorflow as tf

    from src.cmb.alm_utils import splittosingularalm, splittosingularalm_tf

    rng = np.random.default_rng(3)
    lmax = 8
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = sum(l - 1 for l in range(2, lmax))
    realalm_np = rng.standard_normal(n_real).tolist()
    imagalm_np = rng.standard_normal(n_imag).tolist()

    ref = splittosingularalm(realalm_np, imagalm_np, lmax)

    r_tf = tf.constant(realalm_np, dtype=tf.float64)
    i_tf = tf.constant(imagalm_np, dtype=tf.float64)
    out = splittosingularalm_tf(r_tf, i_tf, lmax).numpy()

    for k, (r, o) in enumerate(zip(ref, out)):
        assert abs(r.real - o.real) < 1e-12, f"index {k}: real mismatch"
        assert abs(r.imag - o.imag) < 1e-12, f"index {k}: imag mismatch"


# ── ordering index round-trips ────────────────────────────────────────────────

@pytest.mark.skipif(not _has_healpy(), reason="healpy not available")
def test_ordering_indices_roundtrip():
    """almmotho(almhotmo(x)) == x and vice versa."""
    from src.cmb.alm_utils import almhotmo, almmotho

    rng = np.random.default_rng(4)
    lmax = 10
    n = lmax * (lmax + 1) // 2
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)

    assert np.allclose(almmotho(almhotmo(x, lmax), lmax), x)
    assert np.allclose(almhotmo(almmotho(x, lmax), lmax), x)


# ── almtomap_tf ───────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _has_tf() or not _has_healpy(), reason="TF or healpy not available")
def test_almtomap_tf_matches_healpy():
    """almtomap_tf must produce the same map as hp.alm2map for random alms."""
    import healpy as hp
    import scipy
    import tensorflow as tf

    from src.cmb.alm_utils import (
        almtomap_tf,
        hpalminit,
        splittosingularalm_tf,
    )

    lmax, nside = 6, 4
    npix = 12 * nside**2
    len_alm = lmax * (lmax + 1) // 2

    rng = np.random.default_rng(5)
    # Build healpy-ordered alm (zero out monopole/dipole and m=1 imag)
    hp_alm = rng.standard_normal(len_alm) + 1j * rng.standard_normal(len_alm)
    hp_alm = hpalminit(hp_alm.copy(), lmax)

    ref_map = hp.alm2map(hp_alm, nside, lmax=lmax - 1)

    # Build sph matrix (same as model._ensure_tf_tensors)
    thetas, phis = hp.pix2ang(nside=nside, ipix=np.arange(npix))
    sph_np = np.empty((npix, len_alm), dtype=np.complex128)
    col = 0
    for L in range(lmax):
        for m in range(L + 1):
            vals = scipy.special.sph_harm(m, L, phis, thetas)
            if L == 0:
                vals = vals.real.astype(np.complex128)
            sph_np[:, col] = vals
            col += 1
    sph = tf.convert_to_tensor(sph_np, dtype=np.complex128)

    # Weights (m=0 → 0.5, else 1.0)
    w = np.ones(len_alm, dtype=np.complex128)
    cnt = 0
    for L in range(lmax):
        for m in range(L + 1):
            if m == 0:
                w[cnt] = 0.5 + 0j
            cnt += 1
    weights = tf.convert_to_tensor(w, dtype=np.complex128)

    # Assemble alm in author ordering via the same path as psi_tf
    from src.cmb.alm_utils import almhotmo
    mo_alm = almhotmo(hp_alm, lmax)  # healpy → author ordering
    r_parts = [complex(mo_alm[l*(l+1)//2+m]).real
               for l in range(lmax) for m in range(l+1) if l >= 2]
    i_parts = [complex(mo_alm[l*(l+1)//2+m]).imag
               for l in range(lmax) for m in range(l+1) if l >= 2 and m >= 2]
    r_tf = tf.constant(r_parts, dtype=tf.float64)
    i_tf = tf.constant(i_parts, dtype=tf.float64)
    a_tf = splittosingularalm_tf(r_tf, i_tf, lmax)

    result_map = almtomap_tf(a_tf, nside, lmax, sph, _weights=weights).numpy()

    np.testing.assert_allclose(result_map, ref_map, atol=1e-10,
                               err_msg="almtomap_tf does not match hp.alm2map")
