"""Strict SBC on every block (ROADMAP T0.1, 2026-10-03).

The a_lm/C_l rows of the coverage ensemble could not carry a strict SBC rank:
Block 1's default prior on C_l^TT is flat and improper, so the truth was held at
the fiducial and the a_lm rank was read against a SIMULATED null whose noise
level is calibrated from the chains themselves. These tests cover the opt-in
pieces that remove that circularity, following the Block 4 pattern
(lensing.sample_cl_phiphi_given_phi):

  * model.sample_cl_given_alm(prior_nu, cl_fid) -- proper conjugate
    InvGamma(nu/2, nu*C_l^fid/2) prior on C_l^TT; None = bit-identical old draw
  * run_gibbs_chain(cl_prior_nu, cl_prior_fid) -- passes it to Block 1
  * run_gibbs_chain(phi_fixed=True) -- Blocks 1 + 2 against the LENSED
    likelihood with phi held at phi_initial (the T0.1 a discriminator: is the
    a_lm/C_l block exact once phi is known?)
"""

import numpy as np
import pytest

from diffcmb.alm_utils import packed_dof_per_multipole, packed_length

try:
    import tensorflow  # noqa: F401
    import tensorflow_probability  # noqa: F401
    HAS_TFP = True
except ImportError:
    HAS_TFP = False

try:
    import healpy  # noqa: F401
    HAS_HEALPY = True
except ImportError:
    HAS_HEALPY = False

skip_no_tfp = pytest.mark.skipif(not (HAS_TFP and HAS_HEALPY), reason="TF/TFP/healpy required")

LMAX, NSIDE = 8, 8


@pytest.fixture(scope="module")
def small_model():
    from diffcmb import CosmologyAdvancedSampling
    m = CosmologyAdvancedSampling(_lmax=LMAX, _NSIDE=NSIDE, _noisesig=1.0,
                                   data_mode='synthetic')
    m._ensure_tf_tensors()
    return m


def _alm(seed=0):
    return np.random.default_rng(seed).normal(size=packed_length(LMAX))


# --- Block 1 proper prior ---------------------------------------------------

@skip_no_tfp
def test_cl_prior_nu_none_is_bit_identical_to_flat_prior(small_model):
    """None = the old draw, consuming the RNG stream identically."""
    a = small_model.sample_cl_given_alm(_alm(), np.random.default_rng(7))
    b = small_model.sample_cl_given_alm(_alm(), np.random.default_rng(7),
                                        prior_nu=None, cl_fid=None)
    np.testing.assert_array_equal(a, b)


@skip_no_tfp
def test_cl_proper_prior_matches_conjugate_invgamma(small_model):
    """C_l | a ~ InvGamma(k_l/2 + nu/2, (S_l + nu C_l^fid)/2), k_l = 2l+1."""
    from scipy import stats

    alm = _alm(3)
    nu = 30.0
    cl_fid = np.linspace(0.5, 2.0, LMAX)
    rng = np.random.default_rng(11)
    draws = np.exp(np.array([
        small_model.sample_cl_given_alm(alm, rng, prior_nu=nu, cl_fid=cl_fid)
        for _ in range(4000)]))
    S = small_model.compute_sl_np(alm)
    k = packed_dof_per_multipole(LMAX)
    for i in range(LMAX - 2):
        L = i + 2
        alpha, beta = k[L] / 2.0 + nu / 2.0, (S[L] + nu * cl_fid[L]) / 2.0
        p = stats.kstest(draws[:, i], "invgamma", args=(alpha, 0.0, beta)).pvalue
        assert p > 1e-3, f"l={L}: not the conjugate posterior (KS p={p:.2g})"


@skip_no_tfp
def test_cl_proper_prior_cannot_collapse(small_model):
    """The r036 failure: with S_l ~ 0 the flat-prior draw is ~0, the proper
    prior's draw stays at the scale nu*C_fid/(k+nu) -- it cannot collapse."""
    alm = np.zeros(packed_length(LMAX))
    cl_fid = np.ones(LMAX)
    rng = np.random.default_rng(5)
    flat = np.exp(small_model.sample_cl_given_alm(alm, rng))
    proper = np.exp(small_model.sample_cl_given_alm(alm, rng, prior_nu=30.0, cl_fid=cl_fid))
    assert np.all(flat < 1e-5)
    assert np.all(proper > 0.1)


@skip_no_tfp
@pytest.mark.parametrize("nu", [0.0, -1.0, np.nan])
def test_cl_prior_nu_must_be_positive(small_model, nu):
    with pytest.raises(ValueError, match="prior_nu"):
        small_model.sample_cl_given_alm(_alm(), np.random.default_rng(0),
                                        prior_nu=nu, cl_fid=np.ones(LMAX))


@skip_no_tfp
def test_cl_prior_requires_a_fiducial(small_model):
    with pytest.raises(ValueError, match="cl_fid"):
        small_model.sample_cl_given_alm(_alm(), np.random.default_rng(0), prior_nu=30.0)


# --- run_gibbs_chain wiring ---------------------------------------------------

@skip_no_tfp
def test_gibbs_passes_the_cl_prior_to_block1(small_model):
    """A prior with nu -> huge pins every C_l draw at the fiducial."""
    from diffcmb import run_gibbs_chain

    cl_fid = np.full(LMAX, 0.37)
    samples = run_gibbs_chain(small_model, n_samples=5, n_burnin=2, hmc_step_size=0.01,
                              n_lfs=3, seed=1, cl_prior_nu=1e7, cl_prior_fid=cl_fid)[0]
    np.testing.assert_allclose(np.exp(samples[:, :LMAX - 2]), 0.37, rtol=2e-3)


@skip_no_tfp
def test_gibbs_cl_prior_snapshots_the_fiducial(small_model):
    """Same trap as Block 4: the prior must not chase a caller-mutated array."""
    from diffcmb import run_gibbs_chain

    cl_fid = np.full(LMAX, 0.37)
    out = run_gibbs_chain(small_model, n_samples=3, n_burnin=1, hmc_step_size=0.01,
                          n_lfs=3, seed=1, cl_prior_nu=1e7, cl_prior_fid=cl_fid)[0]
    cl_fid[:] = 99.0
    np.testing.assert_allclose(np.exp(out[:, :LMAX - 2]), 0.37, rtol=2e-3)


@skip_no_tfp
def test_gibbs_rejects_a_cl_prior_without_fiducial(small_model):
    from diffcmb import run_gibbs_chain

    with pytest.raises(ValueError, match="cl_prior_fid"):
        run_gibbs_chain(small_model, n_samples=2, n_burnin=1, cl_prior_nu=30.0)


# --- phi held fixed -------------------------------------------------------------

@skip_no_tfp
def test_phi_fixed_holds_phi_and_still_samples_alm(small_model):
    from diffcmb import run_gibbs_chain

    n_phi = packed_length(LMAX)
    phi0 = np.random.default_rng(2).normal(scale=1e-3, size=n_phi)
    samples, phi_samples, logp, accepts, _ = run_gibbs_chain(
        small_model, n_samples=5, n_burnin=3, hmc_step_size=0.01, n_lfs=3,
        cl_phiphi_full=np.full(LMAX, 1e-6), phi_initial=phi0, phi_fixed=True, seed=3)
    assert phi_samples.shape == (5, n_phi)
    # whiten/unwhiten round trip only: phi itself never moves
    np.testing.assert_allclose(phi_samples, np.broadcast_to(phi0, phi_samples.shape),
                               rtol=1e-12, atol=0)
    assert not np.allclose(samples[0], samples[-1])


@skip_no_tfp
def test_phi_fixed_uses_the_lensed_likelihood(small_model, monkeypatch):
    """Not the blind fit in disguise: the alm block must see phi.

    Two chains from the same seed and start, one with phi = 0 and one with a
    large fixed phi, must give different alm draws."""
    from diffcmb import run_gibbs_chain

    n_phi = packed_length(LMAX)
    kw = {"n_samples": 4, "n_burnin": 2, "hmc_step_size": 0.01, "n_lfs": 3,
          "cl_phiphi_full": np.full(LMAX, 1e-2), "phi_fixed": True, "seed": 4,
          "initial_params": np.array(small_model.x0, dtype=np.float64)}
    a = run_gibbs_chain(small_model, phi_initial=np.zeros(n_phi), **kw)[0]
    b = run_gibbs_chain(small_model, phi_initial=np.full(n_phi, 0.05), **kw)[0]
    assert not np.allclose(a, b)


@skip_no_tfp
def test_phi_fixed_requires_phi_initial_and_rejects_block4(small_model):
    from diffcmb import run_gibbs_chain

    with pytest.raises(ValueError, match="phi_initial"):
        run_gibbs_chain(small_model, n_samples=2, n_burnin=1,
                        cl_phiphi_full=np.full(LMAX, 1e-6), phi_fixed=True)
    with pytest.raises(ValueError, match="phi_fixed"):
        run_gibbs_chain(small_model, n_samples=2, n_burnin=1,
                        cl_phiphi_full=np.full(LMAX, 1e-6),
                        phi_initial=np.zeros(packed_length(LMAX)), phi_fixed=True,
                        sample_cl_phiphi=True)
    with pytest.raises(ValueError, match="phi_fixed"):
        run_gibbs_chain(small_model, n_samples=2, n_burnin=1, phi_fixed=True)
