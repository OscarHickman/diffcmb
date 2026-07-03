"""Correctness check for the messenger-field Gibbs sampler (messenger.py).

Dense brute-force reference: a Bayesian linear-Gaussian toy problem
d = A s + n with A^T A = I (mimicking a full-sky orthonormal SHT synthesis)
and a subset of pixels "masked" (huge noise variance). The true posterior
p(s|d) is Gaussian with precision Lambda = S^-1 + A^T N^-1 A, which is
NOT diagonal despite A^T A = I, because the mask varies N pixel-to-pixel.
This is exactly the mechanism (ROADMAP.md Phase 0c) that makes plain
diagonal-preconditioned CG fail on a masked sky: the messenger sampler
must reproduce this full, non-diagonal covariance without ever forming
or inverting Lambda directly.
"""
import numpy as np
import pytest

from diffcmb.messenger import run_messenger_gibbs


def _build_toy_problem(rng, n_alm=12, n_pix=36, frac_masked=0.3, mask_ninv_floor=1e-10):
    # A with orthonormal columns: A^T A = I_{n_alm}, same property a
    # full-sky orthonormal SHT synthesis has.
    A = np.linalg.qr(rng.standard_normal((n_pix, n_alm)))[0][:, :n_alm]

    cl = rng.uniform(0.5, 3.0, size=n_alm)
    inv_cl_diag = 1.0 / cl

    noise_var = rng.uniform(0.2, 1.0, size=n_pix)
    n_masked = int(frac_masked * n_pix)
    masked_idx = rng.choice(n_pix, size=n_masked, replace=False)
    Ninv = 1.0 / noise_var
    Ninv[masked_idx] = mask_ninv_floor  # ~zero precision: "masked"

    s_true = rng.standard_normal(n_alm) * np.sqrt(cl)
    d = A @ s_true + rng.standard_normal(n_pix) * np.sqrt(noise_var)
    d[masked_idx] = 0.0  # irrelevant: Ninv~0 there

    return A, inv_cl_diag, Ninv, d


def _dense_posterior(A, inv_cl_diag, Ninv, d):
    Lambda = np.diag(inv_cl_diag) + A.T @ (Ninv[:, None] * A)
    Sigma = np.linalg.inv(Lambda)
    mu = Sigma @ (A.T @ (Ninv * d))
    return mu, Sigma


def test_messenger_gibbs_matches_dense_masked_posterior():
    rng = np.random.default_rng(42)
    A, inv_cl_diag, Ninv, d = _build_toy_problem(rng)
    mu_true, Sigma_true = _dense_posterior(A, inv_cl_diag, Ninv, d)

    # Off-diagonal entries must be non-trivial, otherwise this test would
    # not actually exercise the mask-induced coupling the messenger method
    # exists to handle.
    off_diag_scale = np.abs(Sigma_true - np.diag(np.diag(Sigma_true))).max()
    assert off_diag_scale > 1e-3 * np.diag(Sigma_true).mean()

    tau2 = 0.9 * (1.0 / Ninv[Ninv > 1e-6]).min()  # tau2 <= min observed N_ii

    n_burnin = 500
    n_samples = 6000
    thin = 2

    rng_sampler = np.random.default_rng(7)
    s = None
    for _ in range(n_burnin):
        s = run_messenger_gibbs(
            d, Ninv, inv_cl_diag, tau2,
            A_action=lambda x: A @ x, At_action=lambda t: A.T @ t,
            rng=rng_sampler, n_iter=1, s0=s,
        )

    samples = np.empty((n_samples, len(inv_cl_diag)))
    for i in range(n_samples):
        s = run_messenger_gibbs(
            d, Ninv, inv_cl_diag, tau2,
            A_action=lambda x: A @ x, At_action=lambda t: A.T @ t,
            rng=rng_sampler, n_iter=thin, s0=s,
        )
        samples[i] = s

    mu_emp = samples.mean(axis=0)
    Sigma_emp = np.cov(samples, rowvar=False)

    # Generous tolerance: ~5 Monte Carlo standard errors on the mean,
    # accounting for residual autocorrelation across thinned sweeps.
    se_mu = np.sqrt(np.diag(Sigma_true) / n_samples)
    np.testing.assert_allclose(mu_emp, mu_true, atol=8 * se_mu.max())

    rel_cov_err = np.abs(Sigma_emp - Sigma_true) / np.abs(Sigma_true).max()
    assert rel_cov_err.max() < 0.15, (
        f"messenger-field empirical covariance deviates from the dense "
        f"masked-sky posterior by {rel_cov_err.max():.3f} (relative to max "
        f"entry) — the sampler should reproduce the mask-induced "
        f"off-diagonal coupling, not just per-pixel variances"
    )


def test_sample_t_given_s_matches_pointwise_normal_moments():
    from diffcmb.messenger import sample_t_given_s

    rng = np.random.default_rng(0)
    n = 5000
    s_pix = rng.standard_normal(n)
    d = s_pix + rng.standard_normal(n) * 0.3
    Ninv = np.full(n, 1.0 / 0.3**2)
    tau2 = 0.05

    t = sample_t_given_s(s_pix, d, Ninv, tau2, rng)

    Ninv_red = Ninv / (1.0 - tau2 * Ninv)
    precision = Ninv_red + 1.0 / tau2
    expected_mean = (Ninv_red * d + s_pix / tau2) / precision
    expected_var = 1.0 / precision

    resid = (t - expected_mean) / np.sqrt(expected_var)
    assert abs(resid.mean()) < 0.05
    assert abs(resid.std() - 1.0) < 0.05
