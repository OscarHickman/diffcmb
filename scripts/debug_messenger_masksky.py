"""Validate sample_alm_messenger against a dense masked-sky reference,
wired through the real ducc0 model path (ROADMAP.md Phase 0c, Step 3).

Small lmax/NSIDE synthetic problem so it runs on CPU in well under a minute.
Builds the dense full-sky operator J (packed real+imag alm -> map) by
probing sample_alm_messenger's own A_action with unit vectors, then compares
the messenger sampler's empirical posterior mean/covariance against the
dense masked-sky reference Lambda = diag(1/C_l) + J^T diag(Ninv) J.

This also checks the key approximation sample_alm_messenger relies on:
A^T A = NPIX/(4*pi) * I (only exactly true in the continuum limit for a
real HEALPix SHT) — reported as a diagnostic, not (yet) gated on.

Usage: PYTHONPATH=diffcmb .venv/bin/python scripts/debug_messenger_masksky.py
"""
import numpy as np

from diffcmb.model import CosmologyAdvancedSampling
from diffcmb.samplers import (
    _build_full_sky_norm_diag,
    _build_inv_cl_diag,
    _packed_to_alm_ho,
    sample_alm_messenger,
)
from diffcmb.sht_ducc import HealpixSHT

LMAX = 10
NSIDE = 8
NOISE = 1.0
MASK_FSKY = 0.7
N_MESSENGER_ITER = 60
N_BURNIN = 200
N_SAMPLES = 1000
THIN = 2


def build_model():
    model = CosmologyAdvancedSampling(
        LMAX, NSIDE, NOISE, data_mode='synthetic', dtype=None,
        use_matrixfree_sht=True,
    )
    import healpy as hp

    theta, _ = hp.pix2ang(NSIDE, np.arange(model.NPIX))
    cutoff = np.arccos(1 - 2 * MASK_FSKY)
    model.unmasked_idx = np.where(theta < cutoff)[0]
    print(f"  f_sky = {len(model.unmasked_idx) / model.NPIX:.3f}")
    # Actually mask Ninv/prior_map outside unmasked_idx (matching model.py's
    # data_mode='real' branch) -- overriding unmasked_idx alone only crops
    # which pixels the *masked* SHT path sees; sample_alm_messenger uses the
    # raw model.Ninv/model.prior_map for its full-sky operator, so without
    # this the "masked" test was silently a full-sky problem in disguise.
    mask = np.ones(model.NPIX, dtype=bool)
    mask[model.unmasked_idx] = False
    model.Ninv = model.Ninv.copy()
    model.Ninv[mask] = 0.0
    model.prior_map = model.prior_map.copy()
    model.prior_map[mask] = 0.0
    model._ensure_tf_tensors()
    return model


def build_dense_J(model, n_real, n_imag):
    n_alm = n_real + n_imag
    sht_full = HealpixSHT(nside=model.NSIDE, lmax=model.lmax, unmasked_idx=None)
    J = np.empty((model.NPIX, n_alm), dtype=np.float64)
    e = np.zeros(n_alm)
    for i in range(n_alm):
        e[:] = 0.0
        e[i] = 1.0
        alm_ho = _packed_to_alm_ho(e, model.lmax, n_real)
        J[:, i] = sht_full.synthesis_full(alm_ho)
    return J


def main():
    print("Building masked-sky model...")
    model = build_model()
    lmax = model.lmax
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = (lmax - 2) * (lmax - 1) // 2
    n_alm = n_real + n_imag

    print("Building dense full-sky J operator by probing A_action...")
    J = build_dense_J(model, n_real, n_imag)

    print("Checking A^T A ~ diag(norm_const * w_lm) approximation...")
    norm_diag = _build_full_sky_norm_diag(lmax, n_real, n_imag, model.NPIX / (4.0 * np.pi))
    JtJ = J.T @ J
    off_diag = JtJ - np.diag(np.diag(JtJ))
    diag_rel_err = np.abs(np.diag(JtJ) - norm_diag) / norm_diag
    print(f"  diag(J^T J) relative error vs norm_diag: max={diag_rel_err.max():.3e}, mean={diag_rel_err.mean():.3e}")
    print(f"  max |off-diagonal J^T J| / mean(norm_diag): {np.abs(off_diag).max() / norm_diag.mean():.3e}")

    lncl_np = np.log(np.maximum(model.prior_cls[2:lmax], 1e-8))
    lncl_full = np.zeros(lmax)
    lncl_full[2:] = lncl_np
    cl_full = np.exp(lncl_full)
    inv_cl_diag = _build_inv_cl_diag(lmax, cl_full, n_real, n_imag)

    Ninv_full = np.asarray(model.Ninv, dtype=np.float64)
    d_full = np.asarray(model.prior_map, dtype=np.float64)

    print("Building dense masked-sky reference posterior...")
    Lambda = np.diag(inv_cl_diag) + J.T @ (Ninv_full[:, None] * J)
    Sigma_true = np.linalg.inv(Lambda)
    mu_true = Sigma_true @ (J.T @ (Ninv_full * d_full))

    se_mu = np.sqrt(np.diag(Sigma_true) / N_SAMPLES)

    def run_chain(AtA=None, use_block_correction=False, m_group_size=1):
        rng = np.random.default_rng(1)
        s = None
        for _ in range(N_BURNIN):
            s = sample_alm_messenger(
                model, lncl_np, rng, n_messenger_iter=N_MESSENGER_ITER, s0=s,
                AtA=AtA, use_block_correction=use_block_correction,
                m_group_size=m_group_size,
            )
        samples = np.empty((N_SAMPLES, n_alm))
        for i in range(N_SAMPLES):
            for _ in range(THIN):
                s = sample_alm_messenger(
                    model, lncl_np, rng, n_messenger_iter=N_MESSENGER_ITER, s0=s,
                    AtA=AtA, use_block_correction=use_block_correction,
                    m_group_size=m_group_size,
                )
            samples[i] = s
        return samples

    print(f"Running messenger sampler (diagonal A^T A approx): {N_BURNIN} burn-in + {N_SAMPLES}x{THIN} samples...")
    samples_diag = run_chain(AtA=None)
    mu_emp = samples_diag.mean(axis=0)
    Sigma_emp = np.cov(samples_diag, rowvar=False)
    mean_err = np.abs(mu_emp - mu_true) / np.maximum(se_mu, 1e-30)
    rel_cov_err = np.abs(Sigma_emp - Sigma_true) / np.abs(Sigma_true).max()
    print("Results (diagonal approx):")
    print(f"  mean error (in SE units): max={mean_err.max():.2f}, mean={mean_err.mean():.2f}")
    print(f"  cov relative error (vs max entry): max={rel_cov_err.max():.3e}, mean={rel_cov_err.mean():.3e}")

    print(f"\nRunning messenger sampler (exact dense A^T A correction): {N_BURNIN} burn-in + {N_SAMPLES}x{THIN} samples...")
    samples_dense = run_chain(AtA=JtJ)
    mu_emp_d = samples_dense.mean(axis=0)
    Sigma_emp_d = np.cov(samples_dense, rowvar=False)
    mean_err_d = np.abs(mu_emp_d - mu_true) / np.maximum(se_mu, 1e-30)
    rel_cov_err_d = np.abs(Sigma_emp_d - Sigma_true) / np.abs(Sigma_true).max()
    print("Results (exact dense A^T A correction):")
    print(f"  mean error (in SE units): max={mean_err_d.max():.2f}, mean={mean_err_d.mean():.2f}")
    print(f"  cov relative error (vs max entry): max={rel_cov_err_d.max():.3e}, mean={rel_cov_err_d.mean():.3e}")
    print(f"  max |alm| over chain: {np.abs(samples_dense).max():.3e} (diagonal-approx chain: {np.abs(samples_diag).max():.3e})")

    for m_group_size in (1, 3, 5):
        print(f"\nRunning messenger sampler (block-diagonal correction, m_group_size={m_group_size}): {N_BURNIN} burn-in + {N_SAMPLES}x{THIN} samples...")
        samples_block = run_chain(use_block_correction=True, m_group_size=m_group_size)
        mu_emp_b = samples_block.mean(axis=0)
        Sigma_emp_b = np.cov(samples_block, rowvar=False)
        mean_err_b = np.abs(mu_emp_b - mu_true) / np.maximum(se_mu, 1e-30)
        rel_cov_err_b = np.abs(Sigma_emp_b - Sigma_true) / np.abs(Sigma_true).max()
        print(f"Results (block correction, m_group_size={m_group_size}):")
        print(f"  mean error (in SE units): max={mean_err_b.max():.2f}, mean={mean_err_b.mean():.2f}")
        print(f"  cov relative error (vs max entry): max={rel_cov_err_b.max():.3e}, mean={rel_cov_err_b.mean():.3e}")
        print(f"  max |alm| over chain: {np.abs(samples_block).max():.3e}")


if __name__ == "__main__":
    main()
