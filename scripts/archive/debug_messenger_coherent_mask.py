"""
Follow-up to debug_messenger_tau2_schedule.py: that experiment showed the
CURRENT production tau2 (large, near the tau2<=min(N_ii) bound) already
mixes fast on an i.i.d.-randomly-masked toy problem, contradicting the
"just anneal tau2" hypothesis for why production mixes slowly at lmax=300.

The real Planck mask is NOT i.i.d. random masking (30% of pixels scattered
uniformly) -- it excludes large, spatially CONTIGUOUS regions (Galactic
plane + point sources), f_sky=0.772. This is a qualitatively different
structure: contiguous masked regions couple many nearby harmonic modes
together strongly and coherently, which is a well-documented harder case
for Gibbs-type CMB samplers than scattered/independent masking.

This script builds a toy problem with a CONTIGUOUS 1-D "mask" (masked
pixels form one contiguous block, like a stripe) instead of i.i.d. random
mask locations, keeping everything else the same as the previous script, to
test whether contiguous masking -- not tau2 choice -- is what reproduces
production's slow high-inv_cl_diag mixing.
"""
import numpy as np

from diffcmb.messenger import run_messenger_gibbs


def build_toy_problem_contiguous_mask(rng, n_alm=60, n_pix=300, frac_masked=0.3):
    A = np.linalg.qr(rng.standard_normal((n_pix, n_alm)))[0][:, :n_alm]
    inv_cl_diag = np.exp(rng.uniform(np.log(0.01), np.log(100.0), size=n_alm))

    noise_var = rng.uniform(0.2, 1.0, size=n_pix)
    n_masked = int(frac_masked * n_pix)
    # CONTIGUOUS block of masked pixels (a random starting position, one
    # contiguous run), rather than i.i.d. scattered indices.
    start = rng.integers(0, n_pix - n_masked)
    masked_idx = np.arange(start, start + n_masked)
    Ninv = 1.0 / noise_var
    Ninv[masked_idx] = 1e-10

    cl = 1.0 / inv_cl_diag
    s_true = rng.standard_normal(n_alm) * np.sqrt(cl)
    d = A @ s_true + rng.standard_normal(n_pix) * np.sqrt(noise_var)
    d[masked_idx] = 0.0
    return A, inv_cl_diag, Ninv, d


def dense_posterior(A, inv_cl_diag, Ninv, d):
    Lambda = np.diag(inv_cl_diag) + A.T @ (Ninv[:, None] * A)
    Sigma = np.linalg.inv(Lambda)
    mu = Sigma @ (A.T @ (Ninv * d))
    return mu, Sigma


def main():
    rng = np.random.default_rng(123)
    A, inv_cl_diag, Ninv, d = build_toy_problem_contiguous_mask(rng)
    mu_true, Sigma_true = dense_posterior(A, inv_cl_diag, Ninv, d)
    se = np.sqrt(np.diag(Sigma_true))

    tau2_bound = 0.9 * (1.0 / Ninv[Ninv > 1e-6]).min()
    print(f"tau2_bound (production default) = {tau2_bound:.4e}")

    low_mask = inv_cl_diag < np.median(inv_cl_diag)
    high_mask = ~low_mask

    s_far = mu_true + rng.standard_normal(len(mu_true)) * 20 * se

    rng_sampler = np.random.default_rng(42)
    s = s_far.copy()
    checkpoints = [1, 10, 50, 100, 400, 1000, 4000]
    n_done = 0
    print("\n=== fixed_large (production default tau2), contiguous mask ===")
    for target in checkpoints:
        s = run_messenger_gibbs(
            d, Ninv, inv_cl_diag, tau2_bound,
            A_action=lambda x: A @ x, At_action=lambda t: A.T @ t,
            rng=rng_sampler, n_iter=target - n_done, s0=s,
        )
        n_done = target
        z = np.abs(s - mu_true) / se
        print(f"  iter={n_done:5d}  z_low(mean/max)={z[low_mask].mean():6.2f}/{z[low_mask].max():6.2f}  "
              f"z_high(mean/max)={z[high_mask].mean():6.2f}/{z[high_mask].max():6.2f}  "
              f"z_all(mean/max)={z.mean():6.2f}/{z.max():6.2f}")


if __name__ == "__main__":
    main()
