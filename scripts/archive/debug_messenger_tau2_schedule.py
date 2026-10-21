"""
Cheap, small-scale exploration: which tau2 choice/schedule speeds up
messenger-sampler mixing for HIGH-inv_cl_diag ("high-l analog") modes, the
specific pattern ROADMAP.md Phase 0c Step 6's production-scale diagnostic
found mixing slowly (job 11580369: l=2 converges by iter~30, l=100-299 still
9-14x off the reference at iter=3000)?

Before spending more multi-hour SLURM compute on a guessed annealing
direction, use the SAME dense-reference toy problem tests/test_messenger.py
already validates against (cheap, seconds not hours) to empirically compare:
  (a) fixed tau2 at the current production default (0.9 * bound, i.e. as
      LARGE/loose as the tau2<=min(N_ii) constraint allows)
  (b) fixed tau2 much SMALLER than the bound
  (c) a two-stage annealed schedule, both directions (small-then-large,
      large-then-small), same total iteration budget as (a)/(b)

Measures, per mode, the z-score |s - mu_true| / SE(mu_true) as a function of
iteration count, split by whether the mode has high or low inv_cl_diag (the
toy-problem analog of high/low l), starting from a far-from-target warm
state (mimicking the production checkpoint's already-drifted alm state).
"""
import numpy as np

from diffcmb.messenger import run_messenger_gibbs


def build_toy_problem(rng, n_alm=60, n_pix=300, frac_masked=0.3, mask_ninv_floor=1e-10):
    A = np.linalg.qr(rng.standard_normal((n_pix, n_alm)))[0][:, :n_alm]
    # Wide dynamic range in inv_cl_diag, mimicking real C_l falling steeply
    # with l: some modes near-flat prior (low l), some tightly constrained
    # (high l, steep ΛCDM damping tail).
    inv_cl_diag = np.exp(rng.uniform(np.log(0.01), np.log(100.0), size=n_alm))

    noise_var = rng.uniform(0.2, 1.0, size=n_pix)
    n_masked = int(frac_masked * n_pix)
    masked_idx = rng.choice(n_pix, size=n_masked, replace=False)
    Ninv = 1.0 / noise_var
    Ninv[masked_idx] = mask_ninv_floor

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


def run_schedule(d, Ninv, inv_cl_diag, A, tau2_schedule, s0, seed):
    """tau2_schedule: list of (tau2, n_iter) run in order, total iterations
    tracked and z-scores reported after each stage."""
    rng = np.random.default_rng(seed)
    s = s0.copy()
    log = []
    n_done = 0
    for tau2, n_iter in tau2_schedule:
        s = run_messenger_gibbs(
            d, Ninv, inv_cl_diag, tau2,
            A_action=lambda x: A @ x, At_action=lambda t: A.T @ t,
            rng=rng, n_iter=n_iter, s0=s,
        )
        n_done += n_iter
        log.append((n_done, s.copy()))
    return log


def main():
    rng = np.random.default_rng(123)
    A, inv_cl_diag, Ninv, d = build_toy_problem(rng)
    mu_true, Sigma_true = dense_posterior(A, inv_cl_diag, Ninv, d)
    se = np.sqrt(np.diag(Sigma_true))

    tau2_bound = 0.9 * (1.0 / Ninv[Ninv > 1e-6]).min()
    print(f"tau2_bound (production default) = {tau2_bound:.4e}")
    print(f"inv_cl_diag range: [{inv_cl_diag.min():.3e}, {inv_cl_diag.max():.3e}]")

    low_mask = inv_cl_diag < np.median(inv_cl_diag)
    high_mask = ~low_mask

    # Start far from the target, mimicking the production checkpoint's
    # already-drifted alm state relative to a fresh chain.
    s_far = mu_true + rng.standard_normal(len(mu_true)) * 20 * se

    total_iter = 400
    variants = {
        "fixed_large (production default)": [(tau2_bound, total_iter)],
        "fixed_small (0.01x bound)": [(tau2_bound * 0.01, total_iter)],
        "fixed_tiny (0.0001x bound)": [(tau2_bound * 0.0001, total_iter)],
        "anneal small->large": [(tau2_bound * 0.0001, total_iter // 2), (tau2_bound, total_iter // 2)],
        "anneal large->small": [(tau2_bound, total_iter // 2), (tau2_bound * 0.0001, total_iter // 2)],
    }

    for name, schedule in variants.items():
        print(f"\n=== {name} ===")
        log = run_schedule(d, Ninv, inv_cl_diag, A, schedule, s_far, seed=42)
        for n_done, s in log:
            z = np.abs(s - mu_true) / se
            print(f"  iter={n_done:5d}  z_low(mean/max)={z[low_mask].mean():6.2f}/{z[low_mask].max():6.2f}  "
                  f"z_high(mean/max)={z[high_mask].mean():6.2f}/{z[high_mask].max():6.2f}  "
                  f"z_all(mean/max)={z.mean():6.2f}/{z.max():6.2f}")


if __name__ == "__main__":
    main()
