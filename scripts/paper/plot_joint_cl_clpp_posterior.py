"""The paper's headline differentiator: the joint (C_l^TT, C_L^phiphi) posterior.

WHAT THIS IS FOR. No competing method produces this object. QE and MUSE give a
point estimate or a marginal for phi; Commander/Flinch/Almanac sample (map, C_l)
with no phi block at all; a diffusion/score model draws phi samples but carries
no C_l posterior to correlate them with. A joint Gibbs sampler over
(alm, C_l, phi, C_L^phiphi) has the full joint in hand, so the cross-covariance
between the two spectra is measurable rather than assumed.

WHAT IT IS NOT. This is the posterior this sampler produces; it is not a
calibration-validated one. The configuration it is drawn from (job 11903182,
proper prior nu=6) has a strict C_L^phiphi SBC rank of 0.3802 (KS_p = 0.0013) --
much improved by the 2026-08-31 dof fix, not yet uniform. Every figure this
script writes carries that caveat in its own caption text, so it cannot be
lifted into a talk without it.

METHOD, and why each step is needed.

  * WITHIN-CHAIN correlation, standardised per realization before pooling.
    Pooling raw draws across realizations would measure the scatter of the 12
    TRUTHS (cosmic variance) rather than the correlation inside any one
    posterior -- a much larger, and completely different, quantity. Each chain
    is centred and scaled by its own mean/std first, so what is pooled is
    "how do these two spectra move together within a single posterior".

  * THINNING by the run's own tau_int. The correlation estimate itself is
    unbiased by autocorrelation, but its UNCERTAINTY is not: 600 sweeps at
    tau_int ~ 24-56 carry only O(12-25) independent draws per chain. Thinning
    makes the effective sample size visible instead of implied.

  * A CHAIN-LEVEL BOOTSTRAP for the error bar. Resampling the 12 realizations
    (not the sweeps) respects the fact that draws within a chain are correlated
    and only the chains are independent. With 12 chains this is a coarse
    interval and is reported as such.

  * A PERMUTATION NULL. Correlating two spectra estimated from O(10) effective
    draws produces a spurious correlation of order 1/sqrt(N_eff) ~ 0.2-0.3 with
    no physics in it at all. The null shuffles the C_L^phiphi draws against the
    C_l^TT draws WITHIN each chain, destroying the sweep-level pairing while
    keeping every marginal intact, so the reported significance is against what
    this estimator does on uncorrelated data -- not against zero.

Usage:
  PYTHONPATH=diffcmb .venv/bin/python scripts/paper/plot_joint_cl_clpp_posterior.py \
      --indir results/analysis/coverage_ensemble_lmax64_prior_cl4_properprior_doffix \
      --thin 45 --outdir results/analysis/figures
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CL_BINS = [(2, 10), (10, 30), (30, 60), (60, 64)]
CLPP_BINS = [(2, 10), (10, 30), (30, 60), (60, 64)]


def load_pairs(indir, thin):
    """Per-realization (binned C_l^TT, binned C_L^phiphi) draw pairs.

    Both spectra are stored as logs in the chain (C_l^TT in the first lmax-2
    entries of alm_samples, C_L^phiphi in cl_phiphi_samples), so both are
    exponentiated before binning -- binning in log would report the correlation
    of a different, non-linear function of the spectra.
    """
    files = sorted(f for f in glob.glob(os.path.join(indir, "chain_r*.npz"))
                   if not f.endswith("_ckpt.npz"))
    if not files:
        raise SystemExit(f"no chains in {indir}")
    out = []
    for f in files:
        d = np.load(f)
        if "cl_phiphi_samples" not in d.files:
            raise SystemExit(
                f"{os.path.basename(f)} has no cl_phiphi_samples -- this figure "
                "needs a Block-4-ON run (and, for a proper posterior, one with "
                "--cl_phiphi_prior_nu)."
            )
        lmax = int(d["lmax"])
        n_lncl = lmax - 2
        cl = np.exp(d["alm_samples"][::thin, :n_lncl])          # l = 2..lmax-1
        clpp = np.exp(d["cl_phiphi_samples"][::thin])           # L = 2..lmax-1
        cl_b = np.column_stack([cl[:, np.arange(lo, min(hi, lmax)) - 2].mean(axis=1)
                                for lo, hi in CL_BINS])
        clpp_b = np.column_stack([clpp[:, np.arange(lo, min(hi, lmax)) - 2].mean(axis=1)
                                  for lo, hi in CLPP_BINS])
        nu = float(d["cl_phiphi_prior_nu"]) if "cl_phiphi_prior_nu" in d.files else 0.0
        out.append((cl_b, clpp_b, nu))
    return files, out


def standardise(a):
    """Centre and scale a chain's draws by its own mean/std, column-wise."""
    s = a.std(axis=0, ddof=1)
    return (a - a.mean(axis=0)) / np.where(s > 0, s, 1.0)


def pooled_corr(pairs):
    """Mean within-posterior correlation matrix, C_l bins x C_L^phiphi bins."""
    x = np.concatenate([standardise(cl) for cl, _, _ in pairs])
    y = np.concatenate([standardise(pp) for _, pp, _ in pairs])
    n = x.shape[0]
    return (x.T @ y) / (n - 1)


def bootstrap_corr(pairs, n_rep, rng):
    """Chain-level bootstrap: resample REALIZATIONS, not sweeps."""
    k = len(pairs)
    reps = np.empty((n_rep, len(CL_BINS), len(CLPP_BINS)))
    for r in range(n_rep):
        idx = rng.integers(0, k, size=k)
        reps[r] = pooled_corr([pairs[i] for i in idx])
    return reps


def permutation_null(pairs, n_rep, rng):
    """Shuffle the sweep-level pairing within each chain, keep the marginals."""
    reps = np.empty((n_rep, len(CL_BINS), len(CLPP_BINS)))
    for r in range(n_rep):
        shuffled = [(cl, pp[rng.permutation(pp.shape[0])], nu)
                    for cl, pp, nu in pairs]
        reps[r] = pooled_corr(shuffled)
    return reps


def plot(pairs, corr, lo, hi, null_abs95, outpath, caveat):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax = axes[0]
    # Show the STRONGEST cell relative to its own null, not a fixed pair -- a
    # hand-picked bin pair would either flatter or bury the result depending on
    # which one was chosen when the script was written.
    i, j = np.unravel_index(np.argmax(np.abs(corr) / null_abs95), corr.shape)
    x = np.concatenate([standardise(cl)[:, i] for cl, _, _ in pairs])
    y = np.concatenate([standardise(pp)[:, j] for _, pp, _ in pairs])
    ax.scatter(x, y, s=14, alpha=0.45, edgecolor="none", color="#1f4e79")
    fit = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 2)
    detected = abs(corr[i, j]) > null_abs95[i, j]
    ax.plot(xs, np.polyval(fit, xs), color="#c00000", lw=1.6,
            label=(f"r = {corr[i, j]:+.3f}  "
                   f"(95% null $|r|$ < {null_abs95[i, j]:.2f})"))
    ax.axhline(0, color="0.7", lw=0.6)
    ax.axvline(0, color="0.7", lw=0.6)
    cl_lo, cl_hi = CL_BINS[i]
    pp_lo, pp_hi = CLPP_BINS[j]
    ax.set_xlabel(rf"$C_\ell^{{TT}}$, $\ell \in [{cl_lo},{cl_hi})$  "
                  "(standardised per chain)")
    ax.set_ylabel(rf"$C_L^{{\phi\phi}}$, $L \in [{pp_lo},{pp_hi})$  (standardised)")
    ax.set_title("Strongest bin pair"
                 + ("" if detected else " -- still within the null"))
    ax.legend(frameon=False, loc="upper left", fontsize=8)

    ax = axes[1]
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
    ax.set_xticks(range(len(CLPP_BINS)))
    ax.set_xticklabels([f"[{a},{b})" for a, b in CLPP_BINS], fontsize=8)
    ax.set_yticks(range(len(CL_BINS)))
    ax.set_yticklabels([f"[{a},{b})" for a, b in CL_BINS], fontsize=8)
    ax.set_xlabel(r"$C_L^{\phi\phi}$ bin")
    ax.set_ylabel(r"$C_\ell^{TT}$ bin")
    ax.set_title("Within-posterior correlation")
    for i in range(len(CL_BINS)):
        for j in range(len(CLPP_BINS)):
            sig = "*" if abs(corr[i, j]) > null_abs95[i, j] else ""
            ax.text(j, i, f"{corr[i, j]:+.2f}{sig}", ha="center", va="center",
                    fontsize=8,
                    color="white" if abs(corr[i, j]) > 0.35 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Joint $(C_\\ell^{TT},\\,C_L^{\\phi\\phi})$ posterior "
                 "-- the object no marginal method produces", y=1.01)
    fig.text(0.5, -0.06, caveat, ha="center", va="top", fontsize=7.5,
             wrap=True, color="0.25")
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    print(f"wrote {outpath}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--indir", required=True)
    p.add_argument("--thin", type=int, default=45,
                   help="re-derive from the run's own tau_int; do not carry over")
    p.add_argument("--outdir", default="results/analysis/figures")
    p.add_argument("--n_rep", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    files, pairs = load_pairs(args.indir, args.thin)
    nu = pairs[0][2]
    n_eff = sum(cl.shape[0] for cl, _, _ in pairs)
    print(f"=== joint (C_l^TT, C_L^phiphi) posterior: {len(files)} chains, "
          f"thin={args.thin}, {n_eff} pooled draws ===")
    if not (np.isfinite(nu) and nu > 0):
        print("  WARNING: this run used the FLAT improper C_L^phiphi prior, so "
              "the C_L^phiphi\n           posterior is not a proper Bayesian "
              "object in the phi amplitude. Use a\n           --cl_phiphi_prior_nu "
              "run for anything that goes in the paper.")

    corr = pooled_corr(pairs)
    boot = bootstrap_corr(pairs, args.n_rep, rng)
    lo = np.percentile(boot, 2.5, axis=0)
    hi = np.percentile(boot, 97.5, axis=0)
    null = permutation_null(pairs, args.n_rep, rng)
    null_abs95 = np.percentile(np.abs(null), 95, axis=0)

    print(f"\n  {'C_l bin':10s} {'C_LPP bin':10s} {'corr':>7s} "
          f"{'boot 95%':>17s} {'|null| 95%':>11s}  verdict")
    for i, (a, b) in enumerate(CL_BINS):
        for j, (c, e) in enumerate(CLPP_BINS):
            v = ("above null" if abs(corr[i, j]) > null_abs95[i, j]
                 else "consistent with null")
            print(f"  [{a:2d},{b:2d})   [{c:2d},{e:2d})   {corr[i, j]:+7.3f} "
                  f"[{lo[i, j]:+6.3f},{hi[i, j]:+6.3f}] {null_abs95[i, j]:11.3f}  {v}")

    print("\n  The permutation null is the number that matters: with O(10) "
          "effectively\n  independent draws per chain, this estimator produces "
          f"|r| up to ~{null_abs95.max():.2f}\n  on data with no correlation at "
          "all. Only entries marked 'above null' are\n  evidence of anything.")

    n_above = int(np.sum(np.abs(corr) > null_abs95))
    n_cells = corr.size
    expected = 0.05 * n_cells
    floor = float(null_abs95.mean())
    print(f"\n  DETECTION SUMMARY: {n_above}/{n_cells} cells above their null; "
          f"{expected:.1f} expected by chance\n  at the 95% level. "
          + ("No excess -- this ensemble does NOT detect a correlation."
             if n_above <= expected + 1 else
             "An excess over chance; inspect the pattern, not single cells."))
    print(f"  POWER: the noise floor is |r| ~ {floor:.2f} at N_eff = {n_eff}. "
          "Since sigma_r ~ 1/sqrt(N_eff),\n  resolving |r| = 0.10 at 2 sigma "
          f"needs N_eff ~ {int(4 / 0.10 ** 2)}, and |r| = 0.05 needs "
          f"N_eff ~ {int(4 / 0.05 ** 2)}\n  -- i.e. "
          f"{4 / 0.10 ** 2 / max(n_eff, 1):.1f}x and "
          f"{4 / 0.05 ** 2 / max(n_eff, 1):.1f}x this ensemble. A null here is "
          "a statement about\n  sample size, not about physics. Note also that "
          "C_l^TT is the UNLENSED spectrum and\n  C_L^phiphi depends on phi "
          "alone given phi, so the two couple only through the joint\n  "
          "dependence on the data via the lensing likelihood -- a small "
          "correlation is the\n  physically expected outcome, which is exactly "
          "why the noise floor has to be shown.")

    os.makedirs(args.outdir, exist_ok=True)
    caveat = (
        f"Source: {os.path.basename(args.indir.rstrip('/'))}, {len(files)} chains, "
        f"thin={args.thin} ({n_eff} pooled draws), proper C_L^phiphi prior nu={nu:g}. "
        "Correlations are WITHIN-posterior (each chain standardised before pooling), "
        "so this is not cosmic-variance scatter across realizations. * marks entries "
        "outside the 95% permutation null. CAVEAT: this configuration's strict "
        "C_L^phiphi SBC rank is 0.3802 (KS_p=0.0013) -- improved by the dof fix but "
        "not yet uniform, so this is the posterior the sampler produces, not a "
        "calibration-validated one."
    )
    out = os.path.join(args.outdir, "joint_cl_clpp_posterior.png")
    plot(pairs, corr, lo, hi, null_abs95, out, caveat)
    np.savez(os.path.join(args.outdir, "joint_cl_clpp_posterior.npz"),
             corr=corr, boot_lo=lo, boot_hi=hi, null_abs95=null_abs95,
             cl_bins=np.array(CL_BINS), clpp_bins=np.array(CLPP_BINS),
             thin=args.thin, n_pooled=n_eff, prior_nu=nu, indir=args.indir)


if __name__ == "__main__":
    main()
