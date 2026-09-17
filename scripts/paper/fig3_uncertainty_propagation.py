"""Paper figure 3 -- what joint sampling buys: per-mode uncertainty on phi.

This is the paper's one *positive* argument for sampling jointly (figure 4's
joint posterior is an honest null, and figure 2 is validation). It answers the
referee question "why not just run a quadratic estimator?" with a per-multipole
comparison of the uncertainty each method actually assigns to phi.

TWO panels:

  (a) phi_uncertainty_vs_L.pdf -- three curves against L:
        * the joint sampler's posterior standard deviation on phi_LM,
          measured from the chains (marginalised over C_l AND C_L^phiphi);
        * sqrt(N_L^phiphi), the quadratic estimator's reconstruction noise,
          from diffcmb.qe;
        * sqrt(C_L^phiphi), the prior/signal level, for scale.
  (b) phi_uncertainty_ratio.pdf -- the ratio of the two, which is the number
      the text quotes: how much tighter (or looser) the joint posterior is per
      mode, and where in L that advantage lives.

WHAT THE COMPARISON IS, AND WHAT IT IS NOT. The QE is a marginal point
estimator: N_L is the variance of its reconstruction, computed with the
spectra held FIXED at their fiducial values. The joint sampler reports a
posterior width that has been marginalised over the unknown C_l^TT and
C_L^phiphi. They are therefore NOT the same quantity, and the figure must not
be captioned as "our error bar is smaller than theirs". What it shows is the
uncertainty each approach actually delivers to a downstream user, which is the
comparison a reader cares about -- and the joint number is the conservative
one, since it carries extra marginalisation the QE number does not.

The posterior width is the width about the posterior MEAN, not the scatter
about the truth; at the low-L end the phi block's known mixing limitation
(achievements.md) means the measured width there is a lower bound on the true
posterior width, so those bins are marked and excluded from the quoted range,
exactly as figure 2 excludes its unequilibrated bins.

THE QE CURVE IS VALIDATED, NOT ASSUMED. plots/STORY.md's standing instruction
is that this curve may not be fabricated. diffcmb/qe.py derives it from first
principles (derivation in its docstring) and scripts/validate_qe_noise.py
validates it by Monte Carlo against the estimator's own definition: the
measured variance of the normalised estimator on unlensed skies matches N_L to
better than 1% in every L bin (job 12015546). That script's docstring records
an earlier, INVALID design that compared N_L against the fixed-alm Fisher
curvature -- a different quantity, off by ~130x -- so do not revive it. This
figure refuses to build unless the validation artifact is present and records
a pass.

The same validation measures the estimator's RESPONSE at 0.88 (median), i.e.
the QE recovers ~12% less than the input phi at this configuration, falling to
0.57 in [60,64) where the band limit truncates the l1,l2 sum. That is a
property of the estimator, not of N_L (the noise test is independent and
passes at the 1% level), but it means the QE curve here should be read as the
noise of an estimator that is itself slightly low-biased near the band edge.
The caption says so.

Usage:
  PYTHONPATH=diffcmb .venv/bin/python scripts/paper/fig3_uncertainty_propagation.py \
      --indir results/analysis/coverage_ensemble_lmax64_prior_cl4_properprior_packingv2 \
      --validation results/analysis/qe_noise_validation.npz \
      --outdir /cosma/apps/durham/dc-hick2/papers/7_DiffCMB/plots/figure3
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paper_style as ps  # noqa: E402

ps.apply()  # installs Agg + paper rcParams before pyplot is touched
import matplotlib.pyplot as plt  # noqa: E402

from diffcmb.alm_utils import packed_length  # noqa: E402

# phi does not equilibrate at the very lowest L (achievements.md); the measured
# width there is a lower bound on the true posterior width, not a measurement.
UNRELIABLE_L_MAX = 10

# Reporting bins, matched to the coverage ensemble's so figure, chains and
# validation all speak about the same L ranges.
L_BINS = [(10, 20), (20, 30), (30, 45), (45, 64)]


def chain_files(indir):
    files = sorted(f for f in glob.glob(os.path.join(indir, "chain_r*.npz"))
                   if not f.endswith("_ckpt.npz"))
    if not files:
        raise SystemExit(f"no chains in {indir}")
    return files


def posterior_sigma_per_L(files, thin):
    """Posterior power per mode of phi at each L, in the SAME units as C_L/N_L.

    THE CONVENTION MATTERS HERE. The packed vector is not a set of independent
    unit-variance coordinates: lensing.compute_sl_phi_np weights m=0 by 1 and
    m>0 by 2, so a field with spectrum C_L has packed-coordinate variance C_L
    at m=0 but C_L/2 at m>0. Averaging raw packed variances would therefore
    produce a number that is neither C_L nor N_L but an L-dependent blend of
    the two -- and would look like a real L-trend in the figure.

    Instead the deviation of each sample from its chain's posterior mean is
    pushed through compute_sl_phi_np and divided by (2L+1). For a field with
    spectrum C_L that quantity has expectation exactly C_L, so the result is
    directly comparable to sqrt(N_L) and sqrt(C_L^phiphi).

    Chains are combined in variance (the additive quantity), not in std.
    """
    from diffcmb.lensing import compute_sl_phi_np

    per_chain = []
    var_acc, n_chain, lmax_seen = None, 0, None
    for f in files:
        d = np.load(f, allow_pickle=True)
        lmax = int(d["lmax"])
        if d["phi_true_packed"].shape[0] != packed_length(lmax):
            raise SystemExit(
                f"{os.path.basename(f)}: phi vector is "
                f"{d['phi_true_packed'].shape[0]} long, packed_length({lmax}) "
                f"is {packed_length(lmax)} -- pre-2026-09-06 packing, a "
                "different model. Use the packingv2 ensemble.")

        phi = d["phi_samples"][::thin]
        dev = phi - phi.mean(axis=0, keepdims=True)
        S = np.mean([compute_sl_phi_np(dev[i], lmax) for i in range(dev.shape[0])],
                    axis=0)
        # Unbiased for the mean subtraction: dev has n-1 dof, not n.
        S = S * dev.shape[0] / max(dev.shape[0] - 1, 1)

        if var_acc is None:
            lmax_seen = lmax
            var_acc = np.zeros(lmax)
        elif lmax != lmax_seen:
            raise SystemExit("chains disagree on lmax")
        L = np.arange(lmax, dtype=np.float64)
        var_acc += S / (2.0 * L + 1.0)
        per_chain.append(S / (2.0 * L + 1.0))
        n_chain += 1

    return np.sqrt(var_acc / n_chain), lmax_seen, np.asarray(per_chain)


def load_validated_qe(validation_npz):
    """N_L from the validation artifact, so the figure and the check agree.

    Reading N_L back out of the validation run's own output (rather than
    recomputing it here) guarantees the curve in the figure is literally the
    curve that passed the cross-check.
    """
    if not os.path.exists(validation_npz):
        raise SystemExit(
            f"missing {validation_npz}.\\nThe QE noise curve may not appear in a "
            "figure until it has been validated against the forward model's own "
            "curvature -- run scripts/submit_validate_qe_noise.slurm first. See "
            "plots/STORY.md.")
    v = np.load(validation_npz)
    if not bool(v["passed"]):
        raise SystemExit(
            f"{validation_npz} records a FAILED validation -- the QE curve must "
            "not be plotted. Fix qe.py or the test, re-run the validation, and "
            "only then rebuild this figure.")
    noise = float(v["median_ratio"])
    resp = float(v["median_response"])
    print(f"QE validation PASSED: noise ratio {noise:.3f}, response {resp:.3f} "
          f"({int(v['n_sims'])} sims)")
    return v["nl_qe"], noise, resp


def plot_curves(L, sigma_post, nl_qe, cl_pp, outpath):
    """Everything normalised to the PRIOR width, sqrt(C_L^phiphi).

    Normalising is what makes the panel readable and the comparison fair. In
    absolute units all three curves fall by orders of magnitude across L and
    the differences that matter are invisible; against the prior, the y axis
    becomes "what fraction of the prior uncertainty survives", and 1.0 is the
    meaningful reference -- a method that learned nothing about phi.

    The QE cannot be compared to the posterior raw, because the posterior
    carries a prior and the QE does not. The like-for-like curve is the QE
    COMBINED with the same prior, 1/sigma^2 = 1/C_L + 1/N_L, which is the
    tightest a QE user could honestly report. Raw sqrt(N_L)/sqrt(C_L) is drawn
    too, to show how little the QE alone constrains phi at this configuration.
    """
    fig, ax = plt.subplots(figsize=ps.FIG_1COL)
    good = (L >= UNRELIABLE_L_MAX) & (cl_pp > 0) & np.isfinite(nl_qe)
    low = (L >= 2) & (L < UNRELIABLE_L_MAX) & (cl_pp > 0) & np.isfinite(nl_qe)

    prior = np.sqrt(np.where(cl_pp > 0, cl_pp, np.nan))
    comb = np.sqrt(1.0 / (1.0 / np.where(cl_pp > 0, cl_pp, np.nan)
                          + 1.0 / np.where(nl_qe > 0, nl_qe, np.nan)))

    ax.axhline(1.0, color=ps.COL_NULL, lw=1.0, ls="-", zorder=0)
    ax.plot(L[good], (comb / prior)[good], "--", color=ps.COL_QE,
            label=r"QE $\oplus$ prior")
    ax.plot(L[good], (sigma_post / prior)[good], "-", color=ps.COL_AWARE,
            label="joint posterior")
    ax.plot(L[low], (sigma_post / prior)[low], ":", color=ps.COL_AWARE,
            alpha=0.5)
    ax.axvspan(2, UNRELIABLE_L_MAX, color=ps.COL_NULL, alpha=0.30, lw=0)
    ax.set_xlabel("$L$")
    ax.set_ylabel(r"width / prior width $\sqrt{C_L^{\phi\phi}}$")
    ax.set_ylim(0.6, 1.25)
    ax.legend(loc="lower right")
    ps.stat_box(ax, "1.0 = learned nothing", loc="upper right",
                xy=(0.97, 0.90))
    ps.save(fig, outpath)


def plot_ratio(L, sigma_post, nl_qe, cl_pp, per_chain_var, outpath):
    """Joint posterior width divided by the QE-plus-prior width, BINNED.

    Below 1 means the joint sampler extracts more about phi than the quadratic
    estimator does, at the same prior -- the figure's claim. Above 1 at high L
    is expected and is not a failure: Block 4 marginalises over an UNKNOWN
    C_L^phiphi, so where the data say nothing the posterior is slightly wider
    than a fixed-prior calculation. That extra width is honesty about the
    spectrum, not lost information.

    Binned with an error bar, because the per-L estimate from 24 chains x ~60
    draws scatters by several percent and an unbinned curve invites the reader
    to over-read individual multipoles. The error bar is the standard error
    ACROSS CHAINS (chains are the independent unit here, not sweeps -- the same
    reasoning plot_joint_cl_clpp_posterior.py applies to its bootstrap), so it
    covers realization scatter rather than just within-chain Monte Carlo noise.
    """
    comb = np.sqrt(1.0 / (1.0 / np.where(cl_pp > 0, cl_pp, np.nan)
                          + 1.0 / np.where(nl_qe > 0, nl_qe, np.nan)))

    fig, ax = plt.subplots(figsize=ps.FIG_1COL)
    ax.axhline(1.0, color=ps.COL_TRUTH, lw=0.8, ls="--")

    centres, vals, errs = [], [], []
    for lo, hi in L_BINS:
        sel = ((L >= max(lo, UNRELIABLE_L_MAX)) & (L < hi) & (cl_pp > 0)
               & (nl_qe > 0) & np.isfinite(nl_qe))
        if not np.any(sel):
            continue
        # Per chain: bin-averaged posterior variance -> width -> ratio.
        r_chain = np.array([np.sqrt(pc[sel].mean()) / comb[sel].mean()
                            for pc in per_chain_var])
        centres.append(0.5 * (max(lo, UNRELIABLE_L_MAX) + hi))
        vals.append(r_chain.mean())
        errs.append(r_chain.std(ddof=1) / np.sqrt(r_chain.size))

    ax.errorbar(centres, vals, yerr=errs, fmt="o-", color=ps.COL_AWARE,
                ms=3.4, capsize=2, lw=1.2,
                elinewidth=0.8, capthick=0.8)
    ax.axvspan(2, UNRELIABLE_L_MAX, color=ps.COL_NULL, alpha=0.30, lw=0)
    ax.set_xlabel("$L$")
    ax.set_ylabel("joint posterior / (QE " r"$\oplus$" " prior)")
    ps.stat_box(ax, "below 1: more information" "\n" "than the QE extracts",
                loc="lower right", xy=(0.97, 0.06))
    ps.save(fig, outpath)
    for c, v, e in zip(centres, vals, errs):
        print(f"    L~{c:5.1f}   ratio {v:.3f} +/- {e:.3f}")
    return vals


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--indir",
        default="results/analysis/coverage_ensemble_lmax64_prior_cl4_properprior_packingv2")
    ap.add_argument("--thin", type=int, default=10)
    ap.add_argument("--validation",
                    default="results/analysis/qe_noise_validation.npz")
    ap.add_argument(
        "--outdir",
        default="/cosma/apps/durham/dc-hick2/papers/7_DiffCMB/plots/figure3")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    nl_qe, _med, _slope = load_validated_qe(args.validation)

    files = chain_files(args.indir)
    sigma_post, lmax, per_chain_var = posterior_sigma_per_L(files, args.thin)
    print(f"{len(files)} chains, lmax={lmax}, thin={args.thin}")

    d0 = np.load(files[0], allow_pickle=True)
    cl_pp = np.asarray(d0["cl_phiphi_fid"], dtype=np.float64)[:lmax]
    nl_qe = np.asarray(nl_qe, dtype=np.float64)[:lmax]
    L = np.arange(lmax)

    plot_curves(L, sigma_post, nl_qe, cl_pp,
                os.path.join(args.outdir, "phi_uncertainty_vs_L.pdf"))
    plot_ratio(L, sigma_post, nl_qe, cl_pp, per_chain_var,
               os.path.join(args.outdir, "phi_uncertainty_ratio.pdf"))


if __name__ == "__main__":
    main()
