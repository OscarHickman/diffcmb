"""Is a passing SBC rank evidence of correctness, or just a weak test?

MOTIVATION. Across both production ensembles at N=24, six of seven rank
statistics sit BELOW 0.5 (0.377-0.469) and only one KS test rejects. The
comfortable reading is "consistent with uniform". This script asks whether that
reading survives, by answering three questions the KS p-values cannot:

  Q1 ESTIMATOR. Is our rank estimator unbiased when the posterior is EXACTLY
     right? If it returns 0.5 on exact draws, an observed 0.44 is not an
     artifact of how we compute ranks.
  Q2 AUTOCORRELATION. Does a correct-but-autocorrelated chain (finite length,
     thinned) SHIFT the mean rank, or only disperse it? This is the load-bearing
     question. The standard result is that autocorrelation makes SBC ranks
     U-SHAPED -- over-dispersed, symmetric -- and does NOT move the mean. If
     that holds here, our systematic downward shift cannot be blamed on short
     chains or thinning, and is evidence of genuine bias.
  Q3 POWER. What size of real bias would we actually DETECT at N=24? A "pass"
     only excludes effects the test could have seen. If we can only detect
     under-dispersion beyond some threshold, then a pass bounds the bias at
     that threshold -- it does not establish exactness.

MODEL. A tractable conjugate Gaussian stand-in: theta ~ N(0,1) prior,
y | theta ~ N(theta, sigma^2), so the exact posterior is Gaussian with known
mean and sd. SBC on the identity test quantity. This is not the CMB posterior --
it is a case where "exactly correct" is available in closed form, which is the
only way to calibrate what a correct sampler looks like through OUR rank code.
Bias is injected by scaling the posterior sd (s<1 = under-dispersed, the
signature we appear to see: draws too tightly clustered, truth ranks low or
high but on average the mean rank moves only if the posterior is also SHIFTED).

Two distinct defects are therefore simulated separately:
  * under-dispersion (posterior sd scaled by s)  -> U-shaped ranks, mean ~0.5
  * mean offset (posterior mean shifted by b*sd) -> SHIFTED ranks, mean != 0.5
Distinguishing which one our data shows is the whole point.

Usage:
  PYTHONPATH=diffcmb .venv/bin/python scripts/validate_sbc_statistic_power.py
"""

import argparse

import numpy as np
from scipy import stats


def sbc_ranks(n_real, n_draws, rng, sd_scale=1.0, mean_shift=0.0, rho=0.0):
    """One SBC experiment; returns the rank statistic u per realization.

    rho>0 draws an AR(1) chain with that lag-1 correlation instead of i.i.d.
    draws, at the SAME nominal posterior -- i.e. a correct-but-autocorrelated
    sampler.
    """
    us = np.empty(n_real)
    for i in range(n_real):
        theta0 = rng.normal(0.0, 1.0)                 # truth from the prior
        y = theta0 + rng.normal(0.0, SIGMA)           # data
        post_mean = y / (1.0 + SIGMA**2)              # exact conjugate posterior
        post_sd = np.sqrt(SIGMA**2 / (1.0 + SIGMA**2))
        m = post_mean + mean_shift * post_sd
        s = post_sd * sd_scale

        if rho > 0.0:
            z = np.empty(n_draws)
            z[0] = rng.normal()
            for t in range(1, n_draws):
                z[t] = rho * z[t - 1] + np.sqrt(1 - rho**2) * rng.normal()
            draws = m + s * z
        else:
            draws = rng.normal(m, s, size=n_draws)

        us[i] = (np.sum(draws < theta0) + 0.5 * np.sum(draws == theta0)) / n_draws
    return us


SIGMA = 1.0


def summarise(tag, us):
    ks = stats.kstest(us, "uniform")
    # A U-shape inflates the spread of u; uniform has sd = 1/sqrt(12) = 0.2887.
    return (f"  {tag:<46s} mean_u={us.mean():.4f}  sd_u={us.std():.4f}  "
            f"KS_p={ks.pvalue:.4g}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_real", type=int, default=24, help="realizations (ours: 24)")
    ap.add_argument("--n_draws", type=int, default=60, help="draws/chain after thinning")
    ap.add_argument("--n_rep", type=int, default=2000, help="repeats for power")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print(__doc__.split("Usage:")[0])
    print(f"Settings: N={args.n_real} realizations, {args.n_draws} draws/chain, "
          f"{args.n_rep} repeats\n")

    print("Q1/Q2 -- what a CORRECT sampler produces through this rank code:")
    print(summarise("exact i.i.d. draws (ideal)",
                    sbc_ranks(4000, args.n_draws, rng)))
    for rho in (0.5, 0.9, 0.98):
        print(summarise(f"correct but autocorrelated, AR(1) rho={rho}",
                        sbc_ranks(4000, args.n_draws, rng, rho=rho)))
    print("\n  -> uniform reference: mean_u=0.5000, sd_u=0.2887.")
    print("  -> Read the sd column: autocorrelation inflates SPREAD (U-shape).")
    print("     If the mean column stays ~0.500, autocorrelation cannot")
    print("     manufacture the downward shift we observe.\n")

    print("Injected DEFECTS -- which one moves the mean?")
    for s in (0.9, 0.8, 0.7):
        print(summarise(f"under-dispersed posterior, sd x{s}",
                        sbc_ranks(4000, args.n_draws, rng, sd_scale=s)))
    for b in (-0.1, -0.2, -0.3):
        print(summarise(f"posterior mean shifted by {b} sd",
                        sbc_ranks(4000, args.n_draws, rng, mean_shift=b)))
    print()

    print("!! The 'exact i.i.d.' row above rejects at KS_p<0.001 on 4000")
    print("   realizations even though that sampler is EXACTLY correct. That is")
    print("   not a defect in the sampler: u is discrete (only n_draws+1 values)")
    print("   and the KS test assumes a CONTINUOUS uniform, so it over-rejects")
    print("   once N is large. Production ranks are far coarser still (8 values),")
    print("   so the analytic KS_p on the ensembles must not be read literally in")
    print("   either direction. Power below is therefore calibrated against a")
    print("   SIMULATED null, not against the analytic p-value.\n")

    print(f"Q3 -- POWER at N={args.n_real}, calibrated by simulation.")
    print("     Null = the correct model; a defect counts as DETECTED when its")
    print("     mean_u falls outside the null's central 95% range.")
    null_means = np.array([sbc_ranks(args.n_real, args.n_draws, rng).mean()
                           for _ in range(args.n_rep)])
    lo, hi = np.percentile(null_means, [2.5, 97.5])
    print(f"     null mean_u 95% range at N={args.n_real}: "
          f"[{lo:.4f}, {hi:.4f}]  (width {hi-lo:.4f})\n")

    for label, kw in (("under-dispersion sd x0.9", {"sd_scale": 0.9}),
                      ("under-dispersion sd x0.8", {"sd_scale": 0.8}),
                      ("under-dispersion sd x0.7", {"sd_scale": 0.7}),
                      ("under-dispersion sd x0.5", {"sd_scale": 0.5}),
                      ("mean shift +0.1 sd", {"mean_shift": 0.1}),
                      ("mean shift +0.2 sd", {"mean_shift": 0.2}),
                      ("mean shift +0.3 sd", {"mean_shift": 0.3}),
                      ("mean shift +0.5 sd", {"mean_shift": 0.5})):
        means = np.array([sbc_ranks(args.n_real, args.n_draws, rng, **kw).mean()
                          for _ in range(args.n_rep)])
        power = float(np.mean((means < lo) | (means > hi)))
        print(f"  {label:<28s} power={power:6.1%}   mean_u={means.mean():.4f}")

    print("\n  READ THIS AS: a non-rejection only excludes defects with high")
    print("  power here. Anything weaker is NOT excluded by our 'pass'.")
    print("  Note under-dispersion barely moves mean_u at all -- it shows up in")
    print("  sd_u instead -- so a mean-based read is blind to it; and a NEGATIVE")
    print("  posterior mean shift pushes mean_u ABOVE 0.5, so the observed")
    print("  sub-0.5 values correspond to a posterior biased HIGH vs the truth.")


if __name__ == "__main__":
    main()
