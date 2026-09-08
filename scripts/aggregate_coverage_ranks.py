"""
ROADMAP.md Section 1, Priority 1: pool the coverage-ensemble chains written by
`scripts/coverage_ensemble_chain.py` into rank statistics and a uniformity
assessment. This is the analysis half of the headline validation; it runs in
seconds on a login node and can be re-run with different statistics without
touching the chains.

WHAT EACH REPORTED NUMBER ESTABLISHES -- read before quoting any of it.

Field-level ranks (phi, alm): these are the statistic this ensemble design
supports directly. alm_true ~ N(0, C_l^fid) and phi_true ~ N(0, C_L^phiphi,fid)
are genuine draws from exactly-known Gaussian priors, so under a correct
sampler the rank of the truth's binned power within each posterior is uniform.
Caveat retained honestly: Blocks 1 and 4 resample the spectra during the chain,
so the sampler's target marginalises over spectra while the truth was generated
at fixed fiducial spectra. That mismatch is second-order for the field ranks
(the spectrum posterior concentrates near the fiducial that generated the data)
but it is not exactly zero, so a mild rank non-uniformity is not by itself
proof of a sampler bug.

Spectrum-level "ranks" (C_l, C_L^phiphi): these are interval-coverage checks
against the *realized* power in the truth field, NOT strict SBC ranks. Block 1's
exact conditional C_l|alm ~ InvGamma(k_l/2 - 1, S_l/2), with k_l = 2l the PACKED
real dof, is exactly the C_l likelihood (both go as C_l^{-k_l/2} exp(-S_l/2C_l)),
i.e. the implied prior on C_l is flat and improper; the same holds for Block 4
unless --cl_phiphi_prior_nu puts a proper conjugate prior on it. An improper prior cannot be sampled,
so no C_l_true ~ p(C_l) exists to rank. Comparing against realized rather than
ensemble-average power is also what avoids the cosmic-variance bug recorded in
achievements.md. Label these as coverage, not calibration, in any figure.

Pooling: the per-bin KS test has almost no power at N=10-20 realizations, so the
pooled-across-bins p-value is reported as the headline. It is anti-conservative,
because bins within one realization share a chain and are correlated -- treat it
as indicative and read the rank histogram as the primary evidence.

Usage:
  PYTHONPATH=diffcmb .venv/bin/python scripts/aggregate_coverage_ranks.py \
      --indir results/analysis/coverage_ensemble --thin 10
"""
import argparse
import glob
import os

import numpy as np

from diffcmb.alm_utils import packed_dof_per_multipole, packed_sizes
from diffcmb.lensing import compute_sl_phi_np
from diffcmb.samplers import _alm_index_lm

# Posterior draws are autocorrelated, so an unthinned rank is not a draw from
# the rank's null distribution. Thinning is the standard SBC remedy; the default
# is deliberately coarse and should be set from the pilot's measured lag-k
# autocorrelation rather than left at the default.
DEFAULT_THIN = 10

KS_FLAG_P = 0.01


def ks_uniform_p(ranks, n_max):
    """Two-sided KS p-value of `ranks` against Uniform{0..n_max}.

    Falls back to NaN (reported, never silently passed) when scipy is missing
    or the sample is too small to say anything.
    """
    try:
        from scipy import stats
    except ImportError:
        return np.nan
    if len(ranks) < 5:
        return np.nan
    # Map integer ranks to (0,1) mid-points so the continuous KS test applies.
    u = (np.asarray(ranks, dtype=np.float64) + 0.5) / (n_max + 1.0)
    return float(stats.kstest(u, "uniform").pvalue)


def binned_power(coeffs, L_arr, lo, hi):
    """Bin-averaged power of a packed real/imag coefficient vector or sample set."""
    mask = (L_arr >= lo) & (L_arr < hi)
    if not mask.any():
        return None
    p = coeffs ** 2
    return p[..., mask].mean(axis=-1)


def rank_of(truth_scalar, posterior_scalars):
    """Rank of the truth among posterior draws: #(draws < truth), in 0..n."""
    return int(np.sum(np.asarray(posterior_scalars) < truth_scalar))


def realized_spectrum(S, lmax):
    """S_l -> realized C_l = S_l / k_l for l=2..lmax-1, zero below.

    k_l is the PACKED real dof at l (2l, not 2l+1 -- splittosingularalm forces
    Im(a_{l,1}) = 0), so this is both the ML estimate of C_l given
    S_l | C_l ~ C_l * chi^2_{k_l} and, under the flat improper prior
    (a0 = -1), exactly the mode beta/(alpha+1) = (S_l/2)/(k_l/2) of the
    conditional that Blocks 1 and 4 draw from.

    Corrected 2026-08-31 alongside the InvGamma shape fix: this divided by
    2l+1 while the sampler had already moved to k_l = 2l, which biased the
    coverage reference high by (2l+1)/2l and drove the C_l^TT / C_L^phiphi
    rank rows further toward 0 than the statistic's own null accounts for.
    """
    k = packed_dof_per_multipole(lmax)
    cl = np.zeros(lmax, dtype=np.float64)
    for ell in range(2, lmax):
        cl[ell] = S[ell] / k[ell]
    return cl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--indir", type=str,
                   default="results/analysis/coverage_ensemble")
    p.add_argument("--thin", type=int, default=DEFAULT_THIN,
                   help="keep every Nth posterior draw before ranking")
    p.add_argument("--burn_frac", type=float, default=0.0,
                   help="additionally discard this leading fraction of each chain")
    p.add_argument("--out", type=str,
                   default="results/analysis/coverage_ranks.npz")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.indir, "chain_r*.npz")))
    files = [f for f in files if not f.endswith("_ckpt.npz")]
    if not files:
        raise SystemExit(f"no chain_r*.npz found in {args.indir} -- nothing to aggregate")

    print(f"=== Coverage-ensemble rank aggregation: {len(files)} realizations ===")
    print(f"thin={args.thin}, burn_frac={args.burn_frac}\n")
    if len(files) < 10:
        print(f"  ! only {len(files)} realizations -- the ROADMAP calls for O(10-20). "
              f"Uniformity statements below are correspondingly weak.\n")

    ell_bins = None
    records = {}  # (quantity, lo, hi) -> list of (rank, n_eff)

    for f in files:
        d = np.load(f, allow_pickle=True)
        lmax = int(d["lmax"])
        n_lncl = lmax - 2
        n_real = lmax * (lmax + 1) // 2 - 3
        n_imag = packed_sizes(lmax)[1]
        L_arr, _m = _alm_index_lm(lmax, n_real, n_imag)

        if ell_bins is None:
            ell_bins = [(lo, min(hi, lmax)) for lo, hi in
                        [(2, 10), (10, 30), (30, 60), (60, 100), (100, 150)]
                        if lo < lmax]

        alm_s = d["alm_samples"]
        phi_s = d["phi_samples"]
        # Block 4 (C_L^phiphi|phi) is optional: chains run with
        # sample_cl_phiphi=False omit this key entirely. Those chains still
        # carry the SBC-supported field-level ranks (phi_power, alm_power),
        # which are the headline calibration evidence -- only the
        # interval-coverage cl_phiphi rows are unavailable.
        has_clpp = "cl_phiphi_samples" in d.files
        clpp_s = np.exp(d["cl_phiphi_samples"]) if has_clpp else None  # log-spectrum
        cl_s = np.exp(alm_s[:, :n_lncl])
        alm_part = alm_s[:, n_lncl:]

        n_tot = len(alm_s)
        start = int(args.burn_frac * n_tot)
        sl = slice(start, None, max(1, args.thin))
        alm_part, phi_s, cl_s = alm_part[sl], phi_s[sl], cl_s[sl]
        if has_clpp:
            clpp_s = clpp_s[sl]
        n_eff = len(alm_part)

        alm_true = d["alm_true_packed"]
        phi_true = d["phi_true_packed"]
        # Spectrum truth is the REALIZED power in the truth field, not the CAMB
        # ensemble mean -- see the cosmic-variance note in the header.
        cl_realized = realized_spectrum(
            _sl_from_packed(alm_true, lmax), lmax
        )
        clpp_realized = realized_spectrum(compute_sl_phi_np(phi_true, lmax), lmax)

        for lo, hi in ell_bins:
            # --- field-level: strict-SBC-supported statistic ---
            for tag, samp, truth in (("phi_power", phi_s, phi_true),
                                     ("alm_power", alm_part, alm_true)):
                post = binned_power(samp, L_arr, lo, hi)
                tru = binned_power(truth, L_arr, lo, hi)
                if post is None:
                    continue
                records.setdefault((tag, lo, hi), []).append(
                    (rank_of(tru, post), n_eff)
                )
            # --- spectrum-level: interval coverage, not calibration ---
            ells = np.arange(max(2, lo), min(lmax, hi))
            if len(ells) == 0:
                continue
            idx = ells - 2
            spectrum_rows = [("cl_TT", cl_s, cl_realized)]
            if has_clpp:
                spectrum_rows.append(("cl_phiphi", clpp_s, clpp_realized))
            for tag, samp, tru_spec in spectrum_rows:
                post = samp[:, idx].mean(axis=1)
                tru = tru_spec[ells].mean()
                records.setdefault((tag, lo, hi), []).append(
                    (rank_of(tru, post), n_eff)
                )

    # --- report ---
    quantities = ["phi_power", "alm_power", "cl_TT", "cl_phiphi"]
    labels = {
        "phi_power": "phi field power   [SBC-supported rank]",
        "alm_power": "alm field power   [SBC-supported rank]",
        "cl_TT": "C_l^TT            [interval coverage, NOT calibration]",
        "cl_phiphi": "C_L^phiphi        [interval coverage, NOT calibration]",
    }
    summary_rows = []
    for q in quantities:
        if not any(k[0] == q for k in records):
            # e.g. cl_phiphi when every chain ran with Block 4 off -- say so
            # rather than printing an empty section that reads like a failure.
            print(f"\n--- {labels[q]} ---")
            print("  (not present in any input chain -- block disabled; skipped)")
            continue
        print(f"\n--- {labels[q]} ---")
        print("  l-bin          N   mean_u   KS_p    ranks")
        pooled_u = []
        for lo, hi in ell_bins:
            entries = records.get((q, lo, hi))
            if not entries:
                continue
            ranks = np.array([e[0] for e in entries], dtype=np.float64)
            n_max = entries[0][1]
            u = (ranks + 0.5) / (n_max + 1.0)
            pooled_u.extend(u.tolist())
            ks_p = ks_uniform_p(ranks, n_max)
            flag = ""
            if np.isfinite(ks_p) and ks_p < KS_FLAG_P:
                flag = "  <-- FLAG"
            rank_str = " ".join(f"{int(r)}" for r in ranks[:12])
            print(f"  [{lo:4d},{hi:4d})  {len(ranks):3d}   {u.mean():6.3f}  "
                  f"{ks_p:6.3f}   {rank_str}{flag}")
            summary_rows.append((q, lo, hi, len(ranks), float(u.mean()), ks_p))

        if pooled_u:
            pooled_u = np.array(pooled_u)
            try:
                from scipy import stats
                pooled_p = float(stats.kstest(pooled_u, "uniform").pvalue)
            except ImportError:
                pooled_p = np.nan
            # Expected mean of a uniform rank is 0.5; the sd of the mean of M
            # *independent* uniforms is 1/sqrt(12M). Bins are correlated within a
            # realization, so this band is optimistic -- stated, not hidden.
            m = len(pooled_u)
            band = 1.0 / np.sqrt(12.0 * m)
            print(f"  POOLED: N={m}  mean_u={pooled_u.mean():.4f} "
                  f"(uniform expects 0.500 +/- {band:.4f}, optimistic band)  "
                  f"KS_p={pooled_p:.4f}")

    print("\n=== How to read this ===")
    print("  A correct sampler gives mean_u ~ 0.5 and no small KS_p. mean_u -> 0")
    print("  means the truth sits below the whole posterior (posterior biased high);")
    print("  mean_u -> 1 the reverse. A U-shaped rank spread means the posterior is")
    print("  too narrow (overconfident); a central clump means too wide.")
    print("  Field ranks are the calibration evidence. Spectrum rows are coverage")
    print("  only -- the implied C_l prior is improper flat, so no strict rank exists.")
    print("  With O(10) realizations these are weak tests; do not read a single")
    print("  flagged bin as a bug without reproducing it at larger N.")

    np.savez(
        args.out,
        summary=np.array(summary_rows, dtype=object),
        n_realizations=len(files),
        thin=args.thin, burn_frac=args.burn_frac,
        ell_bins=np.array(ell_bins, dtype=np.int64),
    )
    print(f"\nSaved rank summary to {args.out}")


def _sl_from_packed(alm_packed, lmax):
    """S_l = sum_m |a_lm|^2 from the packed real/imag layout.

    compute_sl_phi_np implements exactly this weighting (m=0 real-only, m>0
    doubled) and is layout-identical for alm and phi, so it is reused here
    rather than duplicating the loop -- model.compute_sl_np needs a live model
    instance, which this offline aggregator does not build.
    """
    return compute_sl_phi_np(alm_packed, lmax)


if __name__ == "__main__":
    main()
