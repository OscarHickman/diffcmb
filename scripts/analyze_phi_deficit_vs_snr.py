"""
ROADMAP.md "Currently doing" item 0: the free per-l-bin phi-power deficit vs
S/N plot that discriminates the two live hypotheses for the lmax=300
phi-power deficit found in achievements.md (roughly 51-86% under-recovery
across l=10-300):

  1. Under-mixing retaining a Wiener-suppressed start (Millea, Anderes &
     Wandelt arXiv:2002.00965) -- predicts the deficit is WORST AT LOW S/N,
     since Wiener filtering suppresses exactly the low-S/N modes and the
     lmax=128 pilot's slow-mixing l=[60,100) bin would be the same
     phenomenon at a different scale.
  2. Sampler geometry degrading at high S/N (Taylor, Ashdown & Hobson
     arXiv:0708.2989 report HMC correlation lengths degrading specifically
     at the highest S/N) -- predicts the deficit is WORST AT HIGH S/N, and
     would strengthen the case for an MCLMC port.

No new MCMC compute: this is pure post-processing of the posterior phi
samples already saved by scripts/validate_sim_lmax300_lensing.py's
--phi_n_lfs 80 run (results/analysis/validate_sim_lmax300_phi80.npz, see
achievements.md's "lmax=300 point-validation run at phi_n_lfs=80" entry).

S/N per bin is defined as the recovered signal in units of posterior
uncertainty (post_mean / post_std of the binned phi power trace) -- the
natural free "how detected is this bin" quantity already available from the
sampler's own posterior spread, using the same raw phi**2 power convention
as scripts/validate_sim_lmax300_lensing.py::binned_power_recovery (so the
deficit numbers here are directly comparable to the ones already quoted in
achievements.md).

Usage: PYTHONPATH=diffcmb .venv/bin/python scripts/analyze_phi_deficit_vs_snr.py \
    --checkpoint results/analysis/validate_sim_lmax300_phi80.npz --lmax 300 \
    --out results/analysis/phi_deficit_vs_snr_lmax300.png
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diffcmb.alm_utils import packed_sizes
from diffcmb.samplers import _alm_index_lm


def binned_deficit_and_snr(phi_samples, phi_true_packed, L_arr, ell_bins):
    """Per-bin (deficit_frac, snr, l_mid) using the same raw phi**2 power
    convention as validate_sim_lmax300_lensing.py::binned_power_recovery."""
    power_samples = phi_samples ** 2
    true_power = phi_true_packed ** 2
    rows = []
    for lo, hi in ell_bins:
        mask = (L_arr >= lo) & (L_arr < hi)
        if not mask.any():
            continue
        post_power_mean = power_samples[:, mask].mean(axis=1)
        post_mean = post_power_mean.mean()
        post_std = post_power_mean.std()
        truth_mean = true_power[mask].mean()
        if truth_mean <= 0 or post_std <= 0:
            continue
        deficit = 1.0 - post_mean / truth_mean
        snr = post_mean / post_std
        rows.append((lo, hi, 0.5 * (lo + hi), deficit, snr, truth_mean, post_mean, post_std))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str,
                   default="results/analysis/validate_sim_lmax300_phi80.npz")
    p.add_argument("--lmax", type=int, default=300)
    p.add_argument("--bin_width", type=int, default=10,
                   help="l-bin width for the fine-grained scan (default 10, "
                        "covers l=10..lmax).")
    p.add_argument("--out", type=str,
                   default="results/analysis/phi_deficit_vs_snr_lmax300.png")
    args = p.parse_args()
    lmax = args.lmax

    ck = np.load(args.checkpoint, allow_pickle=True)
    phi_samples = ck["phi_samples"]
    phi_true_packed = ck["phi_true_packed"]
    print(f"Loaded {phi_samples.shape[0]} posterior phi samples from {args.checkpoint}")

    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = packed_sizes(lmax)[1]
    L_arr, _m_arr = _alm_index_lm(lmax, n_real, n_imag)

    ell_bins = [(lo, min(lo + args.bin_width, lmax))
                for lo in range(10, lmax, args.bin_width)]
    rows = binned_deficit_and_snr(phi_samples, phi_true_packed, L_arr, ell_bins)

    print("\n  l-bin          l_mid    deficit    S/N")
    for lo, hi, l_mid, deficit, snr, _truth_mean, _post_mean, _post_std in rows:
        print(f"  [{lo:4d},{hi:4d})   {l_mid:6.1f}   {100*deficit:6.1f}%   {snr:8.2f}")

    l_mid_all = np.array([r[2] for r in rows])
    deficit_all = np.array([r[3] for r in rows])
    snr_all = np.array([r[4] for r in rows])

    # Fractional deficit is a single-realization statistic (compared against
    # the realized phi_true draw, not a CAMB ensemble mean -- same convention
    # as validate_sim_lmax300_lensing.py::binned_power_recovery, deliberately,
    # per the cosmic-variance bug in achievements.md). At low l the mode
    # count per bin (~2l+1) is small enough that a single draw can land far
    # below its ensemble mean by chance, blowing up the fractional ratio
    # without any sampler pathology involved. Guard against that swamping the
    # rank correlation by dropping bins with an implausible |deficit| instead
    # of silently trusting them.
    sane = np.abs(deficit_all) < 5.0
    n_dropped = (~sane).sum()
    if n_dropped:
        print(f"\nDropping {n_dropped} bin(s) with |deficit| >= 500% (a low-l "
              f"cosmic-variance fluke in the realized truth draw, not a sampler "
              f"signal -- see the docstring) before computing the rank correlation.")
    l_mid, deficit, snr = l_mid_all[sane], deficit_all[sane], snr_all[sane]

    # Spearman-style rank correlation (dependency-free: no scipy import needed
    # beyond what's already a hard dependency elsewhere in the repo) between
    # deficit and S/N -- the sign is the whole verdict.
    def rank(x):
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(x))
        return ranks

    rd, rs = rank(deficit), rank(snr)
    rho = np.corrcoef(rd, rs)[0, 1]

    print(f"\nSpearman rank correlation (deficit vs S/N): rho = {rho:.3f}")
    print("=== Verdict ===")
    if rho > 0.3:
        print("Deficit INCREASES with S/N -> consistent with the sampler-geometry "
              "hypothesis (Taylor, Ashdown & Hobson arXiv:0708.2989). Strengthens the "
              "case for an MCLMC port under ROADMAP.md's decision rule 3(a).")
    elif rho < -0.3:
        print("Deficit DECREASES with S/N (worst at low S/N) -> consistent with the "
              "Wiener-suppressed-start / under-mixing hypothesis (Millea, Anderes & "
              "Wandelt arXiv:2002.00965). Same phenomenon as the lmax=128 pilot's "
              "slow l=[60,100) bin at a different scale -- no new compute justified; "
              "resolves under the coverage-pilot work already on the critical path.")
    else:
        print("No clear trend (|rho| <= 0.3) -> neither hypothesis is favored by this "
              "plot alone. Escalate per ROADMAP.md; do not write the deficit off as "
              "mixing in the draft.")

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(snr, 100 * deficit, c=l_mid, cmap="viridis", s=40)
    ax.set_xlabel("S/N (posterior mean / posterior std, per l-bin)")
    ax.set_ylabel("phi-power deficit (%)")
    ax.set_title(f"lmax=300, phi_n_lfs=80: per-l-bin deficit vs S/N (rho={rho:.2f})")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("l (bin midpoint)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved plot to {args.out}")


if __name__ == "__main__":
    main()
