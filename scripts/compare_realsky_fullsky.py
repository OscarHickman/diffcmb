"""Paired comparison of a real-sky strict ensemble with the full-sky one (ROADMAP T1.2a).

The same realization index draws the same truth in both ensembles, so each sky
is compared with itself: per spectrum and l-bin (strict_sbc_ranks.columns), the
posterior width ratio real/full, the z of the posterior mean against the truth
in each, the drift of the mean between chain halves (in posterior sd), and the
bulk ESS. A masked sky should be wider by about f_sky^-1/2 (1.15 at the Planck
common mask's 0.75) and no more biased; a larger z or drift than the full sky's
is what the pilot gate looks for. Pairs whose truths differ are refused.

Usage:
  PYTHONPATH=diffcmb .venv/bin/python scripts/compare_realsky_fullsky.py \
      --real results/analysis/ens_realsky_l64_A3000_n30_nu30_strict \
      --full results/analysis/ens_exact_l64_A3000_n30_nu30_strict
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_convergence import bulk_ess  # noqa: E402
from strict_sbc_ranks import BLOCKS, columns, ln_spectra  # noqa: E402

STATS = ("mean", "sd", "z", "drift", "ess")


def summarise(d):
    """Per block, (n_cols,) arrays of the posterior mean, sd, z against the
    truth, half-to-half drift in sd and bulk ESS of the bin-averaged ln C."""
    lmax = int(d["lmax"])
    cols = columns(lmax)
    draws, truth = ln_spectra(d, lmax)
    out = {}
    for b in BLOCKS:
        binned = np.stack([draws[b][:, sl].mean(1) for _, sl in cols], axis=1)
        t = np.array([truth[b][sl].mean() for _, sl in cols])
        half = binned.shape[0] // 2
        sd = binned.std(0)
        out[b] = {"mean": binned.mean(0), "sd": sd, "z": (binned.mean(0) - t) / sd,
                  "drift": (binned[half:].mean(0) - binned[:half].mean(0)) / sd,
                  "ess": bulk_ess(binned)}
    return [c for c, _ in cols], out


def pairs(real_dir, full_dir):
    for f in sorted(glob.glob(os.path.join(real_dir, "chain_r[0-9][0-9][0-9].npz"))):
        g = os.path.join(full_dir, os.path.basename(f))
        if os.path.exists(g):
            yield f, g


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real", required=True)
    p.add_argument("--full", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    res = {b: {k: [] for k in STATS} for b in BLOCKS}
    full = {b: {k: [] for k in STATS} for b in BLOCKS}
    reals, labels = [], None
    for f, g in pairs(args.real, args.full):
        dr, df = np.load(f), np.load(g)
        for k in ("cl_true", "cl_phiphi_true", "phi_true_packed"):
            if not np.array_equal(dr[k], df[k]):
                raise SystemExit(f"{f} and {g}: {k} differs -- not the same sky")
        labels, sr = summarise(dr)
        _, sf = summarise(df)
        for b in BLOCKS:
            for k in STATS:
                res[b][k].append(sr[b][k])
                full[b][k].append(sf[b][k])
        reals.append(int(dr["realization"]))
    if not reals:
        raise SystemExit(f"no realization is in both {args.real} and {args.full}")

    print(f"Real-sky vs full-sky strict ensembles, {len(reals)} paired skies\n")
    saved = {"realization": np.array(reals), "columns": np.array(labels)}
    for b in BLOCKS:
        r = {k: np.array(v) for k, v in res[b].items()}
        u = {k: np.array(v) for k, v in full[b].items()}
        ratio = r["sd"] / u["sd"]
        print(f"  {b}  (medians over skies; rms z and rms drift)")
        print(f"    {'bin':>9} {'sd ratio':>8} {'rms z real':>10} {'rms z full':>10} "
              f"{'drift real':>10} {'drift full':>10} {'ESS real':>8} {'ESS full':>8}")
        for i, lab in enumerate(labels):
            print(f"    {lab:>9} {np.median(ratio[:, i]):8.2f} "
                  f"{np.sqrt(np.mean(r['z'][:, i] ** 2)):10.2f} "
                  f"{np.sqrt(np.mean(u['z'][:, i] ** 2)):10.2f} "
                  f"{np.sqrt(np.mean(r['drift'][:, i] ** 2)):10.2f} "
                  f"{np.sqrt(np.mean(u['drift'][:, i] ** 2)):10.2f} "
                  f"{np.median(r['ess'][:, i]):8.0f} {np.median(u['ess'][:, i]):8.0f}")
        tag = b.replace("^", "").replace("_", "")
        for k in STATS:
            saved[f"{tag}_{k}_real"], saved[f"{tag}_{k}_full"] = r[k], u[k]
    out = args.out or os.path.join(args.real, "compare_fullsky.npz")
    np.savez(out, **saved)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
