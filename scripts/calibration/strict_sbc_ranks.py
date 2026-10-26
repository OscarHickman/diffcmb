"""Strict SBC of the spectra in a strict exact ensemble (ROADMAP T1.1, T0.1 c).

In strict mode (--cl_prior_nu with --cl_phiphi_prior_nu) both spectra carry
proper conjugate priors and each truth is drawn from its prior, so the rank of
the drawn truth among the posterior draws is a strict SBC statistic, unlike the
flat-prior ensembles, where aggregate_coverage_ranks can only give coverage
against the realized power. Per spectrum and l-bin (the shared ELL_BINS, plus
the single band-limit multipole l = lmax-1), the rank of the bin-averaged ln C
truth among the thinned, bin-averaged draws; uniformity per bin by the
simulation-calibrated discrete test (calibrated at the granularity it is read,
never pooled across bins of one chain).

Also, for T0.1 c: the C_l collapse table. A (chain, l) is collapsed in a chain
quarter when the median C_l draw there is below COLLAPSE_RATIO of the realized
truth power S_l/k_l (r036's C_62 sat at ~2e-5 of it).

The median bulk ESS per block is printed beside the thinning, so the reader can
check the thinned draws are close to independent.
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/
import _paths  # noqa: E402,F401  (puts every script folder on sys.path)
from aggregate_coverage_ranks import (  # noqa: E402
    ELL_BINS,
    _sl_from_packed,
    discrete_uniform_p,
    rank_spread,
    realized_spectrum,
)
from fig_convergence import bulk_ess  # noqa: E402

COLLAPSE_RATIO = 0.01
BLOCKS = ("C_l^TT", "C_L^phiphi")


def columns(lmax):
    """(label, l-index slice into an (n, lmax-2) ln C array) per rank column."""
    cols = [(f"[{lo},{min(hi, lmax)})", slice(lo - 2, min(hi, lmax) - 2))
            for lo, hi in ELL_BINS if lo < min(hi, lmax)]
    return cols + [(f"l={lmax - 1}", slice(lmax - 3, lmax - 2))]


def ln_spectra(d, lmax):
    """Draws and truths of (ln C_l^TT, ln C_L^phiphi), l = 2..lmax-1. Both draw
    arrays are stored as logs (ln C_l leads alm_samples; cl_phiphi_samples is
    the log-spectrum)."""
    n = lmax - 2
    return ({"C_l^TT": d["alm_samples"][:, :n], "C_L^phiphi": d["cl_phiphi_samples"][:, :n]},
            {"C_l^TT": np.log(d["cl_true"][2:lmax]), "C_L^phiphi": np.log(d["cl_phiphi_true"][2:lmax])})


def _check_strict(d, path):
    for k in ("cl_prior_nu", "cl_phiphi_prior_nu"):
        if k not in d.files or not np.isfinite(float(d[k])):
            raise SystemExit(f"{path}: {k} is not set -- not a strict-mode chain, so its "
                             "spectrum ranks are not strict SBC statistics")
    if "cl_phiphi_samples" not in d.files:
        raise SystemExit(f"{path}: Block 4 was off")


def strict_ranks(files, thin, burn_frac):
    lmax, n_samp, cols = None, None, None
    ranks = {b: [] for b in BLOCKS}
    ess = {b: [] for b in BLOCKS}
    reals = []
    for f in files:
        d = np.load(f)
        _check_strict(d, f)
        if lmax is None:
            lmax, n_samp = int(d["lmax"]), d["alm_samples"].shape[0]
            cols = columns(lmax)
        elif int(d["lmax"]) != lmax or d["alm_samples"].shape[0] != n_samp:
            raise SystemExit(f"{f}: lmax or chain length differs from the first chain")
        draws, truth = ln_spectra(d, lmax)
        keep = slice(int(burn_frac * n_samp), None, thin)
        for b in BLOCKS:
            binned = np.stack([draws[b][:, sl].mean(1) for _, sl in cols], axis=1)
            t = np.array([truth[b][sl].mean() for _, sl in cols])
            ranks[b].append(np.sum(binned[keep] < t[None, :], axis=0))
            ess[b].append(bulk_ess(binned[int(burn_frac * n_samp):]))
        reals.append(int(d["realization"]))
    n_draws = len(range(int(burn_frac * n_samp), n_samp, thin))
    ranks = {b: np.array(v) for b, v in ranks.items()}
    return {"ranks": ranks, "n_draws": n_draws, "columns": [c for c, _ in cols],
            "p": {b: np.array([discrete_uniform_p(r, n_draws) for r in ranks[b].T]) for b in BLOCKS},
            "spread": {b: np.array([rank_spread(r, n_draws) for r in ranks[b].T]) for b in BLOCKS},
            "median_ess": {b: np.median(np.array(ess[b]), axis=0) for b in BLOCKS},
            "realization": np.array(reals), "lmax": lmax, "n_samples": n_samp}


def collapse_table(files):
    """Chains and multipoles whose median C_l draw in some quarter is below
    COLLAPSE_RATIO of the realized truth power; quarters are 1-indexed."""
    rows = []
    for f in files:
        d = np.load(f)
        lmax = int(d["lmax"])
        realized = realized_spectrum(_sl_from_packed(d["alm_true_packed"], lmax), lmax)[2:lmax]
        ratio = np.exp(d["alm_samples"][:, : lmax - 2]) / realized[None, :]
        quarters = np.array_split(ratio, 4)
        med = np.stack([np.median(q, axis=0) for q in quarters])  # (4, lmax-2)
        for i in np.where((med < COLLAPSE_RATIO).any(axis=0))[0]:
            rows.append({"realization": int(d["realization"]), "ell": int(i + 2),
                         "quarters_collapsed": [int(q + 1) for q in np.where(med[:, i] < COLLAPSE_RATIO)[0]],
                         "quarter_medians": med[:, i].tolist()})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indir", required=True)
    p.add_argument("--thin", type=int, default=20)
    p.add_argument("--burn_frac", type=float, default=0.0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.indir, "chain_r*.npz")))
    files = [f for f in files if not f.endswith("_ckpt.npz")]
    if not files:
        raise SystemExit(f"no chains in {args.indir}")
    res = strict_ranks(files, args.thin, args.burn_frac)
    nd = res["n_draws"]
    print(f"Strict SBC on {len(files)} chains, {res['n_samples']} sweeps, burn {args.burn_frac}, "
          f"thin {args.thin} -> {nd} draws (uniform: mean rank {nd / 2:.1f}, spread {1 / np.sqrt(12):.4f})")
    print(f"{'spectrum':>11} {'bin':>9} {'mean rank':>10} {'spread':>8} {'p':>8} {'median ESS':>11} {'sweeps/ESS':>11}")
    for b in BLOCKS:
        for j, c in enumerate(res["columns"]):
            r = res["ranks"][b][:, j]
            e = res["median_ess"][b][j]
            flag = "  <- T0.1 band edge" if c.startswith("[60") or c == f"l={res['lmax'] - 1}" else ""
            print(f"{b:>11} {c:>9} {r.mean():>10.2f} {res['spread'][b][j]:>8.4f} {res['p'][b][j]:>8.4f} "
                  f"{e:>11.0f} {res['n_samples'] * (1 - args.burn_frac) / e:>11.1f}{flag}")
    rows = collapse_table(files)
    print(f"\nC_l collapse (quarter median < {COLLAPSE_RATIO} x realized power): "
          f"{len(rows)} (chain, l) pairs" + ("" if rows else " -- none"))
    for row in rows:
        print(f"  r{row['realization']:03d} l={row['ell']}: quarters {row['quarters_collapsed']}, "
              f"medians {np.round(row['quarter_medians'], 4).tolist()}")
    np.savez(args.out, n_draws=nd, thin=args.thin, burn_frac=args.burn_frac,
             columns=np.array(res["columns"]), realization=res["realization"],
             **{f"ranks_{b}": res["ranks"][b] for b in BLOCKS},
             **{f"p_{b}": res["p"][b] for b in BLOCKS},
             **{f"median_ess_{b}": res["median_ess"][b] for b in BLOCKS},
             collapse_realization=np.array([r["realization"] for r in rows], dtype=int),
             collapse_ell=np.array([r["ell"] for r in rows], dtype=int))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
