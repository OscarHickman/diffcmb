"""Paper figure 2 -- the headline effect: C_l^TT lensing-bias reduction.

The paper's single most important figure (ROADMAP.md S1): the lensing-aware
joint sampler removes lensing bias from C_l^TT that a lensing-blind
(Commander-style) Gibbs fit leaves in, and the effect GROWS with l. Two
standalone panel PDFs.

  (a) cl_comparison_single.pdf -- one realization (seed 0): fractional bias
                                   per l-bin, blind vs aware, against the
                                   expected posterior mean of a correct
                                   unlensed model (see compare_cl_bias_reduction.py
                                   docstring for why that reference, not the
                                   realized or fiducial spectrum).
  (b) bias_reduction_all_seeds.pdf -- all 4 independent skies overlaid: aware
                                   beats blind 4/4, blind deficit deepens
                                   monotonically with l 4/4 (job 11991514,
                                   93.7% +/- 1.8% mean reduction).

Reads bias_seed{0,1,2,3}.npz (written by compare_cl_bias_reduction.py, one
per sky) and bias_reduction_seeds.npz (the aggregator's summary) -- no new
computation, this script only re-renders that validated data as standalone
paper panels instead of the analysis script's combined diagnostic PNG.

Usage:
  PYTHONPATH=diffcmb .venv/bin/python scripts/paper/fig2_bias_reduction.py \
      --figdir results/analysis/figures \
      --outdir /cosma/apps/durham/dc-hick2/papers/7_DiffCMB/plots/figure2
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

UNRELIABLE = {(2, 10)}  # phi does not equilibrate here at lmax=128 (see compare_cl_bias_reduction.py)


def plot_single_realization(seed_npz, outpath):
    d = np.load(seed_npz)
    bins = d["bins"]
    rows = [i for i, (lo, hi) in enumerate(bins) if (lo, hi) not in UNRELIABLE]

    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    x = np.arange(len(rows))
    w = 0.38
    for key, lab, col, off in (
            ("blind_mean", "lensing-blind (Commander-style)", "#c44e52", -w / 2),
            ("aware_mean", "lensing-aware (joint sampler)", "#4c72b0", +w / 2)):
        vals = np.array([d[key][i] / d["truth"][i] - 1.0 for i in rows])
        sem_key = key.replace("mean", "sem")
        errs = np.array([d[sem_key][i] / d["truth"][i] for i in rows])
        ax.bar(x + off, vals * 100, w, yerr=errs * 100, label=lab, color=col, capsize=3)
    ax.axhline(0.0, color="k", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"[{bins[i][0]},{bins[i][1]})" for i in rows])
    ax.set_xlabel(r"$\ell$ bin")
    ax.set_ylabel(r"fractional bias in $C_\ell^{TT}$ (%)"
                 "\n(posterior $-$ correct-model expectation)/expectation")
    ax.set_title("One realization (seed 0), $\\ell_{\\max}$=128", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    print(f"wrote {outpath}")


def plot_all_seeds(seed_npzs, outpath):
    runs = []
    for f in seed_npzs:
        d = np.load(f)
        runs.append((os.path.basename(f).replace("bias_seed", "seed").replace(".npz", ""), d))
    bins = runs[0][1]["bins"]
    rows = [i for i, (lo, hi) in enumerate(bins) if (lo, hi) not in UNRELIABLE]
    ell_mid = [(bins[i][0] + bins[i][1]) / 2.0 for i in rows]

    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(runs)))
    for (name, d), col in zip(runs, colors):
        blind = [(d["blind_mean"][i] / d["truth"][i] - 1.0) * 100 for i in rows]
        aware = [(d["aware_mean"][i] / d["truth"][i] - 1.0) * 100 for i in rows]
        ax.plot(ell_mid, blind, "o--", color=col, alpha=0.85, ms=4,
                label=f"{name}, blind")
        ax.plot(ell_mid, aware, "s-", color=col, alpha=0.85, ms=4,
                label=f"{name}, aware")
    ax.axhline(0.0, color="k", lw=1)
    ax.set_xlabel(r"$\ell$ (bin midpoint)")
    ax.set_ylabel(r"fractional bias in $C_\ell^{TT}$ (%)")
    ax.set_title(f"{len(runs)} independent skies -- "
                 "aware beats blind on every sky, blind deficit deepens with $\\ell$",
                 fontsize=10)
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    print(f"wrote {outpath}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figdir", default="results/analysis/figures")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument(
        "--outdir",
        default="/cosma/apps/durham/dc-hick2/papers/7_DiffCMB/plots/figure2")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    seed_npzs = []
    for s in args.seeds:
        f = os.path.join(args.figdir, f"bias_seed{s}.npz")
        if os.path.exists(f):
            seed_npzs.append(f)
        else:
            print(f"  ! missing, skipped: {f}")
    if not seed_npzs:
        raise SystemExit("no per-seed bias files found")

    plot_single_realization(seed_npzs[0],
                            os.path.join(args.outdir, "cl_comparison_single.pdf"))
    plot_all_seeds(seed_npzs,
                  os.path.join(args.outdir, "bias_reduction_all_seeds.pdf"))
    if len(seed_npzs) < 4:
        print(f"\n  NOTE: only {len(seed_npzs)}/4 seeds present -- rerun panel (b) "
              "when the rest land.")


if __name__ == "__main__":
    main()
