"""Across-sky summary of the C_l^TT bias-reduction result.

The single-sky result (seed 0) is the project's strongest number: mean
|fractional bias| 0.0226 -> 0.0016 over the four reliable l bins, with the
lensing-aware posterior consistent with unbiased everywhere and the
lensing-blind deficit growing monotonically to -5.4% at [100,128). "Is that one
realization?" is the first question a referee asks of a single-sim result, and
until this script runs the honest answer is yes.

Each sky has its own phi realization, so the SIZE of the blind deficit is
expected to vary between seeds. The claim under test is therefore not that the
numbers repeat, but that the SIGN and ORDER OF MAGNITUDE do: blind biased low
and increasingly so with l, aware consistent with unbiased. This reports the
per-seed values and their across-seed spread so that claim can be read directly.

Reference for "unbiased" is the expected posterior mean S_l/(k_l-4), NOT the
realized power and NOT the fiducial -- see compare_cl_bias_reduction.py's
docstring; both wrong references reverse the conclusion.

Usage (after running compare_cl_bias_reduction.py per seed):
  PYTHONPATH=diffcmb .venv/bin/python scripts/aggregate_bias_reduction_seeds.py \
      --inputs results/analysis/figures/bias_seed0.npz ...
"""

import argparse
import os

import numpy as np

UNRELIABLE = {(2, 10)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="per-seed .npz written by compare_cl_bias_reduction.py")
    ap.add_argument("--out", default="results/analysis/figures/bias_reduction_seeds")
    args = ap.parse_args()

    runs = []
    for f in args.inputs:
        if not os.path.exists(f):
            print(f"  ! missing, skipped: {f}")
            continue
        runs.append((os.path.basename(f).replace(".npz", ""), np.load(f)))
    if not runs:
        raise SystemExit("no inputs found")

    bins = runs[0][1]["bins"]
    rel = [i for i, (lo, hi) in enumerate(bins) if (lo, hi) not in UNRELIABLE]

    print(f"C_l^TT bias reduction across {len(runs)} independent skies\n")
    print("Per-bin fractional bias (posterior - correct-model expectation)/expectation")
    print("  negative = power deficit; blind should go increasingly negative with l\n")
    hdr = "  " + "bin".ljust(12) + "".join(f"{n[:14]:>16s}" for n, _ in runs)
    for tag, key in (("LENSING-BLIND", "blind_mean"), ("LENSING-AWARE", "aware_mean")):
        print(f"{tag}")
        print(hdr)
        for i in rel:
            lo, hi = bins[i]
            cells = "".join(
                f"{(d[key][i] / d['truth'][i] - 1.0) * 100:>15.2f}%" for _, d in runs
            )
            print(f"  [{lo:>3},{hi:>4})".ljust(14) + cells)
        print()

    print("Headline: mean |fractional bias| over the reliable bins")
    print(f"  {'seed':<18s} {'blind':>10s} {'aware':>10s} {'reduction':>11s}")
    blinds, awares, reds = [], [], []
    for name, d in runs:
        b = float(np.mean([abs(d["blind_mean"][i] / d["truth"][i] - 1.0) for i in rel]))
        a = float(np.mean([abs(d["aware_mean"][i] / d["truth"][i] - 1.0) for i in rel]))
        blinds.append(b)
        awares.append(a)
        reds.append(1 - a / b)
        print(f"  {name:<18s} {b:>10.4f} {a:>10.4f} {1 - a / b:>10.1%}")

    blinds, awares, reds = map(np.array, (blinds, awares, reds))
    n = len(runs)
    print(f"\n  {'ACROSS SKIES':<18s} {blinds.mean():>10.4f} {awares.mean():>10.4f} "
          f"{reds.mean():>10.1%}")
    if n > 1:
        print(f"  {'  sd':<18s} {np.std(blinds, ddof=1):>10.4f} "
              f"{np.std(awares, ddof=1):>10.4f} {np.std(reds, ddof=1):>10.1%}")
        def sem(x):
            return float(np.std(x, ddof=1) / np.sqrt(n))
        print(f"  {'  sem':<18s} {sem(blinds):>10.4f} {sem(awares):>10.4f} "
              f"{sem(reds):>10.1%}")

    # The claim: does every sky show blind worse than aware, and does the blind
    # deficit grow with l? Report both as explicit pass/fail, not as prose.
    all_better = bool(np.all(awares < blinds))
    monotone = []
    for _, d in runs:
        vals = [d["blind_mean"][i] / d["truth"][i] - 1.0 for i in rel]
        monotone.append(bool(np.all(np.diff(vals) <= 0)))
    print(f"\n  aware beats blind on every sky: {all_better}  ({int(np.sum(awares < blinds))}/{n})")
    print(f"  blind deficit monotonically deepens with l: "
          f"{int(np.sum(monotone))}/{n} skies")
    if n < 4:
        print(f"\n  NOTE: only {n} sky/skies present -- rerun when the rest land.")

    np.savez(args.out + ".npz", blinds=blinds, awares=awares, reductions=reds,
             names=np.array([n for n, _ in runs]), bins=bins)
    print(f"\nSaved {args.out}.npz")


if __name__ == "__main__":
    main()
