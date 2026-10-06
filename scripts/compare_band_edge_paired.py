"""ROADMAP T0.1 a (2026-10-07): band-edge C_l, paired by sky across ensembles.

Every T0.1 run since the `--cl_init data` pilot shares skies r032-r039 with the
production control (`_nu30_long_r032_r039`) and the exact dense reference
(`exact_dense_alm_reference_l64_nu30`, phi fixed at truth). With 8 skies the
only usable comparison is PAIRED by sky. This script does it from the saved
chains instead of by hand:

  * per directory, posterior C_l / (S_true/(k-4)) at l = lmax-4..lmax-1 by
    chain quarter (`diagnose_alm_band_edge.cl_ratio_per_chain`), mean +- SE
    over skies;
  * per run, the paired difference to the reference (run's last quarter, and
    all sweeps, minus the reference's all-sweep mean) with its SE and z;
  * with several runs, each run's last quarter paired against the first run's.

It refuses to pair chains whose `alm_true_packed` differ: same realization
index is not enough (the nu = 6 and nu = 30 skies share index but not phi).

Usage (T0.1 a harvest):
  PYTHONPATH=diffcmb .venv/bin/python scripts/compare_band_edge_paired.py \
      --ref results/analysis/exact_dense_alm_reference_l64_nu30 \
      --indir results/analysis/ens_exact_l64_A3000_n30_nu30_long_r032_r039 \
      --indir results/analysis/ens_exact_l64_A3000_n30_nu30_phifixed --exclude 36
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper"))

from diagnose_alm_band_edge import N_QUARTERS, cl_ratio_per_chain  # noqa: E402
from diagnose_calibration_stationarity import chain_files  # noqa: E402

_REAL = re.compile(r"chain_r(\d{3})\.npz$")


def by_realization(indir: str) -> dict[int, str]:
    """{realization: chain file} for one ensemble directory."""
    return {int(_REAL.search(f).group(1)): f for f in chain_files(indir)}


def _mean_se(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard error over axis 0 (skies)."""
    return x.mean(axis=0), x.std(axis=0, ddof=1) / np.sqrt(x.shape[0])


def _check_same_sky(ref_file: str, other_file: str, r: int) -> None:
    a = np.load(ref_file, allow_pickle=True)["alm_true_packed"]
    b = np.load(other_file, allow_pickle=True)["alm_true_packed"]
    if a.shape != b.shape or not np.array_equal(a, b):
        raise SystemExit(f"realization {r}: {other_file} is not the same sky as {ref_file}")


def compare(ref: str, indirs: list[str], exclude: tuple[int, ...] = ()) -> dict:
    """Paired band-edge comparison of `indirs` against the reference `ref`.

    Returns {"skies", "lmax", "ref_ratio" (n_sky, lmax), "ratio" {dir: (n_sky,
    n_seg, lmax)}, "paired" {dir: {"diff", "se", "diff_all", "se_all"}}}, the
    differences being the run's last quarter (or all sweeps) minus the
    reference's all-sweep ratio, mean over skies.
    """
    maps = {d: by_realization(d) for d in [ref, *indirs]}
    skies = sorted(set.intersection(*(set(m) for m in maps.values())) - set(exclude))
    if len(skies) < 2:
        raise SystemExit(f"fewer than 2 common skies across {list(maps)}")
    lmax = int(np.load(maps[ref][skies[0]], allow_pickle=True)["lmax"])
    for d in indirs:
        for r in skies:
            _check_same_sky(maps[ref][r], maps[d][r], r)
    ref_ratio = np.stack([cl_ratio_per_chain(maps[ref][r], lmax, 1)[0] for r in skies])
    ratio, paired = {}, {}
    for d in indirs:
        ratio[d] = np.stack([cl_ratio_per_chain(maps[d][r], lmax, N_QUARTERS) for r in skies])
        diff, se = _mean_se(ratio[d][:, -1] - ref_ratio)
        diff_all, se_all = _mean_se(ratio[d].mean(axis=1) - ref_ratio)
        paired[d] = {"diff": diff, "se": se, "diff_all": diff_all, "se_all": se_all}
    return {"skies": skies, "lmax": lmax, "ref_ratio": ref_ratio, "ratio": ratio,
            "paired": paired}


def _z(diff: float, se: float) -> str:
    return f"{diff:+.3f} +- {se:.3f} (z {diff / se:+.2f})" if se > 0 else f"{diff:+.3f}"


def report(res: dict, ref: str, indirs: list[str]) -> None:
    lmax, skies = res["lmax"], res["skies"]
    ells = range(lmax - 4, lmax)
    print(f"skies ({len(skies)}): " + ", ".join(f"r{r:03d}" for r in skies))
    m, s = _mean_se(res["ref_ratio"])
    print(f"\nreference {ref}: C_l/(S_true/(k-4)), all sweeps, mean +- SE over skies")
    for L in ells:
        print(f"  l = {L}:  {m[L]:.3f} +- {s[L]:.3f}")
    for d in indirs:
        r = res["ratio"][d]
        print(f"\n{d}: by chain quarter (mean over skies)")
        for L in ells:
            print(f"  l = {L}:  " + "  ".join(f"{r[:, q, L].mean():.3f}" for q in range(r.shape[1])))
        p = res["paired"][d]
        print("  paired vs reference: last quarter | all sweeps")
        for L in ells:
            print(f"  l = {L}:  {_z(p['diff'][L], p['se'][L])}   |   "
                  f"{_z(p['diff_all'][L], p['se_all'][L])}")
    if len(indirs) > 1:
        base = indirs[0]
        for d in indirs[1:]:
            print(f"\npaired last quarter, {d} minus {base}")
            for L in ells:
                diff, se = _mean_se(res["ratio"][d][:, -1, L] - res["ratio"][base][:, -1, L])
                print(f"  l = {L}:  {_z(float(diff), float(se))}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="reference ensemble (the dense exact one)")
    ap.add_argument("--indir", action="append", required=True)
    ap.add_argument("--exclude", type=int, action="append", default=[],
                    help="realization to leave out (e.g. 36, the collapsed sky)")
    ap.add_argument("--out", default=None, help="optional .npz for the numbers")
    args = ap.parse_args()
    res = compare(args.ref, args.indir, tuple(args.exclude))
    report(res, args.ref, args.indir)
    if args.out:
        flat = {"skies": np.asarray(res["skies"]), "ref_ratio": res["ref_ratio"]}
        for i, d in enumerate(args.indir):
            flat[f"ratio_{i}"] = res["ratio"][d]
            flat.update({f"{k}_{i}": v for k, v in res["paired"][d].items()})
        np.savez(args.out, indirs=np.asarray(args.indir), **flat)
        print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
