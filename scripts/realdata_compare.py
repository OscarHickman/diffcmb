"""Posterior C_l^TT from two Planck maps, side by side (ROADMAP T1.2a).

Reads two realdata_summary.py outputs from runs that differ only in the map
(same lmax, A_phi and bins; refused otherwise) and reports, per l-bin and per
multipole, the ratio of the posterior medians of D_l and the shift of B's
median in units of A's posterior sd (half the 68 % width in ln D_l).

The two maps observe the same sky, so the posteriors are strongly correlated
and the shift is not a significance: it says how large the map difference is
against the width the paper quotes. SEVEM 2018 vs PR4 is the release check
(the maps differ by <= 1.7 % in power, so a null); SMICA vs SEVEM is the
component-separation systematic.

Usage:
  .venv/bin/python scripts/realdata_compare.py \
      --a results/analysis/realdata_sevemR3_l64_A1_n30_clnu30/summary.npz \
      --b results/analysis/realdata_sevemR4_l64_A1_n30_clnu30/summary.npz
"""

import argparse

import numpy as np

BULK_FROM_ELL = 10  # below this the posterior is prior-dominated at nu = 30


def _ln_sd(s, key):
    return 0.5 * (np.log(s[f"{key}_q84"]) - np.log(s[f"{key}_q16"]))


def compare(a, b):
    for k in ("lmax", "phi_amplitude", "columns"):
        if not np.array_equal(a[k], b[k]):
            raise SystemExit(f"the runs differ in {k} ({a[k]} vs {b[k]}): not a map comparison")
    out = {"columns": a["columns"], "ell": a["ell"]}
    for key, tag in (("C_l^TT/dl", "bin"), ("C_l^TT/dl_ell", "ell")):
        ln_a, ln_b = np.log(a[f"{key}_q50"]), np.log(b[f"{key}_q50"])
        out[f"{tag}/ratio"] = np.exp(ln_b - ln_a)
        out[f"{tag}/shift_sd"] = (ln_b - ln_a) / _ln_sd(a, key)
    bulk = out["ell"] >= BULK_FROM_ELL
    out["ell/max_abs_shift_bulk"] = np.abs(out["ell/shift_sd"][bulk]).max()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="reference summary.npz")
    p.add_argument("--b", required=True, help="summary.npz compared against it")
    args = p.parse_args()
    a, b = np.load(args.a), np.load(args.b)
    c = compare(a, b)
    print(f"B = {args.b}\nA = {args.a}\n")
    print(f"{'bin':<10}{'D_l B/A':>10}{'shift [sd_A]':>14}")
    for i, col in enumerate(c["columns"]):
        print(f"{col:<10}{c['bin/ratio'][i]:>10.3f}{c['bin/shift_sd'][i]:>14.2f}")
    worst = np.argsort(-np.abs(c["ell/shift_sd"]))[:5]
    print("\nlargest per-l shifts: " + ", ".join(
        f"l={c['ell'][i]} {c['ell/ratio'][i]:.3f} ({c['ell/shift_sd'][i]:+.2f} sd)" for i in worst))
    print(f"max |shift| over l >= {BULK_FROM_ELL}: {c['ell/max_abs_shift_bulk']:.2f} sd_A")


if __name__ == "__main__":
    main()
