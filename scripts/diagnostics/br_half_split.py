"""Monte Carlo noise of the Blackwell-Rao (A, n_s) posterior (scripts/br_lcdm.py, ROADMAP T1.4).

For each sky of an ensemble, the posterior from the first and the second half of its chain:
if the BR estimate were free of Monte Carlo noise the two would agree; their difference in
posterior-sd units, divided by sqrt(2), estimates the noise in one half-chain's mean, and
the full chain's is about 1/sqrt(2) of that. A calibration z spread of
sqrt(1 + noise^2) is then explained by it.

Usage: PYTHONPATH=diffcmb .venv/bin/python scripts/diagnostics/br_half_split.py ENSEMBLE_DIR
           --target lcdm|hier [--n_skies 48]
"""
import argparse
import glob
import os

import br_lcdm as br
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ensemble")
    ap.add_argument("--target", choices=("lcdm", "hier"), default="lcdm")
    ap.add_argument("--n_skies", type=int, default=48)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.ensemble, "chain_r[0-9][0-9][0-9].npz")))[: a.n_skies]
    ratio = None
    d_a, d_n = [], []
    for f in files:
        s, nu, fid, lmax = br.chain_s(f)
        if ratio is None:
            ratio = br.tilt_ratio(lmax, br.NS_GRID)
        k = br.dof(lmax)[br.LMIN:br.LMAX_USE]
        grid = br.model_grid(fid, ratio)
        tnu = nu if a.target == "hier" else None
        half = s.shape[0] // 2
        res = []
        for part in (s[:half], s[half:2 * half]):
            ll, _ = br.log_like_grid(part, k, grid, br.chain_log_marginal(part, k, nu=nu, cl_fid=fid), a.target, tnu)
            res.append(br.summarise(br.posterior(ll)))
        sd_a = 0.5 * (res[0]["A_sd"] + res[1]["A_sd"])
        sd_n = 0.5 * (res[0]["ns_sd"] + res[1]["ns_sd"])
        d_a.append((res[0]["A"] - res[1]["A"]) / sd_a)
        d_n.append((res[0]["ns"] - res[1]["ns"]) / sd_n)
        print(f"  {os.path.basename(f)}: half-to-half dA {d_a[-1]:+.2f} sd, dn_s {d_n[-1]:+.2f} sd")
    d_a, d_n = np.array(d_a), np.array(d_n)
    for name, x in (("A", d_a), ("n_s", d_n)):
        half_noise = np.sqrt(np.mean(x ** 2) / 2)
        full_noise = half_noise / np.sqrt(2)
        print(f"{name}: rms half-to-half {np.sqrt(np.mean(x ** 2)):.2f} sd -> MC noise of the full-chain "
              f"mean ~{full_noise:.2f} sd; implied calibration z sd ~{np.sqrt(1 + full_noise ** 2):.2f}")


if __name__ == "__main__":
    main()
