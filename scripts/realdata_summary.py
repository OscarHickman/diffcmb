"""Posterior spectra from the real-data chains, against Planck (ROADMAP T1.2a).

Reads the data_chain_rNNN.npz chains of one coverage_ensemble_chain.py --data_map
run (several chains on ONE data set: same map, same noise draw, different
starts; refused otherwise) and reports, per l-bin of strict_sbc_ranks.columns:

  * C_l^TT: the posterior median and 68 % interval of D_l = l(l+1)C_l/2pi,
    and the Planck 2018 TT spectrum's D_l (COM_PowerSpect_CMB-TT-full_R3.01)
    as a ratio to the posterior median. Both are estimates from the same sky
    on different masks and methods, so the ratio is a consistency read, not a
    calibrated test; the posterior C_l is unlensed, Planck's lensed (< 0.1 %
    apart below l = 64).
  * C_L^phiphi: the posterior median over the prior's centre (A_phi x the
    physical fiducial) and over the physical fiducial. On real data at
    A_phi = 3000 the sky (lensed at A_L = 1) should pull the first well below 1.
  * multi-chain rank-normalised split R-hat (Vehtari et al. 2021) and the
    summed bulk ESS of the bin-averaged ln C.

Usage:
  PYTHONPATH=diffcmb .venv/bin/python scripts/realdata_summary.py \
      --chains results/analysis/realdata_l64_A3000_n30 \
      --planck_tt /cosma8/data/dp004/dc-hick2/Plank/spectra/COM_PowerSpect_CMB-TT-full_R3.01.txt \
      --out results/analysis/realdata_l64_A3000_n30/summary
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_convergence import _rank_normalise, bulk_ess  # noqa: E402
from strict_sbc_ranks import columns  # noqa: E402

PLANCK_TT = "/cosma8/data/dp004/dc-hick2/Plank/spectra/COM_PowerSpect_CMB-TT-full_R3.01.txt"


def multichain_rhat(chains):
    """Rank-normalised split R-hat over a list of (n, k) chains: max of the
    bulk and folded versions, every chain split in half."""
    n = min(c.shape[0] for c in chains) // 2

    def rhat(z_all):
        seqs = np.stack([s for c in z_all for s in (c[:n], c[n:2 * n])])  # (2m, n, k)
        w = seqs.var(axis=1, ddof=1).mean(axis=0)
        b = n * seqs.mean(axis=1).var(axis=0, ddof=1)
        return np.sqrt(((n - 1) / n * w + b / n) / np.where(w > 0, w, np.inf))

    pooled = np.concatenate([c[:2 * n] for c in chains])
    split = np.cumsum([0] + [2 * n] * len(chains))
    z = _rank_normalise(pooled)
    zf = _rank_normalise(np.abs(pooled - np.median(pooled, axis=0)))
    bulk = rhat([z[a:b] for a, b in zip(split[:-1], split[1:])])
    folded = rhat([zf[a:b] for a, b in zip(split[:-1], split[1:])])
    return np.maximum(bulk, folded)


def load_chains(indir):
    """The chains of one real-data run, refused unless every one observed the
    same data map with the same noise draw."""
    files = sorted(glob.glob(os.path.join(indir, "data_chain_r[0-9][0-9][0-9].npz")))
    if not files:
        raise SystemExit(f"no data_chain_rNNN.npz in {indir}")
    ds = [np.load(f) for f in files]
    for f, d in zip(files, ds):
        if not str(d["data_map"]):
            raise SystemExit(f"{f}: not a --data_map chain")
        if (d["data_masked"].shape != ds[0]["data_masked"].shape
                or not np.array_equal(d["data_masked"], ds[0]["data_masked"])):
            raise SystemExit(f"{f}: observed different data from {files[0]}; chains of one "
                             "run must share the map and the noise draw (--noise_realization)")
    return files, ds


def read_planck_dl(path, lmax):
    """Planck D_l [muK^2] for l = 2..lmax-1 from a COM_PowerSpect TT table."""
    t = np.loadtxt(path)
    ell, dl = t[:, 0].astype(int), t[:, 1]
    want = np.arange(2, lmax)
    if not np.all(np.isin(want, ell)):
        raise SystemExit(f"{path} does not cover l = 2..{lmax - 1}")
    return dl[np.searchsorted(ell, want)]


def summarise(ds, planck_dl):
    lmax = int(ds[0]["lmax"])
    n = lmax - 2
    ell = np.arange(2, lmax)
    to_dl = np.log(ell * (ell + 1) / (2 * np.pi))
    fid = np.asarray(ds[0]["cl_phiphi_fid"], dtype=np.float64)[2:lmax]  # prior centre
    amp = float(ds[0]["phi_amplitude"])
    cols = columns(lmax)
    blocks = {"C_l^TT": [d["alm_samples"][:, :n] for d in ds],
              "C_L^phiphi": [d["cl_phiphi_samples"][:, :n] for d in ds]}
    out = {"columns": [c for c, _ in cols], "lmax": lmax, "n_chains": len(ds), "phi_amplitude": amp}
    for b, chains in blocks.items():
        binned = [np.stack([c[:, sl].mean(1) for _, sl in cols], axis=1) for c in chains]
        pooled = np.concatenate(binned)
        out[f"{b}/rhat"] = multichain_rhat(binned)
        out[f"{b}/ess"] = np.sum([bulk_ess(x) for x in binned], axis=0)
        q16, q50, q84 = np.percentile(pooled, [16, 50, 84], axis=0)
        if b == "C_l^TT":
            shift = np.array([to_dl[sl].mean() for _, sl in cols])
            out[f"{b}/dl_q16"], out[f"{b}/dl_q50"], out[f"{b}/dl_q84"] = (
                np.exp(q + shift) for q in (q16, q50, q84))
            planck = np.exp(np.array([np.log(planck_dl[sl]).mean() for _, sl in cols]))
            out[f"{b}/planck_dl"] = planck
            out[f"{b}/planck_over_median"] = planck / out[f"{b}/dl_q50"]
        else:
            ln_fid = np.array([np.log(fid[sl]).mean() for _, sl in cols])
            out[f"{b}/over_prior_q16"], out[f"{b}/over_prior_q50"], out[f"{b}/over_prior_q84"] = (
                np.exp(q - ln_fid) for q in (q16, q50, q84))
            out[f"{b}/over_physical_q50"] = np.exp(q50 - ln_fid) * amp
    return out


def report(s):
    print(f"{s['n_chains']} chains, lmax {s['lmax']}, A_phi {s['phi_amplitude']:g}\n")
    print(f"{'bin':<10}{'D_l^TT 16/50/84 [muK^2]':>28}{'Planck/median':>15}{'R-hat':>8}{'ESS':>8}")
    for i, c in enumerate(s["columns"]):
        print(f"{c:<10}{s['C_l^TT/dl_q16'][i]:>9.0f}{s['C_l^TT/dl_q50'][i]:>9.0f}{s['C_l^TT/dl_q84'][i]:>9.0f}"
              f"{s['C_l^TT/planck_over_median'][i]:>16.3f}{s['C_l^TT/rhat'][i]:>9.3f}{s['C_l^TT/ess'][i]:>8.0f}")
    print(f"\n{'bin':<10}{'C_L^pp / prior centre 16/50/84':>32}{'/ physical':>12}{'R-hat':>8}{'ESS':>8}")
    for i, c in enumerate(s["columns"]):
        print(f"{c:<10}{s['C_L^phiphi/over_prior_q16'][i]:>11.3g}{s['C_L^phiphi/over_prior_q50'][i]:>10.3g}"
              f"{s['C_L^phiphi/over_prior_q84'][i]:>10.3g}{s['C_L^phiphi/over_physical_q50'][i]:>13.3g}"
              f"{s['C_L^phiphi/rhat'][i]:>9.3f}{s['C_L^phiphi/ess'][i]:>8.0f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chains", required=True)
    p.add_argument("--planck_tt", default=PLANCK_TT)
    p.add_argument("--out", required=True, help="output prefix (.npz written)")
    args = p.parse_args()
    files, ds = load_chains(args.chains)
    s = summarise(ds, read_planck_dl(args.planck_tt, int(ds[0]["lmax"])))
    report(s)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out + ".npz", files=np.array(files), **{k: np.asarray(v) for k, v in s.items()})
    print(f"\nsaved {args.out}.npz")


if __name__ == "__main__":
    main()
