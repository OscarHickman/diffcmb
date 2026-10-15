"""Learned vs exact posterior, sky by sky (ROADMAP T1.1, figures 2 and 3).

For each sky of the strict exact ensemble (--cl_prior_nu with
--cl_phiphi_prior_nu), rebuild the sky's summaries with npe_simulate (the same
sky bit for bit, checked against the chain's saved truth), draw from the learned
posterior, and compare bin-averaged ln C against the exact chain:
  * shift = (learned mean - exact mean) / exact sd, with the exact chain's Monte
    Carlo error 1/sqrt(ESS) in the same units;
  * width ratio = learned sd / exact sd;
  * the correlation matrix of the bin-averaged (ln C_l^TT, ln C_L^phiphi), learned
    vs exact (the figure-4 analogue; Doeser & Jasche 2606.10023).
Refuses chains whose configuration differs from the flow's training simulator,
and a flow whose training summaries the current simulator no longer reproduces.
The [60,64) bin is the T0.1 band edge: reported, flagged, never pooled.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper"))
import npe_simulate as ns  # noqa: E402
from fig_convergence import bulk_ess  # noqa: E402
from npe_compress import load_conditioning  # noqa: E402
from npe_flow import ConditionalFlow  # noqa: E402
from npe_sbc import bin_slices  # noqa: E402

BAND_EDGE_LO = 60


def exact_theta(d, lmax):
    """(n, 2(lmax-2)) exact draws of (ln C_l^TT, ln C_L^phiphi), l = 2..lmax-1.
    Both are stored as logs: ln C_l leads alm_samples, and cl_phiphi_samples is
    the log-spectrum (aggregate_coverage_ranks exponentiates it)."""
    return np.concatenate([d["alm_samples"][:, : lmax - 2],
                           d["cl_phiphi_samples"][:, : lmax - 2]], axis=1)


def check_chain(d, cfg, path):
    want = {"lmax": cfg["lmax"], "nside": cfg["nside"], "noisesig": cfg["noisesig"],
            "phi_amplitude": cfg["phi_amplitude"], "cl_prior_nu": cfg["nu_tt"],
            "cl_phiphi_prior_nu": cfg["nu_pp"]}
    bad = [f"{k}: chain {d[k].item()!r}, flow {v!r}" for k, v in want.items()
           if not np.isclose(float(d[k]), float(v))]
    for k in ("lensing_operator", "fiducial"):
        if str(d[k]) != cfg[k]:
            bad.append(f"{k}: chain {str(d[k])!r}, flow {cfg[k]!r}")
    for k in ("start_truth", "phi_fixed_truth"):
        if k in d.files and bool(d[k]):
            bad.append(f"{k} is set: not a production chain")
    if "cl_phiphi_samples" not in d.files:
        bad.append("Block 4 off")
    if bad:
        raise SystemExit(f"{path} does not match the flow's simulator:\n  " + "\n  ".join(bad))


def check_summaries(ctx, cfg):
    """Refuse a flow if npe_simulate no longer reproduces its first training sim's
    summaries: a summary-code change after training feeds the flow inputs it never
    saw and biases it, while the held-out SBC (stored summaries) cannot notice
    (the 2026-10-15 v1 QE regression)."""
    sims_dir = cfg.get("train_args", {}).get("sims_dir")
    paths = sorted(glob.glob(os.path.join(sims_dir or "", "train_*.npz")))
    if not paths:
        raise SystemExit(f"cannot check the flow's summaries: no train_*.npz in {sims_dir!r}")
    d = np.load(paths[0])
    r = int(d["realization"][0])
    x = ns.simulate(ctx, r)["x"]
    if not np.allclose(x, d["x"][0], rtol=1e-9, atol=0):
        raise SystemExit(f"{paths[0]}: the simulator's summaries of training sky {r} differ from "
                         f"the stored ones (max |dx| {np.max(np.abs(x - d['x'][0])):.3g}); the "
                         "summary code changed after the flow was trained")


def compare_sky(exact, learned, bins):
    """Per-bin shift, its MC error, width ratio, and both correlation matrices."""
    be = np.stack([exact[:, sl].mean(1) for *_, sl in bins], axis=1)
    bl = np.stack([learned[:, sl].mean(1) for *_, sl in bins], axis=1)
    sd_e = be.std(0, ddof=1)
    return {"shift": (bl.mean(0) - be.mean(0)) / sd_e, "shift_mc": 1.0 / np.sqrt(bulk_ess(be)),
            "width_ratio": bl.std(0, ddof=1) / sd_e,
            "corr_exact": np.corrcoef(be, rowvar=False),
            "corr_learned": np.corrcoef(bl, rowvar=False)}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--flow", required=True)
    p.add_argument("--chains_dir", required=True)
    p.add_argument("--realizations", type=int, nargs="+", default=list(range(48)))
    p.add_argument("--n_learned", type=int, default=4000)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with open(os.path.join(args.flow, "sim_config.json")) as f:
        cfg = json.load(f)
    lo_t, hi_t = cfg["realization_min"], cfg["realization_max"]
    if any(lo_t <= r <= hi_t for r in args.realizations):
        raise SystemExit("an exact-ensemble sky is inside the flow's training range")
    q = np.load(cfg["qe_file"])
    ctx = ns.make_context(cfg["lmax"], cfg["nside"], cfg["noisesig"], cfg["phi_amplitude"],
                          cfg["nu_tt"], cfg["nu_pp"], q["cl_tt_lensed"], q["nl_qe"],
                          summary_set=cfg.get("summary_set", "v1"))
    check_summaries(ctx, cfg)
    flow = ConditionalFlow.load(args.flow)
    conditioning = load_conditioning(args.flow)
    lmax = int(cfg["lmax"])
    bins = bin_slices(lmax)
    rows = []
    for r in args.realizations:
        path = os.path.join(args.chains_dir, f"chain_r{r:03d}.npz")
        d = np.load(path)
        check_chain(d, cfg, path)
        sim = ns.simulate(ctx, r)
        for k in ("alm_true_packed", "phi_true_packed", "cl_true", "cl_phiphi_true"):
            if not np.array_equal(sim[k], d[k]):
                raise SystemExit(f"{path}: {k} differs from the simulator's sky {r}")
        x = conditioning(sim["x"], sim["data_alm_packed"].astype(np.float32))
        rows.append(compare_sky(exact_theta(d, lmax), flow.sample(x, args.n_learned, seed=r), bins))
    stack = {k: np.array([row[k] for row in rows]) for k in rows[0]}
    print(f"Learned vs exact on {len(rows)} skies ({args.n_learned} learned draws each)")
    print(f"{'block':>11} {'bin':>9} {'mean shift':>11} {'sd(shift)':>10} {'MC err':>8} {'width ratio':>12}")
    for j, (block, lo, hi, _) in enumerate(bins):
        flag = "  <- T0.1 band edge, not pooled" if lo >= BAND_EDGE_LO else ""
        s = stack["shift"][:, j]
        print(f"{block:>11} [{lo:>2},{hi:>3}) {s.mean():>+11.3f} {s.std(ddof=1):>10.3f} "
              f"{np.median(stack['shift_mc'][:, j]):>8.3f} {stack['width_ratio'][:, j].mean():>12.3f}{flag}")
    dc = stack["corr_learned"].mean(0) - stack["corr_exact"].mean(0)
    print(f"mean correlation matrix, learned - exact: max |diff| {np.nanmax(np.abs(dc)):.3f}")
    np.savez(args.out, realization=np.array(args.realizations), **stack,
             bins=np.array([(lo, hi) for _, lo, hi, _ in bins]),
             bin_blocks=np.array([b for b, *_ in bins]), flow=args.flow,
             chains_dir=args.chains_dir, n_learned=args.n_learned)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
