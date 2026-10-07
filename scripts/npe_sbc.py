"""SBC of the learned posterior on held-out simulations (ROADMAP T1.1, figure 1).

For each held-out sky, n_draws posterior samples from the trained flow; the rank
of the truth per theta component and per bin-averaged ln C (the aggregator's
l-bins). Uniformity per bin uses aggregate_coverage_ranks' simulation-calibrated
discrete test and the rank spread (1/sqrt(12) for a calibrated posterior).
Refuses held-out sims whose configuration differs from the flow's, or that
overlap its training realizations.
"""

import argparse
import json
import os

import numpy as np
from aggregate_coverage_ranks import ELL_BINS, discrete_uniform_p, rank_spread
from npe_flow import ConditionalFlow, rank_of_truth
from npe_train import sim_config


def bin_slices(lmax):
    """(name, lo, hi, column slice) for both spectra; theta = [ln C_l^TT, ln C_L^pp], l = 2..lmax-1."""
    n = lmax - 2
    out = []
    for block, off in (("C_l^TT", 0), ("C_L^phiphi", n)):
        for lo, hi in ELL_BINS:
            hi = min(hi, lmax)
            if lo < hi:
                out.append((block, lo, hi, slice(off + lo - 2, off + hi - 2)))
    return out


def sbc_ranks(flow, theta, x, n_draws, lmax, chunk=50, seed=0):
    """Per-component and per-bin ranks, each in 0..n_draws."""
    bins = bin_slices(lmax)
    comp, binned = [], []
    for start in range(0, len(theta), chunk):
        smp = flow.sample_many(x[start:start + chunk], n_draws, seed=seed + start)
        for s, t in zip(smp, theta[start:start + chunk]):
            comp.append(rank_of_truth(s, t))
            binned.append([int(np.sum(s[:, sl].mean(1) < t[sl].mean())) for _, _, _, sl in bins])
    return np.array(comp), np.array(binned), bins


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--flow", required=True)
    p.add_argument("--heldout", required=True)
    p.add_argument("--n_sims", type=int, default=2000)
    p.add_argument("--n_draws", type=int, default=199)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with open(os.path.join(args.flow, "sim_config.json")) as f:
        trained_on = json.load(f)
    d = np.load(args.heldout)
    cfg = sim_config(d)
    diff = {k: (trained_on[k], cfg[k]) for k in cfg if trained_on[k] != cfg[k]}
    if diff:
        raise SystemExit(f"held-out sims differ from the flow's training configuration: {diff}")
    reals = d["realization"][: args.n_sims]
    if np.any((reals >= trained_on["realization_min"]) & (reals <= trained_on["realization_max"])):
        raise SystemExit("held-out realizations overlap the training range")

    flow = ConditionalFlow.load(args.flow)
    lmax = int(cfg["lmax"])
    comp, binned, bins = sbc_ranks(flow, d["theta"][: args.n_sims], d["x"][: args.n_sims],
                                   args.n_draws, lmax)
    print(f"SBC on {len(comp)} held-out sims, {args.n_draws} draws each "
          f"(uniform: mean rank {args.n_draws / 2:.1f}, spread {1 / np.sqrt(12):.4f})")
    print(f"{'block':>11} {'bin':>9} {'mean rank':>10} {'spread':>8} {'p':>8}")
    pvals = []
    for j, (block, lo, hi, _) in enumerate(bins):
        r = binned[:, j]
        pv = discrete_uniform_p(r, args.n_draws)
        pvals.append(pv)
        print(f"{block:>11} [{lo:>2},{hi:>3}) {r.mean():>10.2f} {rank_spread(r, args.n_draws):>8.4f} {pv:>8.4f}")
    np.savez(args.out, component_ranks=comp, bin_ranks=binned, bin_p=np.array(pvals),
             bins=np.array([(lo, hi) for _, lo, hi, _ in bins]),
             bin_blocks=np.array([b for b, _, _, _ in bins]), n_draws=args.n_draws,
             realization=reals, flow=args.flow, heldout=args.heldout, **cfg)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
