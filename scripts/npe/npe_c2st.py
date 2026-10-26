"""Classifier two-sample test, learned vs exact posterior (ROADMAP T1.1).

Per sky of the strict ensemble, a ridge-regularised quadratic logistic classifier
is cross-validated on the 8 bin-averaged (ln C_l^TT, ln C_L^phiphi) coordinates of
npe_vs_exact. Accuracy 0.5 means indistinguishable. Exact draws are autocorrelated,
so each sky keeps one draw per integrated autocorrelation time (n / min bulk ESS)
and the classifier only ever sees about ESS independent exact draws; the same
number of learned draws is used. Three accuracies per sky:
  * raw:    as is (width and shift mismatches count);
  * shape:  each class standardised by its own mean and sd (width and shift removed:
            correlation and non-Gaussian mismatch only);
  * null, null_shape: exact vs exact (alternate thinned draws), raw and standardised
            the same way as the shape test: the accuracy of a classifier with nothing
            to find (standardising each class costs a little accuracy in cross-validation).
The quadratic features see means, widths and the correlation matrix, not higher
moments, so a pass is a pass against those. Not a full-space test: the chains'
ESS does not support one.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/
import _paths  # noqa: E402,F401  (puts every script folder on sys.path)
from fig_convergence import bulk_ess  # noqa: E402
from npe_sbc import bin_slices  # noqa: E402
from npe_vs_exact import iter_skies  # noqa: E402

RIDGE = 1.0
N_FOLDS = 5


def quad_features(x):
    """[x, x_i x_j (i<=j)] for x already standardised over the pooled sample."""
    i, j = np.triu_indices(x.shape[1])
    return np.concatenate([x, x[:, i] * x[:, j]], axis=1)


def fit_logistic(f, y, ridge=RIDGE, iters=50):
    """Ridge logistic regression by Newton iteration; the intercept is unpenalised."""
    f1 = np.concatenate([np.ones((len(f), 1)), f], axis=1)
    pen = ridge * np.eye(f1.shape[1])
    pen[0, 0] = 0.0
    w = np.zeros(f1.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(f1 @ w, -30, 30)))
        step = np.linalg.solve(f1.T @ (f1 * (p * (1 - p))[:, None]) + pen,
                               f1.T @ (p - y) + pen @ w)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def c2st(a, b, seed=0, n_folds=N_FOLDS):
    """Cross-validated accuracy of a quadratic classifier separating rows of a and b."""
    rng = np.random.default_rng(seed)
    x = np.concatenate([a, b])
    y = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
    x = (x - x.mean(0)) / x.std(0)
    f = quad_features(x)
    fold = np.empty(len(y), dtype=int)
    for cls in (0, 1):  # stratified folds
        idx = rng.permutation(np.flatnonzero(y == cls))
        fold[idx] = np.arange(len(idx)) % n_folds
    correct = 0
    for k in range(n_folds):
        tr, te = fold != k, fold == k
        w = fit_logistic(f[tr], y[tr])
        pred = (np.concatenate([np.ones((te.sum(), 1)), f[te]], axis=1) @ w) > 0
        correct += np.sum(pred == (y[te] == 1))
    return correct / len(y)


def standardise(x):
    return (x - x.mean(0)) / x.std(0)


def thin_exact(be):
    """One draw per integrated autocorrelation time; at least 20 draws are kept."""
    n_eff = int(min(bulk_ess(be)))
    stride = max(1, int(np.floor(len(be) / max(n_eff, 20))))
    return be[::stride]


def score_sky(exact_bins, learned_bins, seed):
    e = thin_exact(exact_bins)
    m = len(e)
    rng = np.random.default_rng(seed)
    lr = learned_bins[rng.choice(len(learned_bins), size=m, replace=False)]
    return {"n_exact": m,
            "raw": c2st(e, lr, seed),
            "shape": c2st(standardise(e), standardise(lr), seed),
            "null": c2st(e[0::2], e[1::2], seed),
            "null_shape": c2st(standardise(e[0::2]), standardise(e[1::2]), seed)}


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
        lmax = int(json.load(f)["lmax"])
    bins = bin_slices(lmax)
    pooled = [i for i, (_, lo, *_) in enumerate(bins) if lo < 60]  # band edge not pooled
    rows, reals = [], []
    for r, exact, learned in iter_skies(args.flow, args.chains_dir, args.realizations,
                                        args.n_learned):
        be = np.stack([exact[:, sl].mean(1) for *_, sl in bins], axis=1)[:, pooled]
        bl = np.stack([learned[:, sl].mean(1) for *_, sl in bins], axis=1)[:, pooled]
        rows.append(score_sky(be, bl, r))
        reals.append(r)
        print(f"sky {r:3d}: n_exact {rows[-1]['n_exact']:4d}  raw {rows[-1]['raw']:.3f}  "
              f"shape {rows[-1]['shape']:.3f}  null_shape {rows[-1]['null_shape']:.3f}", flush=True)
    stack = {k: np.array([row[k] for row in rows]) for k in rows[0]}
    print(f"C2ST on {len(rows)} skies, {len(pooled)} pooled bin coordinates (0.5 = indistinguishable)")
    for k in ("raw", "shape", "null", "null_shape"):
        v = stack[k]
        print(f"  {k:>5}: mean {v.mean():.3f}  median {np.median(v):.3f}  "
              f"[{v.min():.3f}, {v.max():.3f}]  s.e. {v.std(ddof=1) / np.sqrt(len(v)):.3f}")
    dn = stack["shape"] - stack["null_shape"]
    print(f"  shape - null_shape: {dn.mean():+.3f} (paired s.e. {dn.std(ddof=1) / np.sqrt(len(dn)):.3f})")
    np.savez(args.out, realization=np.array(reals), pooled_bins=np.array(pooled), **stack,
             flow=args.flow, chains_dir=args.chains_dir)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
