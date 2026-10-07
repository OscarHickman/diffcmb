"""Train the spectra-level learned posterior q(theta | x) (ROADMAP T1.1).

Reads the training blocks written by npe_simulate.py (train_*.npz in --sims_dir),
refuses blocks whose simulation configuration differs, fits npe_flow's
conditional MAF and saves it to --out together with the configuration it was
trained on (sim_config.json), so the SBC and the learned-vs-exact comparison
can check they score the posterior of the simulator they think they do.
"""

import argparse
import glob
import json
import os

import numpy as np
from npe_flow import ConditionalFlow

CONFIG_KEYS = ("lmax", "nside", "noisesig", "phi_amplitude", "nu_tt", "nu_pp",
               "lensing_operator", "fiducial", "qe_file")


def sim_config(d):
    return {k: d[k].item() for k in CONFIG_KEYS}


def load_blocks(paths):
    """Concatenate simulation blocks; refuse mixed configurations or repeated skies."""
    if not paths:
        raise SystemExit("no simulation blocks given")
    theta, x, reals, cfg = [], [], [], None
    for p in paths:
        d = np.load(p)
        c = sim_config(d)
        if cfg is None:
            cfg = c
        elif c != cfg:
            diff = {k: (cfg[k], c[k]) for k in cfg if cfg[k] != c[k]}
            raise SystemExit(f"{p}: simulation configuration differs from the first block: {diff}")
        theta.append(d["theta"])
        x.append(d["x"])
        reals.append(d["realization"])
    reals = np.concatenate(reals)
    if len(np.unique(reals)) != len(reals):
        raise SystemExit("simulation blocks share realizations")
    return np.concatenate(theta), np.concatenate(x), reals, cfg


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sims_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_layers", type=int, default=5)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.sims_dir, "train_*.npz")))
    theta, x, reals, cfg = load_blocks(paths)
    if not (np.all(np.isfinite(theta)) and np.all(np.isfinite(x))):
        raise SystemExit("non-finite training data")
    print(f"{len(theta)} training sims from {len(paths)} blocks; theta dim {theta.shape[1]}, "
          f"x dim {x.shape[1]}; config {cfg}", flush=True)
    flow = ConditionalFlow.fit(theta, x, n_layers=args.n_layers, hidden=args.hidden,
                               epochs=args.epochs, batch_size=args.batch_size,
                               learning_rate=args.learning_rate, patience=args.patience,
                               seed=args.seed)
    flow.save(args.out)
    with open(os.path.join(args.out, "sim_config.json"), "w") as f:
        json.dump({**cfg, "n_train": int(len(theta)), "realization_min": int(reals.min()),
                   "realization_max": int(reals.max()), "best_val_loss": flow.best_val_loss,
                   "history": flow.history, "train_args": vars(args)}, f, indent=1)
    print(f"best val NLL {flow.best_val_loss:.4f} after {len(flow.history)} epochs; saved {args.out}")


if __name__ == "__main__":
    main()
