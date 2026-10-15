"""Learned map compressor for the learned posterior (ROADMAP T1.1, width-gap route b).

The spectra-level NPE conditions on hand summaries (npe_simulate's summary sets),
and its posterior is wider than the exact one by the information those summaries
lose. This network reads the data a_lm itself, plus the summaries x, and is
trained (MSE on standardised theta) to return a point estimate of theta; that
estimate is appended to x as the flow's conditioning (load_conditioning), so the
flow can only gain from it.

Map branch: the packed a_lm are whitened per multipole (training-set sd) and laid
on an (l, m, re/im) grid; a convolution over that grid followed by a square is a
learned local quadratic form, i.e. it can estimate power and the couplings of
neighbouring (l, m) that lensing induces (tests/test_npe_compress.py pins both);
the squares are averaged over m and passed on as log features per l.

CLI: train on the train_*.npz blocks of a "v2" simulation directory (the only
blocks that store data_alm) and save to --out.
"""

import argparse
import glob
import json
import os

import numpy as np
import tensorflow as tf
import tf_keras as tfk

from diffcmb.alm_utils import packed_length, packed_sizes
from diffcmb.samplers import _alm_index_lm

_DT = tf.float32
_LOG_EPS = 1e-6


def _grid_index(lmax):
    """For each cell of the (lmax-2, lmax, 2) grid, its packed index, or
    packed_length(lmax) (a zero column) where m > l or m = 0 has no imaginary part."""
    n_real, n_imag = packed_sizes(lmax)
    L, m = _alm_index_lm(lmax, n_real, n_imag)
    part = (np.arange(n_real + n_imag) >= n_real).astype(np.int64)
    idx = np.full((lmax - 2) * lmax * 2, n_real + n_imag, dtype=np.int64)
    idx[((L - 2) * lmax + m) * 2 + part] = np.arange(n_real + n_imag)
    return idx, L


class MapCompressor:
    def __init__(self, lmax, x_dim, theta_dim, moments, conv_filters=32, seed=0):
        self.config = {"lmax": int(lmax), "x_dim": int(x_dim), "theta_dim": int(theta_dim),
                       "conv_filters": int(conv_filters), "seed": int(seed)}
        self.moments = {k: np.asarray(v, dtype=np.float64) for k, v in moments.items()}
        tf.random.set_seed(seed)
        idx, _ = _grid_index(lmax)
        n_pk = packed_length(lmax)
        a_in = tfk.Input((n_pk,), name="alm")
        x_in = tfk.Input((x_dim,), name="x")
        idx_t = tf.constant(idx, tf.int32)

        def to_grid(a):
            padded = tf.concat([a, tf.zeros_like(a[:, :1])], axis=1)
            return tf.reshape(tf.gather(padded, idx_t, axis=1), [-1, lmax - 2, lmax, 2])

        g = tfk.layers.Lambda(to_grid)(a_in)
        h = tfk.layers.Conv2D(conv_filters, 5, padding="same", use_bias=False)(g)
        h = tfk.layers.Lambda(lambda t: tf.math.log(tf.reduce_mean(t ** 2, axis=2) + _LOG_EPS))(h)
        h = tfk.layers.Conv1D(64, 5, padding="same", activation="relu")(h)
        h = tfk.layers.Dense(256, activation="relu")(tfk.layers.Flatten()(h))
        xb = tfk.layers.Dense(256, activation="relu")(x_in)
        o = tfk.layers.Dense(256, activation="relu")(tfk.layers.Concatenate()([h, xb]))
        self.model = tfk.Model([a_in, x_in], tfk.layers.Dense(theta_dim)(o))

    def _inputs(self, alm, x):
        m = self.moments
        a = np.asarray(alm, np.float64) / m["alm_sd"]
        c = (np.atleast_2d(np.asarray(x, np.float64)) - m["x_mu"]) / m["x_sd"]
        return [tf.constant(a, _DT), tf.constant(c, _DT)]

    def predict(self, alm, x, batch_size=2048):
        """Point estimate of theta, in original units, for each (a_lm, x) row."""
        alm, x = np.atleast_2d(alm), np.atleast_2d(x)
        out = [self.model(self._inputs(alm[i:i + batch_size], x[i:i + batch_size]),
                          training=False).numpy()
               for i in range(0, len(alm), batch_size)]
        return np.concatenate(out).astype(np.float64) * self.moments["t_sd"] + self.moments["t_mu"]

    @classmethod
    def fit(cls, alm, x, theta, lmax, epochs=50, batch_size=256, learning_rate=1e-3,
            val_frac=0.05, patience=5, seed=0, conv_filters=32, verbose=True):
        """MSE on standardised theta; keeps the weights with the best validation loss."""
        alm = np.asarray(alm, np.float32)
        x, theta = np.asarray(x, np.float64), np.asarray(theta, np.float64)
        _, L = _grid_index(lmax)
        sd_l = {ell: float(np.std(alm[:, L == ell])) for ell in np.unique(L)}
        moments = {"alm_sd": np.array([sd_l[ell] for ell in L]),
                   "x_mu": x.mean(0), "x_sd": x.std(0), "t_mu": theta.mean(0), "t_sd": theta.std(0)}
        comp = cls(lmax, x.shape[1], theta.shape[1], moments, conv_filters, seed)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(theta))
        n_val = max(1, int(val_frac * len(theta)))
        val, tr = perm[:n_val], perm[n_val:]
        z = (theta - moments["t_mu"]) / moments["t_sd"]
        opt = tfk.optimizers.Adam(learning_rate)
        weights = comp.model.trainable_variables

        @tf.function
        def step(a, c, t):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean((comp.model([a, c], training=True) - t) ** 2)
            opt.apply_gradients(zip(tape.gradient(loss, weights), weights))
            return loss

        def val_loss():
            out = [np.mean((comp.model(comp._inputs(alm[b], x[b]), training=False).numpy()
                            - z[b]) ** 2, axis=1)
                   for b in np.array_split(val, max(1, len(val) // 2048))]
            return float(np.mean(np.concatenate(out)))

        best, best_w, bad = np.inf, None, 0
        comp.history = []
        for ep in range(epochs):
            order = rng.permutation(tr)
            losses = []
            for b in np.array_split(order, max(1, len(order) // batch_size)):
                a, c = comp._inputs(alm[b], x[b])
                losses.append(step(a, c, tf.constant(z[b], _DT)).numpy())
            vl = val_loss()
            comp.history.append((float(np.mean(losses)), vl))
            if verbose:
                print(f"  epoch {ep + 1}: train MSE {np.mean(losses):.4f}  val {vl:.4f}", flush=True)
            if vl < best:
                best, best_w, bad = vl, [w.numpy() for w in weights], 0
            else:
                bad += 1
                if bad >= patience:
                    break
        for w, v in zip(weights, best_w):
            w.assign(v)
        comp.best_val_loss = best
        return comp

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.config, f)
        np.savez(os.path.join(path, "moments.npz"), **self.moments)
        np.savez(os.path.join(path, "weights.npz"),
                 **{f"w{i}": w.numpy() for i, w in enumerate(self.model.trainable_variables)})

    @classmethod
    def load(cls, path):
        with open(os.path.join(path, "config.json")) as f:
            cfg = json.load(f)
        m = np.load(os.path.join(path, "moments.npz"))
        comp = cls(cfg["lmax"], cfg["x_dim"], cfg["theta_dim"], {k: m[k] for k in m.files},
                   cfg["conv_filters"], cfg["seed"])
        w = np.load(os.path.join(path, "weights.npz"))
        variables = comp.model.trainable_variables
        if len(w.files) != len(variables):
            raise ValueError(f"{path}: {len(w.files)} weight arrays, model has {len(variables)}")
        for i, v in enumerate(variables):
            v.assign(w[f"w{i}"])
        return comp


def load_conditioning(flow_dir):
    """The flow's conditioning as a function of (x, data a_lm): x itself, or x with
    the compressor's estimate appended when sim_config.json names a compressor."""
    with open(os.path.join(flow_dir, "sim_config.json")) as f:
        path = json.load(f).get("compressor")
    if not path:
        return lambda x, alm: np.atleast_2d(x) if np.ndim(x) > 1 else np.asarray(x)
    comp = MapCompressor.load(path)

    def conditioning(x, alm):
        if alm is None:
            raise ValueError(f"the flow at {flow_dir} conditions on a compressor of the "
                             "data a_lm, and none was given")
        x2 = np.atleast_2d(x)
        out = np.concatenate([x2, comp.predict(np.atleast_2d(alm), x2)], axis=1)
        return out if np.ndim(x) > 1 else out[0]

    return conditioning


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sims_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--conv_filters", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from npe_train import load_blocks
    paths = sorted(glob.glob(os.path.join(args.sims_dir, "train_*.npz")))
    theta, x, reals, cfg = load_blocks(paths)
    missing = [q for q in paths if "data_alm" not in np.load(q).files]
    if missing:
        raise SystemExit(f"blocks without data_alm (simulate with --summary_set v2): {missing}")
    alm = np.concatenate([np.load(q)["data_alm"] for q in paths])
    print(f"{len(theta)} sims, a_lm dim {alm.shape[1]}, x dim {x.shape[1]}; config {cfg}", flush=True)
    comp = MapCompressor.fit(alm, x, theta, lmax=int(cfg["lmax"]), epochs=args.epochs,
                             batch_size=args.batch_size, learning_rate=args.learning_rate,
                             patience=args.patience, seed=args.seed,
                             conv_filters=args.conv_filters)
    comp.save(args.out)
    with open(os.path.join(args.out, "sim_config.json"), "w") as f:
        json.dump({**cfg, "n_train": int(len(theta)), "realization_min": int(reals.min()),
                   "realization_max": int(reals.max()), "best_val_mse": comp.best_val_loss,
                   "history": comp.history, "train_args": vars(args)}, f, indent=1)
    print(f"best val MSE {comp.best_val_loss:.4f} (standardised theta; 1 = prior) after "
          f"{len(comp.history)} epochs; saved {args.out}")


if __name__ == "__main__":
    main()
