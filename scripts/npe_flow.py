"""Conditional normalizing flow for the learned posterior (ROADMAP T1.1).

q(theta | x): a stack of conditional masked autoregressive flows (TFP), with a
fixed random permutation between layers and a standard-normal base. theta and x
are standardised with training-set moments; log_prob is returned in the
original theta units. float32 (TFP's MAF builds its inner bijectors in float32);
moments and log-Jacobian terms are float64 numpy.

Validated against an exact linear-Gaussian posterior (tests/test_npe_flow.py)
before it is used on the CMB simulator.
"""

import json
import os

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tf_keras as tfk

tfb, tfd = tfp.bijectors, tfp.distributions
_DT = tf.float32


_LOG_SCALE_CLAMP = 3.0


def _clamped_affine(made):
    """Affine MAF step with a soft-clamped log-scale, |log s| < 3. An unbounded
    log-scale let single layers blow up early in training (toy test: first-epoch
    loss 59 against an exact 1.8) and left heavy, badly fitted tails."""
    def bijector_fn(y, conditional_input=None):
        shift, log_scale = tf.unstack(made(y, conditional_input=conditional_input), axis=-1)
        log_scale = _LOG_SCALE_CLAMP * tf.tanh(log_scale / _LOG_SCALE_CLAMP)
        return tfb.Chain([tfb.Shift(shift), tfb.Scale(log_scale=log_scale)])
    return bijector_fn


def rank_of_truth(samples, truth):
    """SBC rank per dimension: the number of posterior samples below the truth."""
    return np.sum(np.asarray(samples) < np.asarray(truth)[None, :], axis=0)


class ConditionalFlow:
    def __init__(self, dim, cond_dim, n_layers, hidden, seed, moments):
        self.config = {"dim": int(dim), "cond_dim": int(cond_dim), "n_layers": int(n_layers),
                       "hidden": [int(h) for h in hidden], "seed": int(seed)}
        self.moments = {k: np.asarray(v, dtype=np.float64) for k, v in moments.items()}
        rng = np.random.default_rng(seed)
        tf.random.set_seed(seed)
        layers, self._maf_names, self._mades = [], [], []
        for k in range(n_layers):
            made = tfb.AutoregressiveNetwork(
                params=2, event_shape=[dim], conditional=True,
                conditional_event_shape=[cond_dim], hidden_units=list(hidden),
                activation="tanh",
                # small weights: each layer starts near the identity (glorot gave a
                # first-epoch NLL ~100 against ~4 for the identity on the toy test)
                kernel_initializer=tfk.initializers.VarianceScaling(0.1, seed=seed + k))
            name = f"maf{k}"
            layers.append(tfb.MaskedAutoregressiveFlow(
                bijector_fn=_clamped_affine(made), name=name))
            self._maf_names.append(name)
            self._mades.append(made)
            if k < n_layers - 1:
                layers.append(tfb.Permute(rng.permutation(dim).astype(np.int32)))
        self._chain = tfb.Chain(layers[::-1])
        self._dist = tfd.TransformedDistribution(
            tfd.MultivariateNormalDiag(tf.zeros(dim, _DT)), self._chain)
        # build every variable once so checkpoints and training see them
        self._lp(tf.zeros([1, dim], _DT), tf.zeros([1, cond_dim], _DT))
        self._vars = [v for m in self._mades for v in m.trainable_variables]

    def _kw(self, c):
        return {n: {"conditional_input": c} for n in self._maf_names}

    def _lp(self, z, c):
        return self._dist.log_prob(z, bijector_kwargs=self._kw(c))

    def _std_x(self, x):
        return (np.atleast_2d(x) - self.moments["x_mu"]) / self.moments["x_sd"]

    def log_prob(self, theta, x):
        """log q(theta | x) in original units; theta (n, dim), x one condition (cond_dim,)."""
        theta = np.atleast_2d(theta)
        z = (theta - self.moments["t_mu"]) / self.moments["t_sd"]
        c = np.repeat(self._std_x(x), len(theta), axis=0)
        lp = self._lp(tf.constant(z, _DT), tf.constant(c, _DT)).numpy().astype(np.float64)
        return lp - np.sum(np.log(self.moments["t_sd"]))

    def sample(self, x, n, seed=0):
        c = tf.constant(np.repeat(self._std_x(x), n, axis=0), _DT)
        z = self._dist.sample(n, seed=tf.constant([seed, 0], tf.int32),
                              bijector_kwargs=self._kw(c)).numpy()
        return z.astype(np.float64) * self.moments["t_sd"] + self.moments["t_mu"]

    def sample_many(self, xs, n, seed=0):
        """n draws for each condition in xs (k, cond_dim) in one pass -> (k, n, dim)."""
        xs = np.atleast_2d(xs)
        c = tf.constant(np.repeat(self._std_x(xs), n, axis=0), _DT)
        z = self._dist.sample(len(c), seed=tf.constant([seed, 0], tf.int32),
                              bijector_kwargs=self._kw(c)).numpy()
        out = z.astype(np.float64) * self.moments["t_sd"] + self.moments["t_mu"]
        return out.reshape(len(xs), n, -1)

    @classmethod
    def fit(cls, theta, x, n_layers=5, hidden=(256, 256), epochs=100, batch_size=512,
            learning_rate=1e-3, val_frac=0.05, patience=10, seed=0, verbose=True):
        """Maximum-likelihood training; keeps the weights with the best validation loss."""
        theta, x = np.asarray(theta, np.float64), np.asarray(x, np.float64)
        moments = {"t_mu": theta.mean(0), "t_sd": theta.std(0), "x_mu": x.mean(0), "x_sd": x.std(0)}
        flow = cls(theta.shape[1], x.shape[1], n_layers, hidden, seed, moments)
        z = (theta - moments["t_mu"]) / moments["t_sd"]
        c = (x - moments["x_mu"]) / moments["x_sd"]
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(z))
        n_val = max(1, int(val_frac * len(z)))
        val, tr = idx[:n_val], idx[n_val:]
        opt = tfk.optimizers.Adam(learning_rate)

        @tf.function
        def step(zb, cb):
            with tf.GradientTape() as tape:
                loss = -tf.reduce_mean(flow._lp(zb, cb))
            opt.apply_gradients(zip(tape.gradient(loss, flow._vars), flow._vars))
            return loss

        best, best_w, bad = np.inf, None, 0
        flow.history = []
        for ep in range(epochs):
            perm = rng.permutation(tr)
            losses = [step(tf.constant(z[b], _DT), tf.constant(c[b], _DT)).numpy()
                      for b in np.array_split(perm, max(1, len(perm) // batch_size))]
            # Fresh tensors every epoch: TFP bijectors cache inverse() results by
            # input-tensor identity, so re-scoring one validation tensor returned
            # epoch-1 values and early stopping restored the wrong weights.
            vl = float(-tf.reduce_mean(flow._lp(tf.constant(z[val], _DT),
                                                tf.constant(c[val], _DT))).numpy())
            flow.history.append((float(np.mean(losses)), vl))
            if verbose:
                print(f"  epoch {ep + 1}: train {np.mean(losses):.4f}  val {vl:.4f}", flush=True)
            if vl < best:
                best, best_w, bad = vl, [v.numpy() for v in flow._vars], 0
            else:
                bad += 1
                if bad >= patience:
                    break
        for v, w in zip(flow._vars, best_w):
            v.assign(w)
        flow.best_val_loss = best
        return flow

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.config, f)
        np.savez(os.path.join(path, "moments.npz"), **self.moments)
        np.savez(os.path.join(path, "weights.npz"),
                 **{f"w{i}": v.numpy() for i, v in enumerate(self._vars)})

    @classmethod
    def load(cls, path):
        with open(os.path.join(path, "config.json")) as f:
            cfg = json.load(f)
        m = np.load(os.path.join(path, "moments.npz"))
        flow = cls(cfg["dim"], cfg["cond_dim"], cfg["n_layers"], cfg["hidden"], cfg["seed"],
                   {k: m[k] for k in m.files})
        w = np.load(os.path.join(path, "weights.npz"))
        if len(w.files) != len(flow._vars):
            raise ValueError(f"{path}: {len(w.files)} weight arrays, flow has {len(flow._vars)}")
        for i, v in enumerate(flow._vars):
            v.assign(w[f"w{i}"])
        return flow
