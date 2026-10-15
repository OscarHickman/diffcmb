"""npe_flow (ROADMAP T1.1): the conditional flow behind the learned posterior.

Validated against an exact posterior first (standing discipline: dense or exact
reference before production): a linear-Gaussian problem whose posterior is
known in closed form.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytest.importorskip("tensorflow_probability")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import npe_flow as nf  # noqa: E402

# theta ~ N(0, I_D); x = A theta + N(0, s^2 I). Posterior is Gaussian:
#   cov = (I + A^T A / s^2)^-1, mean = cov A^T x / s^2.
D, S = 3, 0.5
A = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, -0.5], [0.3, 0.0, 1.0]])


def _draw(n, seed):
    rng = np.random.default_rng(seed)
    theta = rng.normal(size=(n, D))
    x = theta @ A.T + S * rng.normal(size=(n, D))
    return theta, x


def _exact(x):
    cov = np.linalg.inv(np.eye(D) + A.T @ A / S ** 2)
    return x @ (cov @ A.T / S ** 2).T, cov


@pytest.fixture(scope="module")
def trained():
    theta, x = _draw(20000, 0)
    npe = nf.ConditionalFlow.fit(theta, x, n_layers=3, hidden=(64, 64), epochs=30,
                                 batch_size=256, learning_rate=2e-3, seed=0, verbose=False)
    return npe


def test_flow_recovers_the_exact_gaussian_posterior(trained):
    _, x_test = _draw(5, 1)
    mean_exact, cov_exact = _exact(x_test)
    for i in range(len(x_test)):
        smp = trained.sample(x_test[i], 4000, seed=i)
        assert smp.shape == (4000, D)
        sd_exact = np.sqrt(np.diag(cov_exact))
        np.testing.assert_allclose(smp.mean(0), mean_exact[i], atol=0.1 * sd_exact.max() + 0.03)
        np.testing.assert_allclose(smp.std(0), sd_exact, rtol=0.12)


def test_log_prob_matches_the_exact_density(trained):
    _, x_test = _draw(3, 2)
    mean_exact, cov_exact = _exact(x_test)
    from scipy.stats import multivariate_normal

    for i in range(len(x_test)):
        th = mean_exact[i] + 0.3 * np.sqrt(np.diag(cov_exact))
        want = multivariate_normal(mean_exact[i], cov_exact).logpdf(th)
        got = trained.log_prob(th[None], x_test[i])[0]
        assert abs(got - want) < 0.25


def test_save_and_load_round_trip(trained, tmp_path):
    path = tmp_path / "flow"
    trained.save(path)
    again = nf.ConditionalFlow.load(path)
    _, x_test = _draw(1, 3)
    th = np.zeros((2, D))
    np.testing.assert_allclose(again.log_prob(th, x_test[0]), trained.log_prob(th, x_test[0]),
                               rtol=1e-6)
    np.testing.assert_array_equal(again.sample(x_test[0], 10, seed=7),
                                  trained.sample(x_test[0], 10, seed=7))


def test_ranks_of_truth_are_uniform_for_the_exact_posterior():
    # the SBC rank helper: truth drawn from the same Gaussian as the samples
    rng = np.random.default_rng(0)
    ranks = np.array([nf.rank_of_truth(rng.normal(size=(99, 2)), rng.normal(size=2))
                      for _ in range(4000)])
    assert ranks.shape == (4000, 2) and ranks.min() >= 0 and ranks.max() <= 99
    np.testing.assert_allclose(ranks.mean(0), 49.5, atol=1.5)


def test_recorded_validation_loss_is_the_restored_weights_loss(trained):
    # 2026-10-07: TFP caches inverse() by input-tensor identity, and fit() reused
    # one validation tensor, so the recorded val loss was stale and early stopping
    # restored the wrong weights. The best recorded val loss must equal the
    # restored model's loss on the same split.
    import tensorflow as tf

    theta, x = _draw(20000, 0)
    m = trained.moments
    z, c = (theta - m["t_mu"]) / m["t_sd"], (x - m["x_mu"]) / m["x_sd"]
    val = np.random.default_rng(0).permutation(len(z))[: int(0.05 * len(z))]
    nll = -np.mean(trained._lp(tf.constant(z[val], tf.float32), tf.constant(c[val], tf.float32)).numpy())
    assert abs(nll - trained.best_val_loss) < 1e-4


def test_weight_decay_shrinks_the_kernels_and_val_loss_stays_unpenalised():
    """T1.1 route c: an L2 penalty on the MADE kernels, for the early overfitting.
    The recorded losses must stay the NLL (comparable across penalties)."""
    import tensorflow as tf

    theta, x = _draw(4000, 1)
    kw = {"n_layers": 2, "hidden": (32, 32), "epochs": 4, "batch_size": 256,
          "learning_rate": 2e-3, "seed": 0, "verbose": False}
    plain = nf.ConditionalFlow.fit(theta, x, **kw)
    decayed = nf.ConditionalFlow.fit(theta, x, weight_decay=1e-2, **kw)

    def kernel_norm(f):
        return sum(float(np.sum(v.numpy() ** 2)) for v in f._vars if "kernel" in v.name)

    assert kernel_norm(decayed) < 0.8 * kernel_norm(plain)
    m = decayed.moments
    z, c = (theta - m["t_mu"]) / m["t_sd"], (x - m["x_mu"]) / m["x_sd"]
    val = np.random.default_rng(0).permutation(len(z))[: int(0.05 * len(z))]
    nll = -np.mean(decayed._lp(tf.constant(z[val], tf.float32),
                               tf.constant(c[val], tf.float32)).numpy())
    assert abs(nll - decayed.best_val_loss) < 1e-4
    assert decayed.config["weight_decay"] == 1e-2
