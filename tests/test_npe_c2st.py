"""C2ST classifier behaviour on known cases (scripts/npe/npe_c2st.py)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import npe_c2st as c2  # noqa: E402

COV = np.array([[1.0, 0.6, 0.0], [0.6, 1.0, 0.3], [0.0, 0.3, 1.0]])


def draw(n, seed, scale=1.0, shift=0.0, cov=COV):
    rng = np.random.default_rng(seed)
    return shift + scale * rng.multivariate_normal(np.zeros(3), cov, size=n)


def test_identical_distributions_score_chance():
    accs = [c2.c2st(draw(400, s), draw(400, s + 100), seed=s) for s in range(6)]
    assert abs(np.mean(accs) - 0.5) < 0.03


def test_shifted_distribution_is_detected():
    assert c2.c2st(draw(400, 0), draw(400, 1, shift=1.5)) > 0.8


def test_wider_distribution_is_detected_raw_but_not_after_standardising():
    a, b = draw(500, 0), draw(500, 1, scale=1.4)
    assert c2.c2st(a, b) > 0.58
    shape = np.mean([c2.c2st(c2.standardise(draw(500, s)),
                             c2.standardise(draw(500, s + 50, scale=1.4)), seed=s)
                     for s in range(6)])
    assert abs(shape - 0.5) < 0.03


def test_correlation_mismatch_survives_standardising():
    other = np.array([[1.0, -0.6, 0.0], [-0.6, 1.0, 0.3], [0.0, 0.3, 1.0]])
    acc = c2.c2st(c2.standardise(draw(600, 0)), c2.standardise(draw(600, 1, cov=other)))
    assert acc > 0.7


def test_thin_exact_keeps_one_draw_per_autocorrelation_time():
    rng = np.random.default_rng(0)
    n = 4000
    x = np.zeros((n, 2))
    for t in range(1, n):  # AR(1), rho = 0.9: tau ~ 19
        x[t] = 0.9 * x[t - 1] + rng.standard_normal(2)
    kept = c2.thin_exact(x)
    assert 80 < len(kept) < 600
    assert len(c2.thin_exact(rng.standard_normal((n, 2)))) > 1500


def test_standardised_accuracy_falls_below_chance_more_for_small_samples():
    """Why null_shape must be compared at its own sample size (shape_matched)."""
    def mean_acc(n):
        return np.mean([c2.c2st(c2.standardise(draw(n, s)), c2.standardise(draw(n, s + 500)),
                                seed=s) for s in range(30)])
    small, large = mean_acc(30), mean_acc(240)
    assert small < large - 0.01
    assert small < 0.5


def test_shape_matched_equals_its_null_when_learned_is_exact():
    """With learned draws from the exact distribution, shape_matched agrees with
    null_shape on average (same draws per class), while shape, at twice the draws
    per class, sits measurably above it: the sample-size bias that made
    shape - null_shape look like a detection (2026-11-04)."""
    rows = [c2.score_sky(draw(40, s), draw(400, s + 1000), s) for s in range(300)]
    col = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    assert abs(np.mean(col["shape_matched"] - col["null_shape"])) < 0.02
    assert np.mean(col["shape"] - col["null_shape"]) > 0.025
