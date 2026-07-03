"""Messenger-field Gibbs sampling for the Gaussian constrained problem
p(s | d) with a mask/inhomogeneous noise (Elsner & Wandelt 2013,
arXiv:1210.4931). See ROADMAP.md Phase 0c for why this replaces plain
diagonal-preconditioned CG (samplers.py::sample_alm_cg), which cannot
converge on a masked sky within any tractable iteration budget.

Model: d = A s + n,  n ~ N(0, N),  s ~ N(0, S).

The messenger field t is introduced via the generative reparametrisation
    t | s  ~  N(A s, T)
    d | t  ~  N(t, N - T)
(marginalising t recovers d | s ~ N(A s, N), the original model), which
requires T = tau2*I with tau2 <= min(N_ii) over observed pixels so that
N - T stays positive semi-definite there (masked entries have N_ii = inf
and impose no such constraint). Both Gibbs conditionals are then closed-form
Gaussian draws (no linear solve) — standard conjugate Gaussian updates:

    t | s, d  ~  N( (T^-1+(N-T)^-1)^-1 (T^-1 A s + (N-T)^-1 d),  (T^-1+(N-T)^-1)^-1 )
    s | t     ~  N( (S^-1+T^-1)^-1 T^-1 A^T t,                    (S^-1+T^-1)^-1 )

The s|t step is closed-form (diagonal in harmonic space) whenever A^T A is
proportional to the identity, e.g. an orthonormal SHT synthesis operator on
the full, unmasked sky — that's what makes this method sidestep the masked-
sky conditioning problem plain CG hits (ROADMAP.md Phase 0c).

This module implements the two conditionals generically (works for any A
with A^T A proportional to the identity, passed in as `At_action`); the
production wiring (Step 2, ROADMAP.md) will pass ducc0's full-sky forward/
adjoint SHT as `At_action`/`A_action`.
"""
import numpy as np


def sample_t_given_s(s_pix, d, Ninv, tau2, rng):
    """Draw t | s, d for each pixel independently.

    s_pix : 1-D array, A @ s evaluated at every (full-sky, unmasked) pixel
    d     : 1-D array, observed data at every pixel (masked pixels may hold
            any placeholder value; Ninv=0 there makes them irrelevant)
    Ninv  : 1-D array, 1/N_ii per pixel (0 for masked/unobserved pixels)
    tau2  : scalar, messenger covariance (must satisfy tau2 <= min(N_ii)
            over unmasked pixels, i.e. tau2 * Ninv < 1 everywhere)

    Uses the (N-T) "residual noise" precision Ninv_red = 1/(N-tau2), written
    as Ninv / (1 - tau2*Ninv) to stay finite (-> 0) as Ninv -> 0 (masked
    pixels) without ever forming N = 1/Ninv explicitly.

    Returns t : 1-D array, same shape as s_pix.
    """
    Ninv_red = Ninv / (1.0 - tau2 * Ninv)
    precision = 1.0 / tau2 + Ninv_red
    mean = (s_pix / tau2 + Ninv_red * d) / precision
    noise = rng.standard_normal(len(mean)) / np.sqrt(precision)
    return mean + noise


def sample_s_given_t_dense(At_t, inv_cl_diag, tau2, rng, AtA):
    """Draw s | t exactly, using the FULL A^T A rather than approximating it
    as diagonal (see sample_s_given_t_orthonormal's norm_const docstring).

    A real HEALPix SHT is only approximately orthonormal: A^T A has small
    (~1-2%) off-diagonal terms that the diagonal approximation discards.
    ROADMAP.md Phase 0c Step 3 found this discarded coupling is enough to
    either bias the messenger sampler's posterior (loose safety margin) or
    make the Gibbs chain diverge outright (tight/no margin) under masking.
    This exact update has neither failure mode, at the cost of an O(n^3)
    Cholesky solve per draw — only tractable while A^T A is small enough to
    hold densely (validation/diagnostic use; production at lmax=300 needs a
    scalable structured/low-rank approximation of AtA instead, see
    ROADMAP.md Phase 0c Step 5).

    At_t        : 1-D array, A^T @ t
    inv_cl_diag : 1-D array, 1/C_l per harmonic dof (prior precision)
    tau2        : scalar, messenger covariance
    AtA         : (n_alm, n_alm) array, the actual A^T A (not assumed diagonal)

    Returns s : 1-D array, same shape as At_t.
    """
    precision = AtA / tau2 + np.diag(inv_cl_diag)
    L = np.linalg.cholesky(precision)
    mean = np.linalg.solve(L.T, np.linalg.solve(L, At_t / tau2))
    noise = np.linalg.solve(L.T, rng.standard_normal(len(At_t)))
    return mean + noise


def build_block_cholesky(AtA_blocks, inv_cl_diag, tau2):
    """Precompute per-block Cholesky factors for sample_s_given_t_block.

    AtA_blocks : list of (idx, AtA_block) pairs — idx is a 1-D int array of
                 packed-alm positions belonging to one block (e.g. one m
                 value, see samplers.py::_calibrate_block_AtA), AtA_block is
                 the dense (len(idx), len(idx)) sub-matrix of A^T A restricted
                 to those rows/columns (exact, not approximated).

    Must be rebuilt whenever inv_cl_diag or tau2 change (i.e. once per outer
    Gibbs sweep, not once per model) — unlike the pure-diagonal norm_const
    path, this factorisation depends on the current C_l draw.
    """
    block_chol = []
    for idx, AtA_block in AtA_blocks:
        precision = AtA_block / tau2 + np.diag(inv_cl_diag[idx])
        L = np.linalg.cholesky(precision)
        block_chol.append((idx, L))
    return block_chol


def sample_s_given_t_block(At_t, tau2, rng, block_chol):
    """Draw s | t using a block-diagonal-by-m approximation of A^T A that is
    EXACT within each block (see build_block_cholesky).

    ROADMAP.md Phase 0c Step 5 found the off-diagonal A^T A a real HEALPix
    SHT carries is >99% concentrated between same-m pairs (same-parity L,
    magnitude roughly flat across the whole L range rather than decaying —
    scripts/analyze_AtA_structure.py), and negligible (<0.3% of total
    off-diagonal energy) between different m. Treating A^T A as exactly
    block-diagonal by m (dropping only that <0.3% residual) turns
    sample_s_given_t_dense's O(n_alm^3) whole-matrix Cholesky into a sum of
    small per-block solves — O(sum_m block_size(m)^2) instead — while
    reproducing sample_s_given_t_dense's accuracy far more closely than the
    plain diagonal approximation (see debug_messenger_masksky.py).

    At_t       : 1-D array, A^T @ t (packed alm layout)
    tau2       : scalar, messenger covariance
    block_chol : output of build_block_cholesky

    Returns s : 1-D array, same shape as At_t.
    """
    n = len(At_t)
    s = np.empty(n, dtype=np.float64)
    for idx, L in block_chol:
        rhs = At_t[idx] / tau2
        y = np.linalg.solve(L, rhs)
        mean = np.linalg.solve(L.T, y)
        noise = np.linalg.solve(L.T, rng.standard_normal(len(idx)))
        s[idx] = mean + noise
    return s


def sample_s_given_t_orthonormal(At_t, inv_cl_diag, tau2, rng, norm_const=1.0):
    """Draw s | t in harmonic space, assuming A^T A = norm_const * I.

    At_t        : 1-D array, A^T @ t (i.e. the forward/analysis SHT of t)
    inv_cl_diag : 1-D array, 1/C_l per harmonic dof (the prior precision;
                  see samplers.py::_build_inv_cl_diag for the m=0-vs-m>0
                  factor-of-2 convention this must match once wired into
                  the real model)
    tau2        : scalar, messenger covariance (same value used in the t
                  draw)
    norm_const  : scalar such that A^T A = norm_const * I. 1.0 for an exactly
                  orthonormal A (the Step 1 toy problem); for a real full-sky
                  HEALPix SHT this holds only approximately, with norm_const
                  = NPIX/(4*pi) (the standard HEALPix quadrature-weight
                  normalisation used throughout this codebase, e.g.
                  model.py::build_posterior_mass_sqrt's Ninv_eff).

    Returns s : 1-D array, same shape as At_t.
    """
    precision = inv_cl_diag + norm_const / tau2
    mean = (At_t / tau2) / precision
    noise = rng.standard_normal(len(mean)) / np.sqrt(precision)
    return mean + noise


def run_messenger_gibbs(
    d, Ninv, inv_cl_diag, tau2, A_action, At_action, rng, n_iter, s0=None,
    norm_const=1.0, AtA=None, block_chol=None,
):
    """Run the messenger Gibbs sampler for n_iter sweeps, return the final s.

    A_action(s)  -> A @ s  (harmonic -> full-sky pixel space)
    At_action(t) -> A^T @ t (full-sky pixel space -> harmonic space)

    norm_const  : scalar such that A^T A = norm_const * I; see
                  sample_s_given_t_orthonormal. Ignored if AtA or block_chol
                  is given.
    AtA         : optional (n_alm, n_alm) full A^T A matrix. If given, the
                  s|t step uses the exact dense update (sample_s_given_t_dense)
                  instead of the diagonal approximation — see that function's
                  docstring for why this matters under masking. Ignored if
                  block_chol is given.
    block_chol  : optional output of build_block_cholesky. If given, the s|t
                  step uses the block-diagonal-by-m update
                  (sample_s_given_t_block) — the scalable middle ground
                  between norm_const's cheap-but-biased diagonal and AtA's
                  exact-but-O(n^3) dense solve.

    This is the generic driver Step 1 validates against a dense reference;
    Step 2 (ROADMAP.md) substitutes ducc0's full-sky SHT for A_action/
    At_action and calls this from samplers.py::sample_alm_messenger.
    """
    n_alm = len(inv_cl_diag)
    s = np.zeros(n_alm) if s0 is None else np.asarray(s0, dtype=np.float64).copy()
    for _ in range(n_iter):
        s_pix = A_action(s)
        t = sample_t_given_s(s_pix, d, Ninv, tau2, rng)
        At_t = At_action(t)
        if block_chol is not None:
            s = sample_s_given_t_block(At_t, tau2, rng, block_chol)
        elif AtA is None:
            s = sample_s_given_t_orthonormal(At_t, inv_cl_diag, tau2, rng, norm_const)
        else:
            s = sample_s_given_t_dense(At_t, inv_cl_diag, tau2, rng, AtA)
    return s
