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
    norm_const=1.0,
):
    """Run the messenger Gibbs sampler for n_iter sweeps, return the final s.

    A_action(s)  -> A @ s  (harmonic -> full-sky pixel space)
    At_action(t) -> A^T @ t (full-sky pixel space -> harmonic space)

    norm_const  : scalar such that A^T A = norm_const * I; see
                  sample_s_given_t_orthonormal.

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
        s = sample_s_given_t_orthonormal(At_t, inv_cl_diag, tau2, rng, norm_const)
    return s
