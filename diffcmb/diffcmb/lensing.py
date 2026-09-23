"""Differentiable CMB lensing operator (Phase 1).

Implements the remapping  T_lensed(n) = T_unlensed(n + ∇φ(n))  as a
TF operation differentiable with respect to both the CMB signal alm and the
lensing potential phi_alm.

Public API
----------
* ``deflection_field``        — phi_alm (healpy ordering) → (dθ, dφ) [rad]
* ``precompute_lensing``      — dθ, dφ → HEALPix neighbor indices + bilinear weights
* ``apply_lensing_tf``        — T_map × (neighbors, weights) → T_lensed  [diff. w.r.t. T_map]
* ``lens_map_tf``             — alm + phi_alm_np → T_lensed              [diff. w.r.t. alm]
* ``lens_map_phi_diff_tf``    — T_map + phi_alm_packed → T_lensed        [diff. w.r.t. both]
* ``psi_lensed``              — lensed log-posterior matching _psi_tf_raw interface

Gradient strategy
-----------------
* dL/d alm     : TF autodiff through the Y-matrix matvec (no new infrastructure).
* dL/d phi_alm : custom_gradient implementing the adjoint chain:
    upstream → dL/d(bilinear weights) [FD of hp.get_interp_weights]
             → dL/d(deflection field) [scatter to full sky]
             → dL/d(phi_alm)         [spin-1 SHT adjoint via hp.map2alm_spin]

Reference: Lewis & Challinor 2006 (Phys. Rep. 429, 1); Carron & Lewis 2017 (arXiv:1704.08230).
"""

from typing import NamedTuple

import numpy as np

try:
    import healpy as hp
except ImportError:
    hp = None

try:
    from .alm_utils import almhotmo, almmotho, invgamma_shape_for_spectrum, packed_sizes
except ImportError:  # pragma: no cover - alm_utils needs healpy/scipy
    almhotmo = None
    almmotho = None
    invgamma_shape_for_spectrum = None
    packed_sizes = None

try:
    import tensorflow as tf
except ImportError:
    tf = None


# ---------------------------------------------------------------------------
# Packed alm format helpers  (mirrors the CMB alm encoding in the model)
#
# Encoding: for L=2..lmax-1, m=0..L
#   real_parts[...] : Re(a_{L,m})  for all (L,m) with L≥2
#   imag_parts[...] : Im(a_{L,m})  for m≥2 only  (m=0,1 imaginary is forced to 0)
#
# This matches splittosingularalm / splittosingularalm_tf exactly.
#
# ORDERING: the packed layout is L-major ("author ordering", mo), the same as
# model.py.  healpy and ducc0 alm arrays are m-major (ho, hp.Alm.getidx).  The
# two conversions below MUST route through almmotho/almhotmo -- indexing a
# healpy-ordered array with the author-ordering formula L*(L+1)//2 + m is a
# bijection, so the packed→hp→packed round-trip still looks lossless while
# every coefficient sits at the wrong multipole on the sky.  See
# test_alm_hp_to_packed_uses_true_healpy_ordering.
# ---------------------------------------------------------------------------

def _alm_packed_to_hp(phi_packed: np.ndarray, lmax: int) -> np.ndarray:
    """Packed real+imag → healpy complex alm (length lmax*(lmax+1)//2)."""
    n_real, _ = packed_sizes(lmax)
    real_p = phi_packed[:n_real]
    imag_p = phi_packed[n_real:]
    len_alm = lmax * (lmax + 1) // 2
    alm_mo = np.zeros(len_alm, dtype=np.complex128)
    r_idx = 0
    i_idx = 0
    for L in range(2, lmax):
        for m in range(L + 1):
            mo_idx = L * (L + 1) // 2 + m
            if m == 0:
                alm_mo[mo_idx] = real_p[r_idx]
                r_idx += 1
            else:
                alm_mo[mo_idx] = real_p[r_idx] + 1j * imag_p[i_idx]
                r_idx += 1
                i_idx += 1
    return almmotho(alm_mo, lmax)


def _alm_hp_to_packed(alm_hp: np.ndarray, lmax: int) -> np.ndarray:
    """Healpy complex alm → packed real+imag float64 vector."""
    n_real, n_imag = packed_sizes(lmax)
    real_p = np.zeros(n_real, dtype=np.float64)
    imag_p = np.zeros(n_imag, dtype=np.float64)
    alm_mo = almhotmo(alm_hp, lmax)
    r_idx = 0
    i_idx = 0
    for L in range(2, lmax):
        for m in range(L + 1):
            mo_idx = L * (L + 1) // 2 + m
            if m == 0:
                real_p[r_idx] = alm_mo[mo_idx].real
                r_idx += 1
            else:
                real_p[r_idx] = alm_mo[mo_idx].real
                imag_p[i_idx] = alm_mo[mo_idx].imag
                r_idx += 1
                i_idx += 1
    return np.concatenate([real_p, imag_p])


# ---------------------------------------------------------------------------
# Optional Block 4 — C_L^phiphi | phi exact inverse-Gamma draw
#
# Mirrors model.py::CosmologyAdvancedSampling.compute_sl_np /
# sample_cl_given_alm (Block 1) exactly: log_prob_phi_block's Gaussian prior
# term on phi has the identical per-L quadratic-form structure (m=0 weight 1,
# m>=1 weight 2*(re^2+im^2), same packed layout as the CMB alm above) as the
# CMB alm prior that Block 1 exploits for its exact inverse-Gamma conditional,
# so the same exact-conjugacy argument applies to phi's own power spectrum.
# Implemented as free functions (not a model method) since phi has no
# lmax-independent state beyond the packed vector itself.
# ---------------------------------------------------------------------------

def compute_sl_phi_np(phi_packed: np.ndarray, lmax: int) -> np.ndarray:
    """Compute S_L = sum_{m=-L}^{L} |phi_{L,m}|^2 for L=0..lmax-1.

    phi_packed: 1-D numpy array, packed real+imag layout as returned by
    _alm_hp_to_packed (real parts L=2..lmax-1 m=0..L, then imaginary parts
    L=2..lmax-1 m=1..L) -- same convention as model.py::compute_sl_np's
    alm_flat_np argument.
    """
    n_real, _ = packed_sizes(lmax)
    real_p = phi_packed[:n_real]
    imag_p = phi_packed[n_real:]
    S = np.zeros(lmax)
    r_idx = 0
    i_idx = 0
    for L in range(2, lmax):
        for m in range(L + 1):
            re = real_p[r_idx]
            r_idx += 1
            im = imag_p[i_idx] if m >= 1 else 0.0
            if m >= 1:
                i_idx += 1
            if m == 0:
                S[L] += re * re
            else:
                S[L] += 2.0 * (re * re + im * im)
    return S


def sample_cl_phiphi_given_phi(phi_packed: np.ndarray, lmax: int, rng=None,
                               prior_nu=None, cl_phiphi_fid=None) -> np.ndarray:
    """Sample ln(C_L^phiphi) | phi for L=2..lmax-1 from the exact inverse-Gamma
    conditional implied by log_prob_phi_block's Gaussian prior term:

        C_L^phiphi | phi ~ InvGamma(alpha=k_L/2 - 1, beta=S_L/2)

    where k_L = 2L+1 is the number of REAL packed dof at multipole L (with
    Im(a_{L,1}) restored, the packed vector carries the full 2L+1 real dof).

    where S_L = sum_{m=-L}^{L} |phi_{L,m}|^2 (compute_sl_phi_np). Same
    structure as model.py::sample_cl_given_alm (Block 1), applied to phi.

    OPTIONAL PROPER PRIOR (`prior_nu`, `cl_phiphi_fid`; both None = the exact
    old behaviour, following the same contract as the beam/noise_map kwargs).

    The default conditional above corresponds to a FLAT, IMPROPER prior on
    C_L^phiphi, and that is not a cosmetic issue. Integrating C_L out of the
    joint leaves a marginal on phi proportional to S_L^{-(L-0.5)}; against the
    (2L+1)-component radial measure r^{2L}dr (with S_L = r^2) that is

        p(r) ∝ r^1,

    i.e. FLAT IN S_L -- improper, and *rising* with amplitude. The phi
    amplitude is then constrained by the lensing likelihood alone, which is
    the mechanism behind the 2026-08-28 coverage-ensemble failure (phi frozen
    1e3-1e5x above truth, job 11887897) and the reason the phi SBC rank in the
    2026-08-31 ensemble (job 11899585, mean_u=0.367) is not a valid
    calibration test: the truth was drawn from a proper N(0, C_L^phiphi,fid)
    the sampler is not targeting.

    Passing `prior_nu=nu` with a fiducial spectrum places a conjugate
    InvGamma(alpha_0 = nu/2, beta_0 = nu*C_L^fid/2) prior on each C_L, so

        C_L^phiphi | phi ~ InvGamma(k_L/2 + nu/2, (S_L + nu*C_L^fid)/2)
                         = InvGamma(L + 0.5 + nu/2, (S_L + nu*C_L^fid)/2).

    `nu` reads as an effective number of prior "pseudo-modes", on the same
    footing as the k_L = 2L+1 real modes the data supplies at multipole L --
    so nu is weak where the data is informative (high L) and does most of its
    work at low L, which is exactly where the improper prior hurts.

    REQUIREMENT nu > 0, enforced. With the prior in place the marginal tail
    becomes p(r) ∝ r^{-1-nu} (the k_L cancels), normalisable exactly for
    nu > 0; at nu = 0 it decays as 1/r and still diverges logarithmically. A
    nu <= 0 would leave the target improper while *appearing* to fix it, so it
    is rejected rather than accepted with a warning. (This threshold was nu > 2
    while the code assumed 2L+1 dof -- it has to be re-derived alongside the
    exponent, not carried over.)

    Note on clipping: model.py::sample_cl_given_alm clips the sampled C_l to
    [3e-7, 3e6], a range tuned to typical CMB C_l units. C_L^phiphi lives on
    a very different scale (test fixtures and gate scripts in this repo use
    cl_phiphi_full ~ 1e-6 down to ~1e-12), so that range would silently floor
    away several orders of magnitude of real signal. We use a much wider
    [1e-30, 1e10] clip here instead -- matching the 1e-30 floor already used
    for cl_phiphi_full in build_phi_prior_mass_sqrt/build_phi_posterior_mass_sqrt
    (samplers.py) -- purely to guard against non-finite results, not to impose
    a physically meaningful bound.

    Returns lncl_phiphi array of shape (lmax-2,).
    """
    if rng is None:
        rng = np.random.default_rng()
    if np.any(~np.isfinite(phi_packed)):
        raise ValueError(
            "Non-finite values (NaNs/Infs) detected in phi_packed during "
            "sample_cl_phiphi_given_phi!"
        )
    use_proper_prior = prior_nu is not None
    if use_proper_prior:
        prior_nu = float(prior_nu)
        if not np.isfinite(prior_nu) or prior_nu <= 0.0:
            raise ValueError(
                f"prior_nu must be > 0 for a proper phi marginal (got {prior_nu}); "
                "the marginal tail goes as r^(-1-nu) -- k_L cancels, so the "
                "threshold is nu > 0 for k_L = 2L+1 exactly as it was for 2L. "
                "See this function's docstring."
            )
        if cl_phiphi_fid is None:
            raise ValueError(
                "cl_phiphi_fid is required when prior_nu is set -- the prior is "
                "centred on it."
            )
        cl_phiphi_fid = np.asarray(cl_phiphi_fid, dtype=np.float64)
        if cl_phiphi_fid.shape[0] < lmax:
            raise ValueError(
                f"cl_phiphi_fid must have length >= lmax ({lmax}), got "
                f"{cl_phiphi_fid.shape[0]}"
            )
        if np.any(~np.isfinite(cl_phiphi_fid[2:lmax])) or np.any(cl_phiphi_fid[2:lmax] < 0.0):
            raise ValueError("cl_phiphi_fid must be finite and non-negative for L=2..lmax-1")

    S = compute_sl_phi_np(phi_packed, lmax)
    # alpha = k_L/2 + a0 with k_L the packed vector's REAL dof at L, which is
    # 2L (not 2L+1) because splittosingularalm forces Im(a_{L,1}) = 0. a0 is
    # the prior's InvGamma shape: -1 for the flat improper default, nu/2 for
    # the proper conjugate prior. Derived from the packing, not hardcoded.
    a0 = (prior_nu * 0.5) if use_proper_prior else -1.0
    alpha_L = invgamma_shape_for_spectrum(lmax, a0=a0)
    lncl = np.empty(lmax - 2)
    for i in range(lmax - 2):
        L = i + 2
        alpha = float(alpha_L[L])
        s_val = S[L]
        if not np.isfinite(s_val) or s_val < 0.0:
            s_val = 0.0
        if use_proper_prior:
            # beta_0 = nu*C_L^fid/2, so beta = (S_L + nu*C_L^fid)/2 below.
            s_val = s_val + prior_nu * float(cl_phiphi_fid[L])
        beta = max(s_val * 0.5, 1e-60)
        g = rng.gamma(alpha, scale=1.0)
        val_cl = beta / max(g, 1e-300)
        val_cl = np.clip(val_cl, 1e-30, 1e10)
        lncl[i] = np.log(val_cl)
    return lncl


# ---------------------------------------------------------------------------
# Ancillary (non-centred) joint (phi, C_L^phiphi) rescaling move
#
# Blocks 3 and 4 together are a CENTRED parameterisation of an
# amplitude-and-variance model: Block 3 moves phi at fixed C_L^phiphi, Block 4
# draws C_L^phiphi | phi. That pair funnels, and the funnel is measured, not
# hypothesised -- job 11836793 (2026-08-24) recorded worst lag-1
# autocorrelation 0.945 with Block 4 on against 0.557 with it off at an
# otherwise identical lmax=64 configuration.
#
# The move below is the ANCILLARY half of an ancillary-sufficient interweaving
# strategy (Yu & Meng; the "ancillary vs sufficient reparameterisation" that is
# Millea, Anderes & Wandelt 2020's central methodological result,
# arXiv:2002.00965; the direct analogue of Racine et al. 2016's joint move for
# the (a_lm, C_l) Gibbs funnel, arXiv:1512.06619). It holds the non-centred
# variable xi = phi / sqrt(C) fixed and slides along the funnel axis:
#
#     phi -> alpha_L * phi ,    C_L -> alpha_L^2 * C_L
#
# Interweaving rather than replacing is deliberate: Block 4's exact
# inverse-Gamma draw and Block 3's HMC are both already validated, and this
# composes with them instead of invalidating either. Full non-centring would
# destroy Block 4's conjugacy, since the likelihood would then depend on C.
# ---------------------------------------------------------------------------

class PhiRescaleMove(NamedTuple):
    """Outcome of one ancillary rescaling move.

    `phi`/`cl_phiphi` are the state to CARRY FORWARD (the proposal if accepted,
    otherwise the unchanged current state). `phi_proposed`/`cl_phiphi_proposed`
    are always the proposal, kept for diagnostics and tests. `log_alpha` is the
    per-multipole log rescaling actually drawn (index 0 is L=2), which lets a
    driver tune `proposal_scale` against the observed acceptance rate.
    """

    phi: np.ndarray
    cl_phiphi: np.ndarray
    accepted: bool
    log_accept_ratio: float
    log_alpha: np.ndarray
    phi_proposed: np.ndarray
    cl_phiphi_proposed: np.ndarray


def _packed_coord_multipole(lmax: int) -> np.ndarray:
    """Multipole L of every coordinate in the packed phi layout.

    Traverses exactly as compute_sl_phi_np does: real parts for L=2..lmax-1,
    m=0..L, then imaginary parts for m>=1.
    """
    L_arr = []
    for L in range(2, lmax):
        L_arr.extend([L] * (L + 1))
    for L in range(2, lmax):
        L_arr.extend([L] * L)
    return np.array(L_arr, dtype=np.int64)


def sample_phi_amplitude_rescale(
    phi_packed: np.ndarray,
    cl_phiphi_full: np.ndarray,
    lmax: int,
    neg_log_lik_fn,
    rng=None,
    proposal_scale: float = 0.1,
) -> PhiRescaleMove:
    """One Metropolis-Hastings ancillary rescaling move on (phi, C_L^phiphi).

    Proposes ln alpha_L ~ N(0, proposal_scale^2) independently per multipole
    and maps (phi, C) -> (alpha*phi, alpha^2*C). Because
    S_L(alpha*phi) = alpha^2 * S_L(phi), the Gaussian prior exponent
    S_L/(2 C_L) is exactly invariant, so the move slides along the funnel axis
    that the centred Block 3 / Block 4 alternation mixes slowly across.

    Acceptance ratio. The composite chain must target the same joint density
    Block 4's exact conditional already implies. `sample_cl_phiphi_given_phi`
    draws C_L | phi ~ InvGamma(L - 0.5, S_L/2), which corresponds to

        log p(phi, C) = -psi(phi)
                        - 0.5 * sum_L [ S_L(phi)/C_L + (2L+1) ln C_L ]

    up to a constant -- i.e. a flat improper prior on C_L. The packed phi
    vector holds n_L = (L+1) real + L imaginary = 2L+1 coordinates at multipole
    L, so the map's Jacobian is prod_L alpha_L^(2L+1) for phi times alpha_L^2
    for C_L. The invariant S_L/C_L exponent cancels and the C_L normalisation
    contributes -(2L+1) ln alpha_L, leaving

        log A = -[psi(phi') - psi(phi)]
                + sum_L [ -(2L+1) + (2L+1) + 2 ] ln alpha_L
              = -[psi(phi') - psi(phi)] + 2 * sum_L ln alpha_L

    (verified against a brute-force target evaluation in
    tests/test_phi_ancillary_move.py rather than trusted from the derivation).

    Note the (2L+1) normalisation now agrees with BOTH Block 4's alpha =
    k_L/2 - 1 = L - 0.5 and the packed vector's own 2L+1 coordinates; before
    Im(a_{L,1}) was restored the two disagreed and the coefficient was 1, not 2.

    Parameters
    ----------
    phi_packed      : (n_real+n_imag,) current phi, packed real+imag layout.
    cl_phiphi_full  : (lmax,) current phi power spectrum; entries below L=2 are
                      ignored, matching log_prob_phi_block.
    lmax            : maximum multipole.
    neg_log_lik_fn  : callable phi_packed -> float, the NEGATIVE log likelihood
                      (psi_lensed's sign convention). Must not include the phi
                      Gaussian prior -- that term is handled analytically here
                      and double-counting it would bias the move.
    rng             : numpy Generator (default: fresh default_rng()).
    proposal_scale  : sd of ln alpha_L. 0.0 makes the move an accepted no-op.

    Returns
    -------
    PhiRescaleMove
    """
    if rng is None:
        rng = np.random.default_rng()
    phi_packed = np.asarray(phi_packed, dtype=np.float64)
    cl_phiphi_full = np.asarray(cl_phiphi_full, dtype=np.float64)
    if np.any(~np.isfinite(phi_packed)):
        raise ValueError(
            "Non-finite values (NaNs/Infs) detected in phi_packed during "
            "sample_phi_amplitude_rescale!"
        )
    if proposal_scale < 0.0:
        raise ValueError(f"proposal_scale must be >= 0, got {proposal_scale}")

    n_L = lmax - 2

    if proposal_scale == 0.0:
        # The proposal is deterministically the identity, so MH accepts with
        # probability 1 and no random draw is needed. Short-circuiting (rather
        # than drawing and discarding) is what makes proposal_scale=0 leave the
        # driver's RNG stream untouched, so a chain with the move enabled at
        # scale 0 is bit-identical to the same chain with it disabled -- the
        # regression test that proves the move perturbs state only through its
        # own accept/reject.
        log_alpha = np.zeros(n_L)
        return PhiRescaleMove(
            phi=phi_packed,
            cl_phiphi=cl_phiphi_full,
            accepted=True,
            log_accept_ratio=0.0,
            log_alpha=log_alpha,
            phi_proposed=phi_packed,
            cl_phiphi_proposed=cl_phiphi_full,
        )

    log_alpha = rng.normal(0.0, proposal_scale, size=n_L)
    alpha = np.exp(log_alpha)

    L_arr = _packed_coord_multipole(lmax)
    phi_new = phi_packed * alpha[L_arr - 2]
    cl_new = cl_phiphi_full.copy()
    cl_new[2:lmax] = cl_phiphi_full[2:lmax] * alpha ** 2

    delta_psi = float(neg_log_lik_fn(phi_new)) - float(neg_log_lik_fn(phi_packed))
    log_accept_ratio = -delta_psi + 2.0 * float(np.sum(log_alpha))

    # A non-finite ratio (an overflowing or NaN likelihood at the proposal)
    # rejects rather than propagating garbage into the chain.
    accepted = bool(
        np.isfinite(log_accept_ratio)
        and np.log(rng.uniform()) < log_accept_ratio
    )

    return PhiRescaleMove(
        phi=phi_new if accepted else phi_packed,
        cl_phiphi=cl_new if accepted else cl_phiphi_full,
        accepted=accepted,
        log_accept_ratio=log_accept_ratio,
        log_alpha=log_alpha,
        phi_proposed=phi_new,
        cl_phiphi_proposed=cl_new,
    )


# ---------------------------------------------------------------------------
# Step 1 — phi_alm → deflection field
# ---------------------------------------------------------------------------

# Sign of the deflection relative to +grad(phi). healpy/ducc spin-1 synthesis of
# the E-mode glm = +sqrt(l(l+1)) phi_lm returns (+d phi/d theta,
# +d phi/d varphi / sin theta), i.e. +grad(phi) (checked against
# hp.alm2map_der1 in tests/test_lensing_exact.py). The original code used
# glm = -sqrt(l(l+1)) phi_lm, so every chain made with the BILINEAR operator
# lensed by -grad(phi): T~(n) = T(n - grad phi). Self-consistent (simulation
# and sampler share it, so all validation stands) but opposite to the standard
# convention T~(n) = T(n + grad phi), which matters against external phi maps
# or tracers. Found 2026-09-23. The bilinear path keeps the legacy sign so
# saved chains replay exactly; the exact operator (production from
# 2026-09-23) uses the physical sign.
DEFLECTION_SIGN_LEGACY = -1.0
DEFLECTION_SIGN_PHYSICAL = +1.0


def deflection_field(phi_alm_hp: np.ndarray, nside: int, lmax: int,
                     sign: float = DEFLECTION_SIGN_LEGACY):
    """Convert lensing potential alm to deflection angles at every HEALPix pixel.

    Returns sign * grad(phi) (see DEFLECTION_SIGN_*): the legacy default
    -grad(phi) reproduces every bilinear-operator chain; pass
    DEFLECTION_SIGN_PHYSICAL for the standard +grad(phi).

    Parameters
    ----------
    phi_alm_hp : complex array, healpy-ordering alm of the lensing potential φ
    nside       : HEALPix resolution
    lmax        : maximum multipole

    Returns
    -------
    d_theta : (NPIX,) float64  — colatitude deflection [rad]
    d_phi   : (NPIX,) float64  — longitude deflection [rad]
    """
    if hp is None:
        raise ImportError("healpy is required for deflection_field")

    # Infer lmax from the array so this works for both hp.Alm.getsize(lmax) and
    # lmax*(lmax+1)//2 (our packed-format size, which equals hp.Alm.getsize(lmax-1)).
    lmax_hp = hp.Alm.getlmax(phi_alm_hp.size)
    ells = np.arange(lmax_hp + 1, dtype=float)
    grad_weight = np.sqrt(ells * (ells + 1))
    grad_weight[:2] = 0.0

    glm = hp.almxfl(phi_alm_hp.astype(complex), sign * grad_weight)
    blm = np.zeros_like(glm)

    # Spin-1 SHT: (Q, U) = alm2map_spin([E-alm, B-alm])
    # Q corresponds to the colatitude component, U sinθ × longitude component.
    d_theta, d_phi_sinTheta = hp.alm2map_spin([glm, blm], nside, 1, lmax_hp)

    theta_pix, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    sin_theta = np.clip(np.sin(theta_pix), 1e-10, None)
    d_phi = d_phi_sinTheta / sin_theta

    return d_theta.astype(np.float64), d_phi.astype(np.float64)


def _deflection_field_packed(phi_packed: np.ndarray, nside: int, lmax: int):
    """deflection_field but from a packed phi_alm vector."""
    return deflection_field(_alm_packed_to_hp(phi_packed, lmax), nside, lmax)


# ---------------------------------------------------------------------------
# Adjoint of the deflection: (g_θ, g_φ) on full sky → g_phi_alm (packed)
# ---------------------------------------------------------------------------

def _deflection_adjoint(
    g_theta_full: np.ndarray,
    g_phi_full: np.ndarray,
    nside: int,
    lmax: int,
    sign: float = DEFLECTION_SIGN_LEGACY,
) -> np.ndarray:
    """Backward pass through deflection_field.

    Forward:  phi_alm → glm = sign·√(l(l+1))·phi_lm  → (d_θ, sinθ·d_φ) via alm2map_spin
    Adjoint:  (g_θ, g_φ) → g_glm via map2alm_spin → g_phi_alm = sign·√(l(l+1))·g_glm

    Parameters
    ----------
    g_theta_full : (NPIX,) upstream gradient w.r.t. d_theta
    g_phi_full   : (NPIX,) upstream gradient w.r.t. d_phi
    nside, lmax  : HEALPix parameters

    Returns
    -------
    packed gradient w.r.t. phi_alm, shape (n_real + n_imag,)
    """
    if hp is None:
        raise ImportError("healpy is required for _deflection_adjoint")

    # Convert from (g_theta, g_phi) to (Q-map, U-map) for map2alm_spin.
    # Forward: Q = d_theta, U = sinθ·d_phi  →  d_phi = U/sinθ
    # Adjoint of d_phi = U/sinθ: g_U = g_phi / sinθ  (chain rule, ∂d_phi/∂U = 1/sinθ)
    theta_pix, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    sin_theta = np.clip(np.sin(theta_pix), 1e-10, None)

    g_Q = g_theta_full.astype(np.float64)
    g_U = (g_phi_full / sin_theta).astype(np.float64)

    # Spin-1 SHT adjoint (map2alm_spin).
    # Use lmax-1 so the output alm has size lmax*(lmax+1)//2, matching _alm_hp_to_packed.
    # map2alm_spin includes the 4π/Npix quadrature weight, so it is the adjoint of
    # alm2map_spin in the area-weighted inner product.  For the plain pixel-sum inner
    # product used by the loss, we need an extra Npix/(4π) factor.
    lmax_hp = lmax - 1
    g_glm, _ = hp.map2alm_spin([g_Q, g_U], 1, lmax=lmax_hp)
    # map2alm_spin includes the 4π/Npix quadrature weight; invert it to get
    # the bare transpose that matches the pixel-sum inner product in the loss.
    npix = hp.nside2npix(nside)
    g_glm = g_glm * (npix / (4.0 * np.pi))
    # alm2map_spin sums m and −m, giving a factor-of-2 for m>0 modes.
    # map2alm_spin does not compensate for this, so double the m>0 entries.
    # In healpy ordering, m=0 modes occupy indices 0..lmax_hp; m>0 start after.
    g_glm[lmax_hp + 1:] *= 2.0

    # Adjoint of glm = −√(l(l+1)) · phi_lm
    ells = np.arange(lmax_hp + 1, dtype=float)
    grad_weight = np.sqrt(ells * (ells + 1))
    grad_weight[:2] = 0.0
    g_phi_alm_hp = hp.almxfl(g_glm, sign * grad_weight)

    return _alm_hp_to_packed(g_phi_alm_hp, lmax)


# ---------------------------------------------------------------------------
# Step 2 — precompute neighbor structure (pure numpy, called once per φ draw)
# ---------------------------------------------------------------------------

def precompute_lensing(
    phi_alm_hp: np.ndarray,
    nside: int,
    lmax: int,
    pixel_indices: np.ndarray,
):
    """Compute HEALPix bilinear interpolation structure for a given φ.

    Parameters
    ----------
    phi_alm_hp    : lensing potential alm (healpy ordering)
    nside         : HEALPix resolution
    lmax          : maximum multipole
    pixel_indices : 1-D int array of pixel indices to lens (e.g. model.unmasked_idx)

    Returns
    -------
    neighbors : int32 array (4, n_pix)   — HEALPix neighbor pixel indices
    weights   : float64 array (4, n_pix) — bilinear weights (sum to 1 per pixel)
    d_theta   : float64 array (n_pix,)   — deflection in θ [rad]
    d_phi     : float64 array (n_pix,)   — deflection in φ [rad]
    """
    if hp is None:
        raise ImportError("healpy is required for precompute_lensing")

    d_theta_full, d_phi_full = deflection_field(phi_alm_hp, nside, lmax)

    theta0, phi0 = hp.pix2ang(nside, pixel_indices)
    theta_lensed = np.clip(theta0 + d_theta_full[pixel_indices], 1e-12, np.pi - 1e-12)
    phi_lensed = phi0 + d_phi_full[pixel_indices]

    neighbors, weights = hp.get_interp_weights(nside, theta_lensed, phi_lensed)
    return (
        neighbors.astype(np.int32),
        weights.astype(np.float64),
        d_theta_full[pixel_indices].astype(np.float64),
        d_phi_full[pixel_indices].astype(np.float64),
    )


# ---------------------------------------------------------------------------
# Bilinear weight derivatives — analytic, not finite-differenced
# ---------------------------------------------------------------------------
#
# Bug history (Phase 2, ROADMAP.md): the original implementation got
# dw_k/d_theta' and dw_k/d_phi' by re-invoking hp.get_interp_weights at
# theta'+-eps, phi'+-eps and finite-differencing the returned weights. Away
# from the poles hp.get_interp_weights is a smooth, well-defined bilinear
# function of (theta', phi') within a fixed (neighbors, weights) cell — but
# any FD scheme that *re-queries* it at a shifted point risks the shifted
# point landing in a neighboring cell (different discrete neighbor set),
# producing a wild, step-size-dependent value at exactly the points where
# the shift crosses a cell boundary. tests/test_lensing.py found up to 81%
# of dL/dphi_alm components affected — not because 81% of pixels are near a
# boundary, but because the handful of genuinely bad pixels get smeared
# across nearly every phi_alm mode by the global spin-1 SHT adjoint
# (_deflection_adjoint) that follows.
#
# Fix: hp.get_interp_weights' bilinear scheme is, for any single query
# point, exactly two nested linear interpolations —
#   v  = (theta' - theta_ring1) / (theta_ring2 - theta_ring1)
#   u1 = (phi' - phi_ring1_a) / (phi_ring1_b - phi_ring1_a)   [ring "above"]
#   u2 = (phi' - phi_ring2_a) / (phi_ring2_b - phi_ring2_a)   [ring "below"]
#   w = [(1-v)(1-u1), (1-v)u1, v(1-u2), v*u2]
# — verified to reproduce hp.get_interp_weights' own output to ~1e-14 (away
# from the poles) by direct comparison. Differentiating this closed form
# analytically, using ONLY the single (neighbors, weights) already returned
# for the exact query point, never requires re-querying at a shifted angle,
# so it cannot cross into a different cell and cannot blow up. v, u1, u2
# are recovered directly from the returned weights (v = w2a+w2b, etc.)
# rather than recomputed from angles, so no phi-wraparound logic is needed.
#
# The one genuine exception: within the single ring closest to each pole,
# hp.get_interp_weights collapses "ring above" and "ring below" onto the
# same ring (theta_ring1 == theta_ring2, confirmed by direct inspection —
# ~1.5% of pixels per pole at NSIDE=16, scaling as ~1/(4*NSIDE) of the sky),
# and uses a different, non-bilinear scheme there. dv/dtheta is singular in
# that band; this function returns a zero gradient there (documented,
# bounded, and confined to a thin polar annulus) rather than attempting to
# reverse-engineer HEALPix's internal polar-cap interpolation.

def _bilinear_weight_grads(
    phi_packed: np.ndarray,
    nside: int,
    lmax: int,
    pixel_indices: np.ndarray,
    eps: float = 1e-6,  # unused, kept for backward-compatible call signature
):
    """Compute dw_k/d_theta' and dw_k/d_phi' analytically (see module note above).

    Returns
    -------
    dw_dtheta : (4, n_pix) float64
    dw_dphi   : (4, n_pix) float64
    neighbors : (4, n_pix) int32  — center-phi neighbor indices
    weights   : (4, n_pix) float64 — center-phi bilinear weights
    theta_lensed : (n_pix,) float64 — lensed colatitudes
    phi_lensed   : (n_pix,) float64 — lensed longitudes
    """
    if hp is None:
        raise ImportError("healpy is required for _bilinear_weight_grads")

    phi_alm_hp = _alm_packed_to_hp(phi_packed, lmax)
    d_theta_full, d_phi_full = deflection_field(phi_alm_hp, nside, lmax)

    theta0, phi0 = hp.pix2ang(nside, pixel_indices)
    theta_lensed = np.clip(theta0 + d_theta_full[pixel_indices], 1e-12, np.pi - 1e-12)
    phi_lensed = phi0 + d_phi_full[pixel_indices]

    neighbors, weights = hp.get_interp_weights(nside, theta_lensed, phi_lensed)
    dw_dtheta, dw_dphi = _analytic_bilinear_weight_grads(nside, neighbors, weights)

    return (
        dw_dtheta,
        dw_dphi,
        neighbors.astype(np.int32),
        weights.astype(np.float64),
        theta_lensed,
        phi_lensed,
    )


def _analytic_bilinear_weight_grads(
    nside: int, neighbors: np.ndarray, weights: np.ndarray, return_degenerate: bool = False,
):
    """dw_k/d_theta', dw_k/d_phi' from a single (neighbors, weights) query result.

    neighbors, weights : (4, n_pix) as returned by hp.get_interp_weights, in
    the documented order [ring1_a, ring1_b, ring2_a, ring2_b] (ring1 = the
    ring at or above the query colatitude, ring2 = the ring below;
    verified: neighbors[0] and neighbors[1] always share a colatitude, as
    do neighbors[2] and neighbors[3]).

    Returns (dw_dtheta, dw_dphi), each (4, n_pix) float64, zero in the thin
    polar annulus where the two rings collapse (see module note above). If
    `return_degenerate`, also returns the (n_pix,) bool mask of that annulus
    so callers can substitute a different (e.g. FD-based) derivative there —
    empirically stable there (checked over eps spanning 1e-9 to 1e-4), just
    not representable by the two-ring bilinear formula this function
    implements.
    """
    theta_n, phi_n = hp.pix2ang(nside, neighbors)
    theta1, theta2 = theta_n[0], theta_n[2]
    dtheta_ring = theta2 - theta1
    degenerate = np.abs(dtheta_ring) < 1e-9
    safe_dtheta_ring = np.where(degenerate, 1.0, dtheta_ring)

    v = weights[2] + weights[3]
    one_minus_v = weights[0] + weights[1]

    interval1 = (phi_n[1] - phi_n[0]) % (2.0 * np.pi)
    interval2 = (phi_n[3] - phi_n[2]) % (2.0 * np.pi)
    tiny = 1e-12
    has_ring1_extent = interval1 > tiny
    has_ring2_extent = interval2 > tiny

    u1 = np.where(one_minus_v > tiny, weights[1] / np.where(one_minus_v > tiny, one_minus_v, 1.0), 0.0)
    u2 = np.where(v > tiny, weights[3] / np.where(v > tiny, v, 1.0), 0.0)

    dv_dtheta = np.where(degenerate, 0.0, 1.0 / safe_dtheta_ring)
    du1_dphi = np.where(has_ring1_extent, 1.0 / np.where(has_ring1_extent, interval1, 1.0), 0.0)
    du2_dphi = np.where(has_ring2_extent, 1.0 / np.where(has_ring2_extent, interval2, 1.0), 0.0)

    dw_dtheta = np.stack([
        -dv_dtheta * (1.0 - u1),
        -dv_dtheta * u1,
        dv_dtheta * (1.0 - u2),
        dv_dtheta * u2,
    ])
    dw_dphi = np.stack([
        -one_minus_v * du1_dphi,
        one_minus_v * du1_dphi,
        -v * du2_dphi,
        v * du2_dphi,
    ])

    dw_dtheta[:, degenerate] = 0.0
    dw_dphi[:, degenerate] = 0.0

    if return_degenerate:
        return dw_dtheta.astype(np.float64), dw_dphi.astype(np.float64), degenerate
    return dw_dtheta.astype(np.float64), dw_dphi.astype(np.float64)


# ---------------------------------------------------------------------------
# Step 3 — differentiable lensing application (alm gradient only)
# ---------------------------------------------------------------------------

def apply_lensing_tf(
    T_map_full: "tf.Tensor",
    neighbors_tf: "tf.Tensor",
    weights_tf: "tf.Tensor",
):
    """Bilinear lensing interpolation, differentiable w.r.t. T_map_full.

    T_lensed[i] = Σ_k weights[k,i] * T_map_full[neighbors[k,i]]

    Parameters
    ----------
    T_map_full   : float64 tensor (NPIX,) — unlensed map on full sphere
    neighbors_tf : int32 tensor (4, n_pix)
    weights_tf   : float64 tensor (4, n_pix)

    Returns
    -------
    T_lensed : float64 tensor (n_pix,)
    """
    if tf is None:
        raise ImportError("tensorflow is required for apply_lensing_tf")

    n_pix = tf.shape(neighbors_tf)[1]
    npix_full = tf.shape(T_map_full)[0]

    @tf.custom_gradient
    def _lens(T_in):
        T_lensed = tf.zeros(n_pix, dtype=tf.float64)
        for k in range(4):
            T_lensed = T_lensed + weights_tf[k] * tf.gather(T_in, neighbors_tf[k])

        def grad(upstream):
            g = tf.zeros(npix_full, dtype=tf.float64)
            for k in range(4):
                g = g + tf.math.unsorted_segment_sum(
                    weights_tf[k] * upstream,
                    neighbors_tf[k],
                    num_segments=npix_full,
                )
            return g

        return T_lensed, grad

    return _lens(T_map_full)


# ---------------------------------------------------------------------------
# Step 4 — lensing differentiable w.r.t. phi_alm (custom_gradient)
# ---------------------------------------------------------------------------

def lens_map_phi_diff_tf(
    T_map_full_tf: "tf.Tensor",
    phi_packed_tf: "tf.Tensor",
    nside: int,
    lmax: int,
    pixel_indices: np.ndarray,
):
    """Bilinear lensing differentiable w.r.t. both T_map_full and phi_alm.

    The phi_alm gradient uses a custom backward pass:
        upstream → dL/d(bilinear weights)  [analytic, see _analytic_bilinear_weight_grads]
                 → dL/d(deflection field)   [scatter back to full sky]
                 → dL/d(phi_alm)            [spin-1 SHT adjoint]

    Parameters
    ----------
    T_map_full_tf : float64 tensor (NPIX,) — unlensed map on full sphere
    phi_packed_tf : float64 tensor (n_phi,) — lensing potential in packed real+imag format
    nside, lmax   : HEALPix parameters
    pixel_indices : (n_unmasked,) int array

    Returns
    -------
    T_lensed : float64 tensor (n_unmasked,)

    Notes
    -----
    Graph-traceable (Phase 1.5, ROADMAP.md): every escape hatch into
    numpy/healpy — the initial bilinear-geometry precompute *and* the
    backward pass — goes through `tf.py_function` rather than a bare
    `.numpy()` call, so this op survives being embedded inside a
    `tf.function`-traced graph (same pattern as `sht_ducc.py`'s
    `masked_synthesis_tf`).

    Bug fix (Phase 2, ROADMAP.md): the backward pass previously computed
    dL/d(theta_lensed) and dL/d(phi_lensed) by *re-invoking*
    hp.get_interp_weights at theta'+-eps, phi'+-eps and finite-differencing
    the interpolated T value. Away from the poles this is exactly the
    failure mode _analytic_bilinear_weight_grads fixes (re-querying risks
    landing in a different bilinear cell). Within the thin polar annulus
    where HEALPix's own scheme is genuinely non-bilinear, that same FD is
    actually fine (checked stable across eps spanning 1e-9 to 1e-4 there) —
    it was the wrong tool everywhere else, not there. So: analytic
    everywhere except the polar annulus, and the original small-eps FD
    (now correctly scoped to only that annulus) as the fallback. Whichever
    pixels this affects, the global spin-1 SHT adjoint no longer smears bad
    values from the wrong-tool cases across most phi_alm modes (previously
    up to 81% of components mismatched vs FD in the regression test).
    """
    if tf is None:
        raise ImportError("tensorflow is required for lens_map_phi_diff_tf")
    if hp is None:
        raise ImportError("healpy is required for lens_map_phi_diff_tf")

    pixel_indices = np.asarray(pixel_indices)
    npix_full = 12 * nside * nside
    n_unmasked = len(pixel_indices)

    def _geometry_np(T_map, phi_packed):
        T_np = T_map.numpy()
        phi_alm_hp = _alm_packed_to_hp(phi_packed.numpy(), lmax)
        d_theta_full, d_phi_full = deflection_field(phi_alm_hp, nside, lmax)
        theta0, phi0 = hp.pix2ang(nside, pixel_indices)
        theta_lensed = np.clip(theta0 + d_theta_full[pixel_indices], 1e-12, np.pi - 1e-12)
        phi_lensed = phi0 + d_phi_full[pixel_indices]
        neighbors, weights = hp.get_interp_weights(nside, theta_lensed, phi_lensed)

        dw_dtheta, dw_dphi, degenerate = _analytic_bilinear_weight_grads(
            nside, neighbors, weights, return_degenerate=True
        )
        T_neighbors = T_np[neighbors]  # (4, n_unmasked)
        dT_dtheta = np.sum(dw_dtheta * T_neighbors, axis=0)
        dT_dphi = np.sum(dw_dphi * T_neighbors, axis=0)

        if np.any(degenerate):
            # Thin polar annulus (see _analytic_bilinear_weight_grads):
            # re-query at a small angular offset, but only here, where FD is
            # empirically stable rather than boundary-crossing-prone.
            eps_angle = 1e-7
            th_p = np.clip(theta_lensed + eps_angle, 1e-12, np.pi - 1e-12)
            th_m = np.clip(theta_lensed - eps_angle, 1e-12, np.pi - 1e-12)
            nbrs_tp, wts_tp = hp.get_interp_weights(nside, th_p, phi_lensed)
            nbrs_tm, wts_tm = hp.get_interp_weights(nside, th_m, phi_lensed)
            dT_dtheta_fd = (
                np.sum(T_np[nbrs_tp] * wts_tp, axis=0) - np.sum(T_np[nbrs_tm] * wts_tm, axis=0)
            ) / (2.0 * eps_angle)

            ph_p, ph_m = phi_lensed + eps_angle, phi_lensed - eps_angle
            nbrs_pp, wts_pp = hp.get_interp_weights(nside, theta_lensed, ph_p)
            nbrs_pm, wts_pm = hp.get_interp_weights(nside, theta_lensed, ph_m)
            dT_dphi_fd = (
                np.sum(T_np[nbrs_pp] * wts_pp, axis=0) - np.sum(T_np[nbrs_pm] * wts_pm, axis=0)
            ) / (2.0 * eps_angle)

            dT_dtheta = np.where(degenerate, dT_dtheta_fd, dT_dtheta)
            dT_dphi = np.where(degenerate, dT_dphi_fd, dT_dphi)

        return (
            neighbors.astype(np.int32), weights.astype(np.float64),
            dT_dtheta.astype(np.float64), dT_dphi.astype(np.float64),
        )

    def _backward_np(upstream, neighbors, weights, dT_dtheta, dT_dphi):
        g = upstream.numpy()           # (n_unmasked,)
        neighbors = neighbors.numpy()
        weights = weights.numpy()
        dT_dtheta = dT_dtheta.numpy()
        dT_dphi = dT_dphi.numpy()

        # --- gradient w.r.t. T_map (scatter adjoint) ---
        g_T = np.zeros(npix_full, dtype=np.float64)
        for k in range(4):
            np.add.at(g_T, neighbors[k], weights[k] * g)

        # --- gradient w.r.t. phi_packed ---
        dL_dth = g * dT_dtheta
        dL_dph = g * dT_dphi

        # Scatter to full sky
        dL_dth_full = np.zeros(npix_full, dtype=np.float64)
        dL_dph_full = np.zeros(npix_full, dtype=np.float64)
        dL_dth_full[pixel_indices] = dL_dth
        dL_dph_full[pixel_indices] = dL_dph

        # Propagate through deflection adjoint → packed phi gradient
        g_phi = _deflection_adjoint(dL_dth_full, dL_dph_full, nside, lmax)

        return g_T.astype(np.float64), g_phi.astype(np.float64)

    @tf.custom_gradient
    def _lens(T_map, phi_p):
        neighbors_tf, weights_tf, dT_dtheta_tf, dT_dphi_tf = tf.py_function(
            func=_geometry_np,
            inp=[T_map, phi_p],
            Tout=[tf.int32, tf.float64, tf.float64, tf.float64],
        )
        neighbors_tf.set_shape([4, n_unmasked])
        weights_tf.set_shape([4, n_unmasked])
        dT_dtheta_tf.set_shape([n_unmasked])
        dT_dphi_tf.set_shape([n_unmasked])

        # Forward: bilinear gather
        T_lensed = tf.zeros(n_unmasked, dtype=tf.float64)
        for k in range(4):
            T_lensed = T_lensed + weights_tf[k] * tf.gather(T_map, neighbors_tf[k])

        def backward(upstream):
            # A downstream tf.gather (e.g. the matrix-free path's masked-
            # likelihood restriction of T_lensed) makes TF's autodiff hand
            # this an IndexedSlices rather than a dense tensor; _backward_np
            # calls .numpy() directly, which IndexedSlices doesn't support.
            upstream = tf.convert_to_tensor(upstream)
            g_T, g_phi = tf.py_function(
                func=_backward_np,
                inp=[upstream, neighbors_tf, weights_tf, dT_dtheta_tf, dT_dphi_tf],
                Tout=[tf.float64, tf.float64],
            )
            g_T.set_shape([npix_full])
            g_phi.set_shape(phi_p.shape)
            return g_T, g_phi

        return T_lensed, backward

    return _lens(T_map_full_tf, phi_packed_tf)


# ---------------------------------------------------------------------------
# Step 4b — EXACT lensing: evaluate the band-limited alm at deflected positions
# ---------------------------------------------------------------------------
#
# Why (ROADMAP D3, 2026-09-22/23): the bilinear operator above interpolates the
# unlensed map on the nside grid. At nside = lmax that interpolation smooths the
# map by an amount that grows as (l/nside)^2 and dominates the physical lensing
# effect on C_l^TT at every l the project runs (-5.3% at [100,128) for
# lmax=128 against a physical +0.15%; scripts/validate_lensing_power_transfer.py,
# job 12037446). The exact operator below evaluates
#     T~(n_i) = sum_lm a_lm Y_lm(theta_i + d_theta_i, phi_i + d_phi_i)
# directly with ducc0.sht.synthesis_general (non-uniform SHT, accuracy
# `epsilon`) at the GEODESICALLY deflected positions n' = exp_n(+grad phi)
# (_exact_lensing_geometry), the standard curved-sky remapping. The bilinear
# path uses a first-order coordinate offset and the legacy -grad(phi) sign.
#
# Gradients:
#   d/d a_lm : adjoint_synthesis_general at the same positions, times the
#              m>0 weight 2 -- the same (unconjugated) convention as
#              sht_ducc.full_synthesis_tf, so it composes with the rest of
#              psi_lensed unchanged.
#   d/d phi  : dT~_i/d(deflection) = grad T at n' contracted with the
#              Jacobian of the geodesic map; grad T at n' comes from one spin-1
#              synthesis_general of glm = +sqrt(l(l+1)) a_lm, which returns
#              (dT/dtheta', dT/dphi' / sin theta'). They then go through the
#              existing _deflection_adjoint (spin-1 SHT adjoint) to phi_alm.
# Both are checked against finite differences in tests/test_lensing_exact.py.

EXACT_LENSING_EPSILON = 1e-11


def _unit_basis(theta, phi):
    """n, e_theta, e_phi (each (n, 3)) at (theta, phi)."""
    st, ct, sp, cp = np.sin(theta), np.cos(theta), np.sin(phi), np.cos(phi)
    n = np.stack([st * cp, st * sp, ct], axis=1)
    e_th = np.stack([ct * cp, ct * sp, -st], axis=1)
    e_ph = np.stack([-sp, cp, np.zeros_like(phi)], axis=1)
    return n, e_th, e_ph


def _exact_lensing_geometry(phi_packed, nside, lmax, pixel_indices, with_jacobian=False):
    """Geodesically deflected positions n' = exp_n(grad phi) at pixel_indices.

    The deflection d = a e_theta + b e_phi (a = d_theta, b = sin(theta) d_phi,
    both physical angles) moves each point a distance |d| along the great
    circle in direction d/|d|:
        n' = cos|d| n + (sin|d|/|d|) d.
    This is the standard curved-sky lensing remapping (lenspyx convention); the
    bilinear path's coordinate offset (theta + d_theta, phi + d_phi) agrees to
    first order in |d| but is wrong at O(|d|^2 cot theta), which matters for
    the degree-scale deflections of the amplified-lensing validation runs
    and near the poles.

    Returns loc (n, 2) = (theta', phi'); with_jacobian also returns the basis
    at n' and the derivatives dn'/d(d_theta), dn'/d(d_phi) (d_phi = the
    longitude deflection deflection_field returns), each (n, 3), for the phi
    gradient.
    """
    pixel_indices = np.asarray(pixel_indices)
    phi_alm_hp = _alm_packed_to_hp(np.asarray(phi_packed, dtype=np.float64), lmax)
    d_theta_full, d_phi_full = deflection_field(phi_alm_hp, nside, lmax,
                                                sign=DEFLECTION_SIGN_PHYSICAL)
    theta0, phi0 = hp.pix2ang(nside, pixel_indices)
    sin0 = np.clip(np.sin(theta0), 1e-10, None)     # same clip as deflection_field
    a = d_theta_full[pixel_indices]
    b = d_phi_full[pixel_indices] * sin0
    n, e_th, e_ph = _unit_basis(theta0, phi0)
    alpha = np.hypot(a, b)
    small = alpha < 1e-8
    safe = np.where(small, 1.0, alpha)
    s_fn = np.where(small, 1.0 - alpha ** 2 / 6.0, np.sin(safe) / safe)       # sin a / a
    ds_fn = np.where(small, -alpha / 3.0,
                     (safe * np.cos(safe) - np.sin(safe)) / safe ** 2)       # d/da (sin a / a)
    dvec = a[:, None] * e_th + b[:, None] * e_ph
    n_new = np.cos(alpha)[:, None] * n + s_fn[:, None] * dvec
    n_new /= np.linalg.norm(n_new, axis=1, keepdims=True)
    theta = np.arccos(np.clip(n_new[:, 2], -1.0, 1.0))
    phi = np.mod(np.arctan2(n_new[:, 1], n_new[:, 0]), 2.0 * np.pi)
    loc = np.ascontiguousarray(np.stack([theta, phi], axis=1))
    if not with_jacobian:
        return loc
    # d alpha / d(a, b) = (a, b)/alpha (zero at alpha = 0, where it multiplies zero)
    ua = np.where(small, 0.0, a / safe)
    ub = np.where(small, 0.0, b / safe)
    common = -np.sin(alpha)[:, None] * n + ds_fn[:, None] * dvec
    dn_da = ua[:, None] * common + s_fn[:, None] * e_th
    dn_db = ub[:, None] * common + s_fn[:, None] * e_ph
    dn_ddphi = dn_db * sin0[:, None]                # b = sin(theta0) * d_phi
    _, e_th_new, e_ph_new = _unit_basis(theta, phi)
    return loc, e_th_new, e_ph_new, dn_da, dn_ddphi


def _alm_m_weight(lmax):
    _, ms = hp.Alm.getlm(lmax - 1)
    return np.where(ms == 0, 1.0, 2.0)


def exact_lens_np(alm_ho, phi_packed, nside, lmax, pixel_indices, nthreads=0,
                  with_gradient=False):
    """Exact lensed map at pixel_indices (numpy). Optionally dT/dtheta, dT/dphi there.

    alm_ho : complex healpy-ordered alm of the unlensed field, lmax_hp = lmax-1.
    Returns T (n,) or (T, loc, dT/d(d_theta), dT/d(d_phi)) -- derivatives with
    respect to the deflection components deflection_field returns, through the
    geodesic remapping, ready for _deflection_adjoint.
    """
    import ducc0

    alm_ho = np.ascontiguousarray(alm_ho, dtype=np.complex128)
    geo = _exact_lensing_geometry(phi_packed, nside, lmax, np.asarray(pixel_indices),
                                  with_jacobian=with_gradient)
    loc = geo[0] if with_gradient else geo
    T = ducc0.sht.synthesis_general(
        alm=alm_ho[np.newaxis, :], spin=0, lmax=lmax - 1, loc=loc,
        epsilon=EXACT_LENSING_EPSILON, nthreads=nthreads)[0]
    if not with_gradient:
        return T
    _, e_th_new, e_ph_new, dn_da, dn_ddphi = geo
    ells, _ = hp.Alm.getlm(lmax - 1)
    glm = np.sqrt(ells * (ells + 1.0)) * alm_ho        # +grad T (see DEFLECTION_SIGN_*)
    grad = ducc0.sht.synthesis_general(
        alm=np.ascontiguousarray(np.stack([glm, np.zeros_like(glm)])), spin=1,
        lmax=lmax - 1, loc=loc, epsilon=EXACT_LENSING_EPSILON, nthreads=nthreads)
    g_th, g_ph = grad[0], grad[1]                      # dT/dtheta', dT/dphi' / sin theta'
    # dT = g_th (e_theta'.dn') + g_ph (e_phi'.dn'): stable at the poles of n'.
    dT_dtheta = g_th * np.sum(e_th_new * dn_da, axis=1) + g_ph * np.sum(e_ph_new * dn_da, axis=1)
    dT_dphi = g_th * np.sum(e_th_new * dn_ddphi, axis=1) + g_ph * np.sum(e_ph_new * dn_ddphi, axis=1)
    return T, loc, dT_dtheta, dT_dphi


def lens_alm_exact_tf(alm_ho_tf, phi_packed_tf, nside, lmax, pixel_indices, nthreads=0):
    """Exact lensing, differentiable w.r.t. the unlensed alm AND phi.

    alm_ho_tf     : complex128 tensor, healpy-ordered alm, lmax_hp = lmax-1
    phi_packed_tf : float64 tensor, packed phi alm
    Returns T_lensed : float64 tensor at pixel_indices.
    """
    if tf is None:
        raise ImportError("tensorflow is required for lens_alm_exact_tf")
    import ducc0

    pixel_indices = np.asarray(pixel_indices)
    npix_full = 12 * nside * nside
    n_pix = len(pixel_indices)
    n_alm = hp.Alm.getsize(lmax - 1)
    w_m = _alm_m_weight(lmax)

    def _forward_np(alm_t, phi_t):
        T, loc, dth, dph = exact_lens_np(alm_t.numpy(), phi_t.numpy(), nside, lmax,
                                          pixel_indices, nthreads, with_gradient=True)
        return T, loc, dth, dph

    def _backward_np(g_t, loc_t, dth_t, dph_t):
        g = g_t.numpy().astype(np.float64)
        loc = loc_t.numpy()
        g_alm = ducc0.sht.adjoint_synthesis_general(
            map=np.ascontiguousarray(g[np.newaxis, :]), spin=0, lmax=lmax - 1,
            loc=loc, epsilon=EXACT_LENSING_EPSILON, nthreads=nthreads)[0]
        g_alm = (w_m * g_alm).astype(np.complex128)
        g_th_full = np.zeros(npix_full)
        g_ph_full = np.zeros(npix_full)
        g_th_full[pixel_indices] = g * dth_t.numpy()
        g_ph_full[pixel_indices] = g * dph_t.numpy()
        g_phi = _deflection_adjoint(g_th_full, g_ph_full, nside, lmax,
                                    sign=DEFLECTION_SIGN_PHYSICAL)
        return g_alm, g_phi.astype(np.float64)

    @tf.custom_gradient
    def _lens(alm_t, phi_t):
        T, loc, dth, dph = tf.py_function(
            func=_forward_np, inp=[alm_t, phi_t],
            Tout=[tf.float64, tf.float64, tf.float64, tf.float64])
        T.set_shape([n_pix])
        loc.set_shape([n_pix, 2])
        dth.set_shape([n_pix])
        dph.set_shape([n_pix])

        def backward(upstream):
            upstream = tf.convert_to_tensor(upstream)
            g_alm, g_phi = tf.py_function(
                func=_backward_np, inp=[upstream, loc, dth, dph],
                Tout=[tf.complex128, tf.float64])
            g_alm.set_shape([n_alm])
            g_phi.set_shape(phi_t.shape)
            return tf.cast(g_alm, alm_t.dtype), g_phi

        return T, backward

    return _lens(alm_ho_tf, phi_packed_tf)


# ---------------------------------------------------------------------------
# Step 5 — full alm-differentiable pipeline (phi treated as external numpy)
# ---------------------------------------------------------------------------

def lens_map_tf(model, alm_tf: "tf.Tensor", phi_alm_np: np.ndarray):
    """alm + phi_alm_np → T_lensed, differentiable w.r.t. alm.

    phi_alm is treated as a fixed external parameter (no phi gradient).
    Use lens_map_phi_diff_tf for full joint differentiability.

    Parameters
    ----------
    model       : CosmologyAdvancedSampling (must have _ensure_tf_tensors called)
    alm_tf      : float64 tensor (n_real + n_imag,) — packed CMB alm
    phi_alm_np  : complex float64 array (healpy ordering) — lensing potential

    Returns
    -------
    T_lensed : float64 tensor (n_unmasked,)
    """
    if tf is None:
        raise ImportError("tensorflow is required for lens_map_tf")
    if hp is None:
        raise ImportError("healpy is required for lens_map_tf")

    from .alm_utils import splittosingularalm_tf

    lmax = model.lmax
    nside = model.NSIDE
    n_real = lmax * (lmax + 1) // 2 - 3

    _real_p = alm_tf[:n_real]
    _imag_p = alm_tf[n_real:]
    _a = splittosingularalm_tf(_real_p, _imag_p, lmax)

    if getattr(model, "beam_pixwin_per_l", None) is not None:
        _a = _a * tf.cast(tf.gather(model.beam_pixwin_per_l, model.l_indices), _a.dtype)

    if getattr(model, "lensing_operator", "bilinear") == "exact":
        _a_ho = tf.gather(_a, model._alm_mo_to_ho_idx)
        phi_packed = tf.constant(_alm_hp_to_packed(np.asarray(phi_alm_np), lmax), tf.float64)
        return lens_alm_exact_tf(tf.cast(_a_ho, tf.complex128), phi_packed, nside,
                                 lmax, model.unmasked_idx, model.sht_nthreads)

    if getattr(model, "use_matrixfree_sht", False):
        # alm → full-sky unlensed map directly (ducc0 synthesizes all Npix
        # pixels; no dense Y matrix, no scatter-from-unmasked step needed —
        # the mask is applied to the likelihood post-lensing, not here).
        from .sht_ducc import full_synthesis_tf

        _a_ho = tf.gather(_a, model._alm_mo_to_ho_idx)
        T_full = full_synthesis_tf(tf.cast(_a_ho, tf.complex128), model._sht)
    else:
        from .model import matvec_on_device

        _a_c = model.alm_weights * tf.cast(_a, model.dtype)
        T_parts = []
        for sph_p in model.sph_parts:
            _Ya = 2.0 * tf.math.real(matvec_on_device(sph_p, _a_c))
            T_parts.append(tf.cast(_Ya, tf.float64))
        T_unlensed_unmasked = tf.concat(T_parts, axis=0)  # (n_unmasked,)

        # Scatter unmasked pixels onto full sphere for bilinear interpolation
        npix_full = 12 * nside * nside
        unmasked_idx_tf = tf.constant(model.unmasked_idx, dtype=tf.int32)
        T_full = tf.math.unsorted_segment_sum(
            T_unlensed_unmasked, unmasked_idx_tf, num_segments=npix_full
        )

    neighbors, weights, _, _ = precompute_lensing(
        phi_alm_np, nside, lmax, model.unmasked_idx
    )
    neighbors_tf = tf.constant(neighbors, dtype=tf.int32)
    weights_tf = tf.constant(weights, dtype=tf.float64)

    return apply_lensing_tf(T_full, neighbors_tf, weights_tf)


# ---------------------------------------------------------------------------
# Step 6 — lensed log-posterior (drop-in for model._psi_tf_raw)
# ---------------------------------------------------------------------------

def psi_lensed(
    model,
    params_tf: "tf.Tensor",
    phi_packed_tf: "tf.Tensor",
):
    """Lensed log-posterior: 0.5‖d − T_lensed(alm, φ)‖²_N + prior(alm, C_l).

    Matches the _psi_tf_raw interface: a single params_tf vector
    [lncl (lmax-2) | real_alm | imag_alm] plus the lensing potential.

    Differentiable w.r.t. both params_tf (alm and C_l) and phi_packed_tf.

    Parameters
    ----------
    model         : CosmologyAdvancedSampling (must have _ensure_tf_tensors called)
    params_tf     : float64 tensor [lncl, real_alm, imag_alm] — same layout as _psi_tf_raw
    phi_packed_tf : float64 tensor (n_real+n_imag,) — lensing potential packed alm

    Returns
    -------
    psi : scalar float64 tensor
    """
    if tf is None:
        raise ImportError("tensorflow is required for psi_lensed")

    from .alm_utils import splittosingularalm_tf

    lmax = model.lmax
    nside = model.NSIDE
    n_lncl = lmax - 2
    n_real = lmax * (lmax + 1) // 2 - 3

    # Parse params_tf (same slicing as _psi_tf_raw)
    lncl_raw = tf.cast(params_tf[:n_lncl], tf.float64)
    real_alm = tf.cast(params_tf[n_lncl : n_lncl + n_real], tf.float64)
    imag_alm = tf.cast(params_tf[n_lncl + n_real :], tf.float64)

    lncl_start = tf.zeros(2, tf.float64)
    lncl_full = tf.concat([lncl_start, lncl_raw], axis=0)  # length lmax

    # alm → full-sphere unlensed map. The PRIOR is on the sky alm (_a_sky);
    # only the forward model sees the beam. (Until 2026-09-23 the prior below
    # was computed from the beamed alm, inconsistent with model._psi_tf_raw;
    # harmless while production ran with beam_fwhm_arcmin=None.)
    _a_sky = splittosingularalm_tf(real_alm, imag_alm, lmax)
    _a = _a_sky

    if getattr(model, "beam_pixwin_per_l", None) is not None:
        _a = _a * tf.cast(tf.gather(model.beam_pixwin_per_l, model.l_indices), _a.dtype)

    exact = getattr(model, "lensing_operator", "bilinear") == "exact"
    if getattr(model, "use_matrixfree_sht", False):
        from .sht_ducc import full_synthesis_tf

        _a_ho = tf.gather(_a, model._alm_mo_to_ho_idx)
        if not exact:  # the exact operator consumes the alm directly
            T_full = full_synthesis_tf(tf.cast(_a_ho, tf.complex128), model._sht)
    else:
        from .model import matvec_on_device

        _a_c = model.alm_weights * tf.cast(_a, model.dtype)
        T_parts = []
        for sph_p in model.sph_parts:
            _Ya = 2.0 * tf.math.real(matvec_on_device(sph_p, _a_c))
            T_parts.append(tf.cast(_Ya, tf.float64))
        T_unlensed_unmasked = tf.concat(T_parts, axis=0)

        npix_full = 12 * nside * nside
        unmasked_idx_tf = tf.constant(model.unmasked_idx, dtype=tf.int32)
        T_full = tf.math.unsorted_segment_sum(
            T_unlensed_unmasked, unmasked_idx_tf, num_segments=npix_full
        )

    # Lensed map — differentiable w.r.t. both the alm and phi_packed_tf
    if exact:
        T_lensed = lens_alm_exact_tf(
            tf.cast(_a_ho, tf.complex128), phi_packed_tf, nside, lmax,
            model.unmasked_idx, model.sht_nthreads)
    else:
        T_lensed = lens_map_phi_diff_tf(
            T_full, phi_packed_tf, nside, lmax, model.unmasked_idx
        )

    # Lensed likelihood
    if getattr(model, "use_matrixfree_sht", False):
        T_lensed_masked = tf.gather(T_lensed, tf.constant(model.unmasked_idx, dtype=tf.int32))
        psi_lik = 0.5 * tf.reduce_sum(
            (model.prior_map_masked - T_lensed_masked) ** 2 * model.Ninv_masked
        )
    else:
        psi_lik = tf.constant(0.0, dtype=tf.float64)
        start = 0
        for i, (map_p, ninv_p) in enumerate(zip(  # noqa: B905
            model.prior_map_parts, model.Ninv_parts
        )):
            n = int(model.sph_parts[i].shape[0])
            T_lensed_part = T_lensed[start : start + n]
            psi_lik = psi_lik + 0.5 * tf.reduce_sum(
                (map_p - T_lensed_part) ** 2 * ninv_p
            )
            start += n

    # alm Gaussian prior  0.5 Σ_lm |a_lm|² / C_l
    _abs_a2 = tf.math.real(_a_sky * tf.math.conj(_a_sky))  # fp64, not fp32
    _as = tf.math.unsorted_segment_sum(
        _abs_a2 * model.l_weights, model.l_indices, num_segments=lmax
    )
    psi_prior_alm = 0.5 * tf.reduce_sum(
        tf.cast(_as, tf.float64) / (tf.math.exp(lncl_full) + 1e-30)
    )

    # C_l entropy  Σ_l (l + 0.5) ln C_l
    _l = tf.range(lmax, dtype=tf.float64)
    psi_cl = tf.reduce_sum((_l + 0.5) * lncl_full)

    return psi_lik + psi_prior_alm + psi_cl


# ---------------------------------------------------------------------------
# Phase 2, Block 3 — phi | alm, C_l, d  (lensing potential conditional)
# ---------------------------------------------------------------------------

def log_prob_phi_block(
    model,
    params_tf: "tf.Tensor",
    phi_packed_tf: "tf.Tensor",
    cl_phiphi_full: np.ndarray,
):
    """log p(phi | alm, C_l, d) up to a constant — target for the Block 3 HMC step.

    alm and C_l enter through params_tf and are held fixed (no gradient taken
    w.r.t. them here — that is Block 2's job). Only phi_packed_tf is the HMC
    state; the returned scalar is differentiable w.r.t. it.

    log p(phi | ...) = -psi_lensed(alm, C_l, phi) - phi_prior(phi | C_l^phiphi)

    where the phi prior is the same Gaussian-per-l form used for the alm prior
    in psi_lensed, but keyed off cl_phiphi_full instead of the CMB C_l.

    Parameters
    ----------
    model          : CosmologyAdvancedSampling (must have _ensure_tf_tensors called)
    params_tf      : float64 tensor [lncl, real_alm, imag_alm] — held fixed
    phi_packed_tf  : float64 tensor (n_real+n_imag,) — HMC state
    cl_phiphi_full : float64 array (lmax,) — lensing potential power spectrum,
                     either a fixed LCDM prediction or (later) a jointly
                     sampled Block 4 state. cl_phiphi_full[l] for l < 2 is
                     ignored (monopole/dipole excluded, same as CMB C_l).

    Returns
    -------
    log_prob : scalar float64 tensor, differentiable w.r.t. phi_packed_tf
    """
    if tf is None:
        raise ImportError("tensorflow is required for log_prob_phi_block")

    lmax = model.lmax
    # tf.convert_to_tensor (not tf.constant) so that passing a tf.Variable
    # here — as samplers.py's traced HMC step does, so Block 4's per-sweep
    # spectrum update is visible to the compiled graph rather than baked in
    # at trace time — reads its current value at call time instead of
    # freezing it. Identical behaviour to the old tf.constant for a plain
    # numpy-array cl_phiphi_full (eager callers, existing tests).
    cl_phiphi_tf = tf.cast(tf.convert_to_tensor(cl_phiphi_full), tf.float64)

    neg_log_lik = psi_lensed(model, params_tf, phi_packed_tf)

    _real_p = phi_packed_tf[: lmax * (lmax + 1) // 2 - 3]
    _imag_p = phi_packed_tf[lmax * (lmax + 1) // 2 - 3 :]
    from .alm_utils import splittosingularalm_tf
    _phi_a = splittosingularalm_tf(_real_p, _imag_p, lmax)

    _abs_phi2 = tf.math.real(_phi_a * tf.math.conj(_phi_a))  # fp64, not fp32
    _phi_s = tf.math.unsorted_segment_sum(
        _abs_phi2 * model.l_weights, model.l_indices, num_segments=lmax
    )
    phi_prior_neg_log = 0.5 * tf.reduce_sum(
        tf.cast(_phi_s, tf.float64) / (cl_phiphi_tf + 1e-30)
    )

    return -(neg_log_lik + phi_prior_neg_log)


def estimate_phi_diag_fisher(
    model,
    params_tf: "tf.Tensor",
    phi_packed_tf: "tf.Tensor",
    lmax: int,
    n_probes: int = 8,
    rng=None,
    eps: float = 1e-9,
) -> np.ndarray:
    """Coordinate-sampled diagonal estimate of psi_lensed's likelihood
    curvature w.r.t. phi, averaged per L, for preconditioning the phi HMC
    block (samplers.py::build_phi_posterior_mass_sqrt).

    build_phi_prior_mass_sqrt only uses the prior curvature 1/cl_phiphi,
    ignoring the lensing likelihood's curvature entirely -- this was found
    to leave the phi HMC block barely mixing at production lmax (see
    ROADMAP.md's "Simulation validation" entry): the whitened posterior
    variance is highly non-uniform across L (tight where the likelihood
    dominates at low L, prior-dominated and looser at high L), so a single
    global step size settles at whatever the tightest dimension needs,
    moving almost nowhere in the rest.

    Method: for up to n_probes randomly-sampled packed-phi coordinates at
    each L (all of them if the L has fewer than n_probes modes), estimate
    the diagonal Hessian entry H_ii via a single-sided finite difference of
    the already-validated *analytic* gradient of psi_lensed w.r.t. phi
    (dpsi/dphi is validated against FD in test_psi_lensed_phi_grad_vs_fd),
    perturbing only that one coordinate -- not double backprop through the
    matrix-free SHT's tf.custom_gradient, which does not support
    second-order autodiff and would risk resurrecting the exact FD-gradient
    bug class already fixed once (achievements.md's "phi-block HMC
    gradient"). Average the sampled H_ii within each L (fixed-L curvature
    is physically near-isotropic across m).

    An earlier version of this function used a true Hutchinson estimator
    (a single joint random-direction probe per sample, cheaper in principle
    -- O(n_probes) gradient evals total instead of O(n_probes * n_L_bins)).
    It was abandoned: perturbing all packed-phi modes simultaneously drives
    the aggregate perturbation across the same C0-but-not-C1
    bilinear-interpolation-cell boundary documented in
    test_psi_lensed_phi_grad_vs_fd's eps note, and even after rescaling eps
    to avoid that, the off-diagonal coupling in this Hessian turned out
    large enough that single-digit probe counts gave estimates off by
    orders of magnitude. Per-coordinate sampling reuses the exact
    single-index FD computation already validated to be stable and costs
    more gradient evals, but is run once after a short HMC warm-up, not
    every sweep.

    eps=1e-9 matches the FD step used throughout tests/test_lensing.py's
    phi-gradient checks -- a larger eps can cross a genuine (C0-but-not-C1)
    bilinear-interpolation-cell boundary in the deflection field and make
    the FD estimate itself unstable, not the analytic gradient wrong.

    Returns
    -------
    diag_fisher_per_L : float64 array (lmax,), indexed by L (zero for L<2
        and for any L with no positive sampled estimate -- FD Hessian
        entries can be slightly negative from numerical noise for a
        quantity that's only PSD in the true continuum limit; clipped to
        >=0 before use as a mass-matrix addend).
    """
    if tf is None:
        raise ImportError("tensorflow is required for estimate_phi_diag_fisher")
    if rng is None:
        rng = np.random.default_rng()

    phi_np = (
        phi_packed_tf.numpy() if hasattr(phi_packed_tf, "numpy") else np.asarray(phi_packed_tf)
    )

    def grad_at(phi_val: np.ndarray) -> np.ndarray:
        phi_var = tf.Variable(phi_val, dtype=tf.float64)
        with tf.GradientTape() as tape:
            val = psi_lensed(model, params_tf, phi_var)
        return tape.gradient(val, phi_var).numpy()

    g0 = grad_at(phi_np)

    from .samplers import _alm_index_lm

    n_real, n_imag = packed_sizes(lmax)
    L_arr, _m_arr = _alm_index_lm(lmax, n_real, n_imag)

    diag_fisher_per_L = np.zeros(lmax, dtype=np.float64)
    for L in range(2, lmax):
        idx_at_L = np.where(L_arr == L)[0]
        if len(idx_at_L) == 0:
            continue
        sampled = (
            idx_at_L
            if len(idx_at_L) <= n_probes
            else rng.choice(idx_at_L, size=n_probes, replace=False)
        )
        h_ii = np.empty(len(sampled), dtype=np.float64)
        for k, i in enumerate(sampled):
            phi_p = phi_np.copy()
            phi_p[i] += eps
            g1 = grad_at(phi_p)
            h_ii[k] = (g1[i] - g0[i]) / eps
        diag_fisher_per_L[L] = np.mean(np.maximum(h_ii, 0.0))
    return diag_fisher_per_L


def estimate_phi_block_hessian(
    model,
    params_tf: "tf.Tensor",
    phi_packed_tf: "tf.Tensor",
    lmax: int,
    n_probes: int = 6,
    rng=None,
    eps: float = 1e-9,
) -> dict:
    """Nystrom low-rank approximation of psi_lensed's Hessian w.r.t. phi,
    block-diagonal by m (real and imag channels separately) -- the
    structurally motivated fix for the cross-L Hessian coupling
    diagnose_phi_hessian_coupling.py found significant (achievements.md,
    ROADMAP.md 2026-08-17: mean coupling ratios 0.263/0.621, up to 0.97 at
    individual chain points, between L-bins as far apart as [10,30) and
    [60,lmax)). estimate_phi_diag_fisher's per-L diagonal and
    build_phi_prior_mass_sqrt's prior-only diagonal are both diagonal-in-L
    and cannot represent this by construction, however they are tuned --
    every diagonal-mass-matrix configuration tried failed the equilibration
    gate (achievements.md's "Coverage-test prerequisite findings").

    Grouping by m (not a nearest-neighbour band in L) follows this
    codebase's existing precedent for the analogous problem in the alm
    Block-2 messenger sampler (messenger.py::build_block_cholesky,
    samplers.py::_calibrate_block_AtA): there, same-m coupling in the SHT
    operator's A^T A was found to dominate (>99% of off-diagonal energy,
    scripts/analyze_AtA_structure.py) over cross-m coupling, with magnitude
    roughly flat across the whole L range rather than decaying -- i.e. a
    long-range-in-L, same-m structure, not a local band. The cross-L
    coupling diagnose_phi_hessian_coupling.py found (between bins spanning
    m=0..L for both) is consistent with the same picture, but has not been
    independently decomposed by m for the phi Hessian specifically -- this
    is an assumption carried over from the messenger precedent, to be
    checked against whether the resulting pilot (ROADMAP.md) actually
    clears the equilibration gate.

    Method (Nystrom): for each m-block (indices sharing the same m and
    real/imag channel, size K = number of L's spanning that m), sample
    up to n_probes source coordinates uniformly from the block and, for
    each, perturb it by eps and take the finite difference of the
    already-validated analytic gradient (same eps/single-coordinate-probe
    method as estimate_phi_diag_fisher -- deliberately not a joint
    multi-coordinate probe, for the same bilinear-interpolation-cell-
    boundary-instability reason documented there). This gives n_probes
    exact columns of the block's true (K, K) Hessian restricted to the
    sampled source indices S. The Nystrom approximation
        H_block ≈ H[:, S] @ pinv(H[S, S]) @ H[:, S]^T
    (H[S, S] symmetrized and eigenvalue-clipped for numerical safety
    before pseudo-inversion) reconstructs a rank-<=n_probes PSD estimate
    of the full block from only n_probes gradient evaluations per block --
    same asymptotic cost as estimate_phi_diag_fisher's per-L diagonal
    (n_probes gradient evals per bin), now per m-block instead of per L.

    Returns
    -------
    blocks : dict {(channel, m): (idx, H_nystrom)} -- channel in
        ('real', 'imag'); idx a sorted int array of packed-phi positions
        in the block (packed layout matching samplers.py::_alm_index_lm);
        H_nystrom the (K, K) symmetric PSD Nystrom approximation.
        Blocks with K < 2 (nothing to couple) are omitted.
    """
    if tf is None:
        raise ImportError("tensorflow is required for estimate_phi_block_hessian")
    if rng is None:
        rng = np.random.default_rng()

    phi_np = (
        phi_packed_tf.numpy() if hasattr(phi_packed_tf, "numpy") else np.asarray(phi_packed_tf)
    )

    def grad_at(phi_val: np.ndarray) -> np.ndarray:
        phi_var = tf.Variable(phi_val, dtype=tf.float64)
        with tf.GradientTape() as tape:
            val = psi_lensed(model, params_tf, phi_var)
        return tape.gradient(val, phi_var).numpy()

    g0 = grad_at(phi_np)

    from .samplers import _alm_index_lm

    n_real, n_imag = packed_sizes(lmax)
    L_arr, m_arr = _alm_index_lm(lmax, n_real, n_imag)
    channel_arr = np.array(["real"] * n_real + ["imag"] * n_imag)

    blocks = {}
    for channel in ("real", "imag"):
        chan_mask = channel_arr == channel
        for m in sorted(np.unique(m_arr[chan_mask])):
            idx = np.where(chan_mask & (m_arr == m))[0]
            K = len(idx)
            if K < 2:
                continue
            n_s = min(n_probes, K)
            s_local = rng.choice(K, size=n_s, replace=False)
            s_global = idx[s_local]

            cols = np.empty((K, n_s), dtype=np.float64)
            for k, j in enumerate(s_global):
                phi_p = phi_np.copy()
                phi_p[j] += eps
                g1 = grad_at(phi_p)
                cols[:, k] = (g1[idx] - g0[idx]) / eps

            H_SS = cols[s_local, :]
            H_SS = 0.5 * (H_SS + H_SS.T)
            eigval, eigvec = np.linalg.eigh(H_SS)
            # Regularisation floor relative to this block's own eigenvalue
            # scale, not an absolute constant: psi_lensed's curvature scale
            # varies by many orders of magnitude across L (cl_phiphi spans
            # ~1e-16 to ~1e-8 in typical use), so a fixed absolute floor like
            # 1e-12 is either far too small (letting FD-noise eigenvalues
            # near zero through, then dividing by them -- the bug an earlier
            # version of this hit: reconstructed blocks off by ~1e15x) or far
            # too large (discarding real curvature) depending on scale.
            eig_floor = max(float(np.max(np.abs(eigval))), 1e-300) * 1e-6
            eigval_clipped = np.clip(eigval, eig_floor, None)
            H_SS_pinv = (eigvec * (1.0 / eigval_clipped)) @ eigvec.T
            H_nystrom = cols @ H_SS_pinv @ cols.T
            H_nystrom = 0.5 * (H_nystrom + H_nystrom.T)
            blocks[(channel, int(m))] = (idx, H_nystrom)

    return blocks
