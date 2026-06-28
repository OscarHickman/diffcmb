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

Reference: Lewis & Challinor 2006 (Phys. Rep. 429, 1); Carron & Lewis 2017 (arXiv:1701.01712).
"""

import numpy as np

try:
    import healpy as hp
except ImportError:
    hp = None

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
# ---------------------------------------------------------------------------

def _alm_packed_to_hp(phi_packed: np.ndarray, lmax: int) -> np.ndarray:
    """Packed real+imag → healpy complex alm (length lmax*(lmax+1)//2)."""
    n_real = lmax * (lmax + 1) // 2 - 3
    real_p = phi_packed[:n_real]
    imag_p = phi_packed[n_real:]
    len_alm = lmax * (lmax + 1) // 2
    alm_hp = np.zeros(len_alm, dtype=np.complex128)
    r_idx = 0
    i_idx = 0
    for L in range(2, lmax):
        for m in range(L + 1):
            ho_idx = L * (L + 1) // 2 + m
            if m <= 1:
                alm_hp[ho_idx] = real_p[r_idx]
                r_idx += 1
            else:
                alm_hp[ho_idx] = real_p[r_idx] + 1j * imag_p[i_idx]
                r_idx += 1
                i_idx += 1
    return alm_hp


def _alm_hp_to_packed(alm_hp: np.ndarray, lmax: int) -> np.ndarray:
    """Healpy complex alm → packed real+imag float64 vector."""
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = (lmax - 2) * (lmax - 1) // 2
    real_p = np.zeros(n_real, dtype=np.float64)
    imag_p = np.zeros(n_imag, dtype=np.float64)
    r_idx = 0
    i_idx = 0
    for L in range(2, lmax):
        for m in range(L + 1):
            ho_idx = L * (L + 1) // 2 + m
            if m <= 1:
                real_p[r_idx] = alm_hp[ho_idx].real
                r_idx += 1
            else:
                real_p[r_idx] = alm_hp[ho_idx].real
                imag_p[i_idx] = alm_hp[ho_idx].imag
                r_idx += 1
                i_idx += 1
    return np.concatenate([real_p, imag_p])


# ---------------------------------------------------------------------------
# Step 1 — phi_alm → deflection field
# ---------------------------------------------------------------------------

def deflection_field(phi_alm_hp: np.ndarray, nside: int, lmax: int):
    """Convert lensing potential alm to deflection angles at every HEALPix pixel.

    The deflection d = ∇φ.  In harmonic space the gradient of a scalar is
    a spin-1 E-mode field with alm = −√(l(l+1)) φ_lm.

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

    glm = hp.almxfl(phi_alm_hp.astype(complex), -grad_weight)
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
) -> np.ndarray:
    """Backward pass through deflection_field.

    Forward:  phi_alm → glm = −√(l(l+1))·phi_lm  → (d_θ, sinθ·d_φ) via alm2map_spin
    Adjoint:  (g_θ, g_φ) → g_glm via map2alm_spin → g_phi_alm = −√(l(l+1))·g_glm

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
    g_phi_alm_hp = hp.almxfl(g_glm, -grad_weight)

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
# Bilinear weight derivatives via finite differences of hp.get_interp_weights
# ---------------------------------------------------------------------------

def _bilinear_weight_grads(
    phi_packed: np.ndarray,
    nside: int,
    lmax: int,
    pixel_indices: np.ndarray,
    eps: float = 1e-6,
):
    """Compute dw_k/d_θ' and dw_k/d_φ' via finite differences.

    The bilinear weights from hp.get_interp_weights are continuous functions of the
    lensed position (θ', φ').  For eps << pixel_size (~0.016 rad at NSIDE=64) the
    4 neighbor pixels are stable so centered FD is exact up to O(eps²).

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

    # dw/d_theta: FD in theta', holding phi' fixed
    th_p = np.clip(theta_lensed + eps, 1e-12, np.pi - 1e-12)
    th_m = np.clip(theta_lensed - eps, 1e-12, np.pi - 1e-12)
    _, w_tp = hp.get_interp_weights(nside, th_p, phi_lensed)
    _, w_tm = hp.get_interp_weights(nside, th_m, phi_lensed)
    dw_dtheta = (w_tp - w_tm) / (2.0 * eps)

    # dw/d_phi: FD in phi', holding theta' fixed
    _, w_pp = hp.get_interp_weights(nside, theta_lensed, phi_lensed + eps)
    _, w_pm = hp.get_interp_weights(nside, theta_lensed, phi_lensed - eps)
    dw_dphi = (w_pp - w_pm) / (2.0 * eps)

    return (
        dw_dtheta.astype(np.float64),
        dw_dphi.astype(np.float64),
        neighbors.astype(np.int32),
        weights.astype(np.float64),
        theta_lensed,
        phi_lensed,
    )


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
        upstream → dL/d(bilinear weights)  [via FD of hp.get_interp_weights]
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
    """
    if tf is None:
        raise ImportError("tensorflow is required for lens_map_phi_diff_tf")
    if hp is None:
        raise ImportError("healpy is required for lens_map_phi_diff_tf")

    # Precompute bilinear geometry from current phi (numpy)
    phi_np = phi_packed_tf.numpy()
    _, _, neighbors, weights, theta_lensed, phi_lensed = _bilinear_weight_grads(
        phi_np, nside, lmax, pixel_indices
    )
    neighbors_c = tf.constant(neighbors, dtype=tf.int32)
    weights_c = tf.constant(weights, dtype=tf.float64)
    npix_full = 12 * nside * nside
    n_unmasked = len(pixel_indices)

    @tf.custom_gradient
    def _lens(T_map, phi_p):
        # Forward: bilinear gather
        T_lensed = tf.zeros(n_unmasked, dtype=tf.float64)
        for k in range(4):
            T_lensed = T_lensed + weights_c[k] * tf.gather(T_map, neighbors_c[k])

        def backward(upstream):
            g = upstream.numpy()           # (n_unmasked,)
            T_np = T_map.numpy()           # current T_map values

            # --- gradient w.r.t. T_map (scatter adjoint) ---
            g_T = np.zeros(npix_full, dtype=np.float64)
            for k in range(4):
                np.add.at(g_T, neighbors[k], weights[k] * g)

            # --- gradient w.r.t. phi_packed ---
            # dL/d(theta_lensed) and dL/d(phi_lensed) at each output pixel.
            # Use scalar bilinear FD: evaluate T(θ'±ε,φ') as a single number per
            # pixel so that any neighbor-reordering in hp.get_interp_weights cancels.
            # eps_angle must be small enough to stay within the current HEALPix
            # cell (~0.064 rad at NSIDE=16) to avoid crossing cell boundaries
            # where the bilinear derivative is discontinuous.  1e-7 is far
            # below that threshold while maintaining float64 precision.
            eps_angle = 1e-7
            th_p = np.clip(theta_lensed + eps_angle, 1e-12, np.pi - 1e-12)
            th_m = np.clip(theta_lensed - eps_angle, 1e-12, np.pi - 1e-12)
            nbrs_tp, wts_tp = hp.get_interp_weights(nside, th_p, phi_lensed)
            nbrs_tm, wts_tm = hp.get_interp_weights(nside, th_m, phi_lensed)
            T_tp = np.sum(T_np[nbrs_tp] * wts_tp, axis=0)
            T_tm = np.sum(T_np[nbrs_tm] * wts_tm, axis=0)
            dL_dth = g * (T_tp - T_tm) / (2.0 * eps_angle)

            ph_p = phi_lensed + eps_angle
            ph_m = phi_lensed - eps_angle
            nbrs_pp, wts_pp = hp.get_interp_weights(nside, theta_lensed, ph_p)
            nbrs_pm, wts_pm = hp.get_interp_weights(nside, theta_lensed, ph_m)
            T_pp = np.sum(T_np[nbrs_pp] * wts_pp, axis=0)
            T_pm = np.sum(T_np[nbrs_pm] * wts_pm, axis=0)
            dL_dph = g * (T_pp - T_pm) / (2.0 * eps_angle)

            # Scatter to full sky
            dL_dth_full = np.zeros(npix_full, dtype=np.float64)
            dL_dph_full = np.zeros(npix_full, dtype=np.float64)
            dL_dth_full[pixel_indices] = dL_dth
            dL_dph_full[pixel_indices] = dL_dph

            # Propagate through deflection adjoint → packed phi gradient
            g_phi = _deflection_adjoint(dL_dth_full, dL_dph_full, nside, lmax)

            return (
                tf.constant(g_T, dtype=tf.float64),
                tf.constant(g_phi, dtype=tf.float64),
            )

        return T_lensed, backward

    return _lens(T_map_full_tf, phi_packed_tf)


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
    from .model import matvec_on_device

    lmax = model.lmax
    nside = model.NSIDE
    n_real = lmax * (lmax + 1) // 2 - 3

    # alm → unlensed map on unmasked pixels via Y matrix
    _real_p = alm_tf[:n_real]
    _imag_p = alm_tf[n_real:]
    _a = splittosingularalm_tf(_real_p, _imag_p, lmax)
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
    from .model import matvec_on_device

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

    # alm → full-sphere unlensed map
    _a = splittosingularalm_tf(real_alm, imag_alm, lmax)
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

    # Lensed map — differentiable w.r.t. both T_full (→ alm) and phi_packed_tf
    T_lensed = lens_map_phi_diff_tf(
        T_full, phi_packed_tf, nside, lmax, model.unmasked_idx
    )

    # Lensed likelihood
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
    _abs_a2 = tf.cast(tf.math.abs(_a), tf.float32) ** 2
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
