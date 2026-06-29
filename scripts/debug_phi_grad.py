"""Debug script: isolate where the phi gradient chain breaks.

Tests each sub-adjoint piece independently to find the root cause of the
171x mismatch in test_phi_grad_deflection_adjoint_vs_fd.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import healpy as hp

LMAX = 20
NSIDE = 16
NPIX = 12 * NSIDE * NSIDE
lmax_hp = LMAX - 1  # = 19

rng = np.random.default_rng(42)

print(f"NSIDE={NSIDE}, LMAX={LMAX}, lmax_hp={lmax_hp}, NPIX={NPIX}")
print(f"4pi/NPIX = {4*np.pi/NPIX:.6f},  NPIX/4pi = {NPIX/(4*np.pi):.6f}")
print()

# ===========================================================================
# TEST 1: Is map2alm_spin the adjoint of alm2map_spin (with what factor)?
# ===========================================================================
print("=" * 70)
print("TEST 1: adjoint relationship of alm2map_spin / map2alm_spin")
print("=" * 70)

# Random E-mode alm (size = hp.Alm.getsize(lmax_hp))
alm_size = hp.Alm.getsize(lmax_hp)
glm = rng.standard_normal(alm_size) + 1j * rng.standard_normal(alm_size)
glm[0] = 0
glm[1] = 0  # zero monopole/dipole
blm = np.zeros_like(glm)

# Forward: alm2map_spin
Q, U = hp.alm2map_spin([glm, blm], NSIDE, 1, lmax_hp)

# Random test map
g_Q = rng.standard_normal(NPIX)
g_U = rng.standard_normal(NPIX)

# Inner product <(Q,U), (g_Q, g_U)>_pix = sum(Q*g_Q) + sum(U*g_U)
inner_pix = np.sum(Q * g_Q) + np.sum(U * g_U)

# Adjoint candidate 1: map2alm_spin (bare)
g_glm_bare, _ = hp.map2alm_spin([g_Q, g_U], 1, lmax=lmax_hp)
inner_alm_bare = np.sum(np.real(np.conj(glm) * g_glm_bare))

# Adjoint candidate 2: map2alm_spin * (NPIX/4pi)
g_glm_norm = g_glm_bare * (NPIX / (4 * np.pi))
inner_alm_norm = np.sum(np.real(np.conj(glm) * g_glm_norm))

print(f"  <(Q,U), (gQ,gU)>_pix = {inner_pix:.8e}")
print(f"  <glm, map2alm_spin(gQ,gU)>_alm (bare)       = {inner_alm_bare:.8e}")
print(f"  ratio (pix/alm_bare)                         = {inner_pix/inner_alm_bare:.6f}")
print(f"  <glm, (NPIX/4pi)*map2alm_spin(gQ,gU)>_alm   = {inner_alm_norm:.8e}")
print(f"  ratio (pix/alm_norm)                         = {inner_pix/inner_alm_norm:.6f}")
print()

# ===========================================================================
# TEST 2: Is _alm_packed_to_hp / _alm_hp_to_packed a proper adjoint pair?
# ===========================================================================
print("=" * 70)
print("TEST 2: _alm_packed_to_hp adjoint = _alm_hp_to_packed")
print("=" * 70)

from diffcmb.lensing import _alm_hp_to_packed, _alm_packed_to_hp

n_real = LMAX * (LMAX + 1) // 2 - 3
n_imag = (LMAX - 2) * (LMAX - 1) // 2
n_packed = n_real + n_imag

packed_a = rng.standard_normal(n_packed)
packed_b = rng.standard_normal(n_packed)
hp_a = _alm_packed_to_hp(packed_a, LMAX)

# Random complex alm to use as 'g_hp_b' (gradient in alm space)
g_hp_b = rng.standard_normal(alm_size) + 1j * rng.standard_normal(alm_size)

# Forward: packed -> hp
# Adjoint: hp -> packed (via _alm_hp_to_packed)
g_packed_b = _alm_hp_to_packed(g_hp_b, LMAX)

# Inner product in packed space: sum(packed_a * g_packed_b)
inner_packed = np.sum(packed_a * g_packed_b)

# Inner product in alm space: Re(sum(conj(hp_a) * g_hp_b))
# But only over the elements that are accessed (L=2..LMAX-1, m=0..L)
inner_alm = np.sum(np.real(np.conj(hp_a) * g_hp_b))

print(f"  <packed_a, A_adj(g_hp_b)>_packed = {inner_packed:.8e}")
print(f"  <hp_a, g_hp_b>_alm               = {inner_alm:.8e}")
print(f"  ratio                             = {inner_packed/inner_alm:.6f}")
print("  (should be 1.0 for correct adjoint)")
print()

# ===========================================================================
# TEST 3: Verify deflection_field forward computation
# ===========================================================================
print("=" * 70)
print("TEST 3: deflection_field basic sanity")
print("=" * 70)

from diffcmb.lensing import _deflection_adjoint, deflection_field

# Use the standard-size phi_alm (getsize(LMAX) = 231) like in the tests
phi_alm_hp = rng.standard_normal(hp.Alm.getsize(LMAX)) + 1j * rng.standard_normal(hp.Alm.getsize(LMAX))
phi_alm_hp *= 1e-4  # small amplitude

d_theta, d_phi = deflection_field(phi_alm_hp, NSIDE, LMAX)
print(f"  deflection_field with size-{hp.Alm.getsize(LMAX)} phi_alm_hp")
print(f"  d_theta: max={np.max(np.abs(d_theta)):.4e}, dtype={d_theta.dtype}")
print(f"  d_phi:   max={np.max(np.abs(d_phi)):.4e}")
print()

# ===========================================================================
# TEST 4: Full adjoint test: is _deflection_adjoint the adjoint of deflection_field?
# ===========================================================================
print("=" * 70)
print("TEST 4: _deflection_adjoint vs expected adjoint (inner product test)")
print("=" * 70)

# Random packed phi
packed_phi = rng.standard_normal(n_packed) * 1e-4
phi_hp = _alm_packed_to_hp(packed_phi, LMAX)

# Forward deflection
d_theta_full, d_phi_full = deflection_field(phi_hp, NSIDE, LMAX)

# Random 'upstream' gradient maps
g_theta = rng.standard_normal(NPIX)
g_phi = rng.standard_normal(NPIX)

# Inner product in pixel space: <(d_theta, d_phi), (g_theta, g_phi)>
inner_pix_deflect = np.sum(d_theta_full * g_theta) + np.sum(d_phi_full * g_phi)

# Adjoint: should give g_packed such that <g_packed, delta_phi> = inner_pix_deflect * something
g_packed_result = _deflection_adjoint(g_theta, g_phi, NSIDE, LMAX)

# Inner product in packed space: <g_packed_result, packed_phi>
inner_packed_deflect = np.sum(g_packed_result * packed_phi)

print(f"  <(d_theta,d_phi), (g_theta,g_phi)>_pix     = {inner_pix_deflect:.8e}")
print(f"  <A_adj(g_theta,g_phi), phi_packed>_packed   = {inner_packed_deflect:.8e}")
print(f"  ratio (pix/packed)                          = {inner_pix_deflect/inner_packed_deflect:.6f}")
print("  (should be 1.0 for correct adjoint)")
print()

# Also check with the (NPIX/4pi) correction
corrected = inner_packed_deflect * (NPIX / (4 * np.pi))
print(f"  <adj*(NPIX/4pi), phi>  = {corrected:.8e}")
print(f"  ratio with correction  = {inner_pix_deflect/corrected:.6f}")
print()

# ===========================================================================
# TEST 5: Direct FD check of _deflection_adjoint for a single phi component
# ===========================================================================
print("=" * 70)
print("TEST 5: FD check of the adjoint formula (single component)")
print("=" * 70)

# For a single phi_packed component j, FD computes:
# (inner_pix(phi + eps*e_j) - inner_pix(phi - eps*e_j)) / (2*eps)
# This should equal g_packed_result[j]

eps = 1e-7
component = 5  # check component 5

phi_p = packed_phi.copy()
phi_p[component] += eps
phi_m = packed_phi.copy()
phi_m[component] -= eps

d_th_p, d_ph_p = deflection_field(_alm_packed_to_hp(phi_p, LMAX), NSIDE, LMAX)
d_th_m, d_ph_m = deflection_field(_alm_packed_to_hp(phi_m, LMAX), NSIDE, LMAX)

inner_p = np.sum(d_th_p * g_theta) + np.sum(d_ph_p * g_phi)
inner_m = np.sum(d_th_m * g_theta) + np.sum(d_ph_m * g_phi)
g_fd_single = (inner_p - inner_m) / (2 * eps)

print(f"  FD gradient at component {component}: {g_fd_single:.8e}")
print(f"  Adjoint result at component {component}: {g_packed_result[component]:.8e}")
print(f"  Ratio (adjoint/FD): {g_packed_result[component]/g_fd_single:.6f}")
print("  (should be 1.0 if adjoint is correct)")
print()

# Check multiple components
print("  Checking first 10 components:")
for j in range(min(10, n_packed)):
    phi_pj = packed_phi.copy()
    phi_pj[j] += eps
    phi_mj = packed_phi.copy()
    phi_mj[j] -= eps
    d_tp, d_pp = deflection_field(_alm_packed_to_hp(phi_pj, LMAX), NSIDE, LMAX)
    d_tm, d_pm = deflection_field(_alm_packed_to_hp(phi_mj, LMAX), NSIDE, LMAX)
    ip = np.sum(d_tp * g_theta) + np.sum(d_pp * g_phi)
    im_ = np.sum(d_tm * g_theta) + np.sum(d_pm * g_phi)
    gfd = (ip - im_) / (2 * eps)
    gad = g_packed_result[j]
    ratio = gad / gfd if abs(gfd) > 1e-15 else float('nan')
    print(f"    j={j:3d}: FD={gfd:12.5e}, adj={gad:12.5e}, ratio={ratio:.4f}")

print()

# ===========================================================================
# TEST 6: Check if bilinear weight grad backward is correct in isolation
# ===========================================================================
print("=" * 70)
print("TEST 6: bilinear weight grad backward")
print("=" * 70)

from diffcmb.lensing import _bilinear_weight_grads

pix = np.arange(NPIX)
packed_phi_test = rng.standard_normal(n_packed) * 1e-4
T_map = rng.standard_normal(NPIX) * 50.0
g_upstream = np.ones(NPIX)  # like sum loss

dw_dtheta, dw_dphi, neighbors, weights, theta_lensed, phi_lensed = \
    _bilinear_weight_grads(packed_phi_test, NSIDE, LMAX, pix)

T_at_nbrs = T_map[neighbors]  # (4, NPIX)

# Backward: dL/d(theta_lensed) and dL/d(phi_lensed)
dL_dth = g_upstream * np.sum(T_at_nbrs * dw_dtheta, axis=0)
dL_dph = g_upstream * np.sum(T_at_nbrs * dw_dphi,   axis=0)

# Check one component via FD: perturb d_theta[i] and see change in T_lensed
eps2 = 1e-7
i_test = 100

th_p = theta_lensed.copy()
th_p[i_test] += eps2
_, w_p = hp.get_interp_weights(NSIDE, th_p, phi_lensed)
Tl_p = np.sum(T_map[neighbors] * w_p)  # sum over all pixels

th_m = theta_lensed.copy()
th_m[i_test] -= eps2
_, w_m = hp.get_interp_weights(NSIDE, th_m, phi_lensed)
Tl_m = np.sum(T_map[neighbors] * w_m)

dL_dth_fd = (Tl_p - Tl_m) / (2 * eps2)

print(f"  At pixel {i_test}:")
print(f"    dL/d(theta_lensed) autodiff = {dL_dth[i_test]:.8e}")
print(f"    dL/d(theta_lensed) FD       = {dL_dth_fd:.8e}")
print(f"    ratio                       = {dL_dth[i_test]/dL_dth_fd if abs(dL_dth_fd)>1e-15 else 'N/A'}")
print()

# ===========================================================================
# TEST 7: Test sinθ factor in the adjoint
# ===========================================================================
print("=" * 70)
print("TEST 7: sinθ factor correctness")
print("=" * 70)

theta_pix, _ = hp.pix2ang(NSIDE, np.arange(NPIX))
sin_theta = np.clip(np.sin(theta_pix), 1e-10, None)

# The forward: U = sin_theta * d_phi → d_phi = U / sin_theta
# adjoint of (d_phi = U/sin_theta) w.r.t. U: g_U = g_phi / sin_theta
# Let's verify by FD on a simple identity

# f(U) = sum(U / sin_theta * g_phi) where g_phi is constant
# df/dU[i] = g_phi[i] / sin_theta[i]  ← g_U[i]
# So inner product: <g_U, dU> = sum(g_phi / sin_theta * dU)
# And <g_phi, d_phi> = <g_phi, (U+dU)/sin_theta - U/sin_theta> = sum(g_phi * dU / sin_theta) ✓

# This is trivially correct. Let's confirm the sign hasn't flipped.
g_phi_test = rng.standard_normal(NPIX)
g_U_test = g_phi_test / sin_theta
# Both should have same sign at each pixel
same_sign = np.sum(g_phi_test * g_U_test > 0)
print(f"  g_phi and g_U = g_phi/sin_theta: same sign at {same_sign}/{NPIX} pixels (expect ~all)")
print()

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print("If TEST 4 ratio != 1.0, the deflection adjoint is wrong.")
print("If TEST 5 ratios are inconsistent, there's an element-wise error.")
print("If TEST 6 ratio != 1.0, the bilinear backward is wrong.")
print("If TEST 1 ratio = NPIX/4pi, we need to add that normalization factor.")
print()
print("Done.")
