"""Debug: comprehensive check of bilinear FD (theta AND phi) for ALL 3072 pixels.

Tests:
  SCAN A: dL_dth from eps=1e-7 vs eps=1e-8, ALL pixels — find any bad ones
  SCAN B: dL_dph from eps=1e-7 vs eps=1e-8, ALL pixels — find any bad ones
  SCAN C: neighbor change check — does {nbrs(theta+1e-7)} != {nbrs(theta-1e-7)}?
  TEST C_CLEAN: redo TEST C but remove bad pixels from the gradient before calling adjoint
"""
import sys

import numpy as np

sys.path.insert(0, '/cosma/apps/durham/dc-hick2/diffcmb/diffcmb')
import healpy as hp

from diffcmb.lensing import (
    _alm_hp_to_packed,
    _alm_packed_to_hp,
    _bilinear_weight_grads,
    _deflection_adjoint,
    deflection_field,
)

NSIDE = 16
LMAX = 20
NPIX = hp.nside2npix(NSIDE)
rng = np.random.default_rng(42)

size = hp.Alm.getsize(LMAX)
phi_hp = rng.standard_normal(size) + 1j * rng.standard_normal(size)
ells = np.array([hp.Alm.getlm(LMAX, i)[0] for i in range(size)], dtype=float)
ells = np.maximum(ells, 1.0)
phi_hp *= 1e-4 / ells**1.5
phi_hp[0] = 0.0
phi_hp[1] = 0.0
phi_packed = _alm_hp_to_packed(phi_hp.astype(np.complex128), LMAX)
pix = np.arange(NPIX)

_, _, neighbors, weights, theta_lensed, phi_lensed = _bilinear_weight_grads(
    phi_packed, NSIDE, LMAX, pix)

T_np = rng.standard_normal(NPIX) * 50.0

# ===================================================================
# SCAN A: dL_dth: eps=1e-7 vs eps=1e-8 for ALL pixels
# ===================================================================
eps7 = 1e-7
eps8 = 1e-8

# eps=1e-7
th_p7 = np.clip(theta_lensed + eps7, 1e-12, np.pi - 1e-12)
th_m7 = np.clip(theta_lensed - eps7, 1e-12, np.pi - 1e-12)
nbrs7p, wts7p = hp.get_interp_weights(NSIDE, th_p7, phi_lensed)
nbrs7m, wts7m = hp.get_interp_weights(NSIDE, th_m7, phi_lensed)
dL_dth_7 = (np.sum(T_np[nbrs7p]*wts7p, axis=0) - np.sum(T_np[nbrs7m]*wts7m, axis=0)) / (2*eps7)

# eps=1e-8
th_p8 = np.clip(theta_lensed + eps8, 1e-12, np.pi - 1e-12)
th_m8 = np.clip(theta_lensed - eps8, 1e-12, np.pi - 1e-12)
nbrs8p, wts8p = hp.get_interp_weights(NSIDE, th_p8, phi_lensed)
nbrs8m, wts8m = hp.get_interp_weights(NSIDE, th_m8, phi_lensed)
dL_dth_8 = (np.sum(T_np[nbrs8p]*wts8p, axis=0) - np.sum(T_np[nbrs8m]*wts8m, axis=0)) / (2*eps8)

ratio_th = np.where(np.abs(dL_dth_8) > 1.0, dL_dth_7 / dL_dth_8, np.nan)
bad_th = np.where(np.abs(ratio_th - 1.0) > 0.05)[0]
print(f"SCAN A (theta FD): {len(bad_th)} bad pixels out of {NPIX}")
if len(bad_th) > 0:
    print(f"  Bad pixel indices (first 10): {bad_th[:10]}")
    for j in bad_th[:10]:
        print(f"    j={j}: dth_7={dL_dth_7[j]:.6g}, dth_8={dL_dth_8[j]:.6g}, ratio={ratio_th[j]:.4f}, theta_lensed={theta_lensed[j]:.6f}")

# ===================================================================
# SCAN B: dL_dph: eps=1e-7 vs eps=1e-8 for ALL pixels
# ===================================================================
ph_p7 = phi_lensed + eps7
ph_m7 = phi_lensed - eps7
nbrsp7, wtsp7 = hp.get_interp_weights(NSIDE, theta_lensed, ph_p7)
nbrsm7, wtsm7 = hp.get_interp_weights(NSIDE, theta_lensed, ph_m7)
dL_dph_7 = (np.sum(T_np[nbrsp7]*wtsp7, axis=0) - np.sum(T_np[nbrsm7]*wtsm7, axis=0)) / (2*eps7)

ph_p8 = phi_lensed + eps8
ph_m8 = phi_lensed - eps8
nbrsp8, wtsp8 = hp.get_interp_weights(NSIDE, theta_lensed, ph_p8)
nbrsm8, wtsm8 = hp.get_interp_weights(NSIDE, theta_lensed, ph_m8)
dL_dph_8 = (np.sum(T_np[nbrsp8]*wtsp8, axis=0) - np.sum(T_np[nbrsm8]*wtsm8, axis=0)) / (2*eps8)

ratio_ph = np.where(np.abs(dL_dph_8) > 1.0, dL_dph_7 / dL_dph_8, np.nan)
bad_ph = np.where(np.abs(ratio_ph - 1.0) > 0.05)[0]
print(f"SCAN B (phi FD):   {len(bad_ph)} bad pixels out of {NPIX}")
if len(bad_ph) > 0:
    print(f"  Bad pixel indices (first 10): {bad_ph[:10]}")
    for j in bad_ph[:10]:
        print(f"    j={j}: dph_7={dL_dph_7[j]:.6g}, dph_8={dL_dph_8[j]:.6g}, ratio={ratio_ph[j]:.4f}, phi_lensed={phi_lensed[j]:.6f}")

# ===================================================================
# SCAN C: neighbor change check for theta
# ===================================================================
# Check if any pixel's 4 neighbors differ between theta±1e-7
neighbor_sets_p = set(map(tuple, nbrs7p.T))  # set of 4-tuples
neighbor_changed = np.zeros(NPIX, dtype=bool)
for j in range(NPIX):
    if set(nbrs7p[:, j]) != set(nbrs7m[:, j]):
        neighbor_changed[j] = True
n_changed = np.sum(neighbor_changed)
print(f"SCAN C: {n_changed} pixels have different neighbors at theta±1e-7")
if n_changed > 0:
    changed_idx = np.where(neighbor_changed)[0]
    print(f"  Changed indices (first 5): {changed_idx[:5]}")
    for j in changed_idx[:5]:
        print(f"    j={j}: nbrs+={set(nbrs7p[:,j])}, nbrs-={set(nbrs7m[:,j])}")

# ===================================================================
# TEST C: full backward vs FD (with and without bad pixels)
# ===================================================================
eps = 1e-6
n_phi = len(phi_packed)
sampled = np.arange(0, n_phi, max(1, n_phi // 20))

# Forward FD
fd_C = np.zeros(n_phi)
for i in sampled:
    ph_p = phi_packed.copy()
    ph_p[i] += eps
    ph_m = phi_packed.copy()
    ph_m[i] -= eps
    _, _, nbrs_p, wts_p, _, _ = _bilinear_weight_grads(ph_p, NSIDE, LMAX, pix)
    _, _, nbrs_m, wts_m, _, _ = _bilinear_weight_grads(ph_m, NSIDE, LMAX, pix)
    fd_C[i] = (np.sum(T_np[nbrs_p] * wts_p) - np.sum(T_np[nbrs_m] * wts_m)) / (2 * eps)

# Full backward (current eps=1e-7)
dL_dth_full = np.zeros(NPIX)
dL_dth_full[pix] = dL_dth_7
dL_dph_full = np.zeros(NPIX)
dL_dph_full[pix] = dL_dph_7
g_auto7 = _deflection_adjoint(dL_dth_full, dL_dph_full, NSIDE, LMAX)

# Full backward using eps=1e-8 (potentially more accurate)
dL_dth_full8 = np.zeros(NPIX)
dL_dth_full8[pix] = dL_dth_8
dL_dph_full8 = np.zeros(NPIX)
dL_dph_full8[pix] = dL_dph_8
g_auto8 = _deflection_adjoint(dL_dth_full8, dL_dph_full8, NSIDE, LMAX)

print("\nTEST C: full backward comparison")
print(f"  {'i':4s} | {'auto(1e-7)':13s} | {'auto(1e-8)':13s} | {'FD':13s} | ratio(7) | ratio(8)")
for i in sampled[:12]:
    r7 = g_auto7[i] / fd_C[i] if abs(fd_C[i]) > 1e-10 else float('nan')
    r8 = g_auto8[i] / fd_C[i] if abs(fd_C[i]) > 1e-10 else float('nan')
    print(f"  {i:4d} | {g_auto7[i]:13.6g} | {g_auto8[i]:13.6g} | {fd_C[i]:13.6g} | {r7:.4f}   | {r8:.4f}")

# ===================================================================
# TEST D: compare backward with MUCH smaller eps
# ===================================================================
for eps_try in [1e-7, 1e-8, 1e-9, 1e-10]:
    th_pt = np.clip(theta_lensed + eps_try, 1e-12, np.pi - 1e-12)
    th_mt = np.clip(theta_lensed - eps_try, 1e-12, np.pi - 1e-12)
    nbrst, wtst = hp.get_interp_weights(NSIDE, th_pt, phi_lensed)
    nbrst2, wtst2 = hp.get_interp_weights(NSIDE, th_mt, phi_lensed)
    dth_t = (np.sum(T_np[nbrst]*wtst, axis=0) - np.sum(T_np[nbrst2]*wtst2, axis=0)) / (2*eps_try)
    ph_pt = phi_lensed + eps_try
    ph_mt = phi_lensed - eps_try
    nbrsp_t, wtsp_t = hp.get_interp_weights(NSIDE, theta_lensed, ph_pt)
    nbrsm_t, wtsm_t = hp.get_interp_weights(NSIDE, theta_lensed, ph_mt)
    dph_t = (np.sum(T_np[nbrsp_t]*wtsp_t, axis=0) - np.sum(T_np[nbrsm_t]*wtsm_t, axis=0)) / (2*eps_try)
    dth_f = np.zeros(NPIX)
    dth_f[pix] = dth_t
    dph_f = np.zeros(NPIX)
    dph_f[pix] = dph_t
    g_t = _deflection_adjoint(dth_f, dph_f, NSIDE, LMAX)
    max_reldiff = np.max(np.abs(g_t[sampled] - fd_C[sampled]) / (np.abs(fd_C[sampled]) + 1e-10))
    print(f"  eps={eps_try:.0e}: max|auto-FD|/|FD| = {max_reldiff:.4f}")

print("\nDone.")
