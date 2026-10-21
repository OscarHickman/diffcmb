"""Debug: isolate _deflection_adjoint correctness vs bilinear-FD correctness."""
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
npix = hp.nside2npix(NSIDE)
rng = np.random.default_rng(42)

# Build random phi_packed (same as test)
size = hp.Alm.getsize(LMAX)
phi_hp = rng.standard_normal(size) + 1j * rng.standard_normal(size)
ells = np.array([hp.Alm.getlm(LMAX, i)[0] for i in range(size)], dtype=float)
ells = np.maximum(ells, 1.0)
phi_hp *= 1e-4 / ells**1.5
phi_hp[0] = 0.0
if LMAX >= 2:
    phi_hp[1] = 0.0
phi_packed = _alm_hp_to_packed(phi_hp.astype(np.complex128), LMAX)
pix = np.arange(npix)

print("=" * 65)
print("TEST A: _deflection_adjoint vs FD of deflection_field")
print("=" * 65)
# Random upstream gradients over FULL sky
g_theta = rng.standard_normal(npix)
g_phi_pix = rng.standard_normal(npix)

# Adjoint
adj = _deflection_adjoint(g_theta, g_phi_pix, NSIDE, LMAX)

# FD of L = <g_theta, d_theta> + <g_phi_pix, d_phi>
phi_alm_hp0 = _alm_packed_to_hp(phi_packed, LMAX)
d_theta0, d_phi0 = deflection_field(phi_alm_hp0, NSIDE, LMAX)
L0 = np.dot(g_theta, d_theta0) + np.dot(g_phi_pix, d_phi0)

eps = 1e-6
n_phi = len(phi_packed)
sampled = np.arange(0, n_phi, max(1, n_phi // 20))
fd_A = np.zeros(n_phi)
for i in sampled:
    ph_p = phi_packed.copy()
    ph_p[i] += eps
    ph_m = phi_packed.copy()
    ph_m[i] -= eps
    d_tp, d_pp = deflection_field(_alm_packed_to_hp(ph_p, LMAX), NSIDE, LMAX)
    d_tm, d_pm = deflection_field(_alm_packed_to_hp(ph_m, LMAX), NSIDE, LMAX)
    lp = np.dot(g_theta, d_tp) + np.dot(g_phi_pix, d_pp)
    lm = np.dot(g_theta, d_tm) + np.dot(g_phi_pix, d_pm)
    fd_A[i] = (lp - lm) / (2 * eps)

print("  i | adj           | FD (deflect)  | ratio")
for i in sampled[:10]:
    r = adj[i] / fd_A[i] if abs(fd_A[i]) > 1e-10 else float('nan')
    print(f"  {i:3d} | {adj[i]:13.6g} | {fd_A[i]:13.6g} | {r:.4f}")


print()
print("=" * 65)
print("TEST B: scalar bilinear FD vs FD of T_interp w.r.t. theta'")
print("=" * 65)
T_np = rng.standard_normal(npix) * 50.0

# Compute lensed positions
_, _, neighbors, weights, theta_lensed, phi_lensed = _bilinear_weight_grads(
    phi_packed, NSIDE, LMAX, pix)

# Scalar bilinear FD (my backward)
eps_angle = 1e-4
th_p = np.clip(theta_lensed + eps_angle, 1e-12, np.pi - 1e-12)
th_m = np.clip(theta_lensed - eps_angle, 1e-12, np.pi - 1e-12)
nbrs_tp, wts_tp = hp.get_interp_weights(NSIDE, th_p, phi_lensed)
nbrs_tm, wts_tm = hp.get_interp_weights(NSIDE, th_m, phi_lensed)
T_tp = np.sum(T_np[nbrs_tp] * wts_tp, axis=0)
T_tm = np.sum(T_np[nbrs_tm] * wts_tm, axis=0)
dL_dth_myFD = (T_tp - T_tm) / (2.0 * eps_angle)  # (upstream=1)

# Direct FD of T_interp w.r.t. theta'_j (reference)
dL_dth_ref = np.zeros(npix)
eps2 = 1e-6
for j in range(0, npix, max(1, npix // 30)):
    # perturb theta'_j only
    th_j_p = min(theta_lensed[j] + eps2, np.pi - 1e-12)
    th_j_m = max(theta_lensed[j] - eps2, 1e-12)
    nbrs_p, wts_p = hp.get_interp_weights(NSIDE, np.array([th_j_p]), np.array([phi_lensed[j]]))
    nbrs_m, wts_m = hp.get_interp_weights(NSIDE, np.array([th_j_m]), np.array([phi_lensed[j]]))
    T_p2 = np.sum(T_np[nbrs_p] * wts_p)
    T_m2 = np.sum(T_np[nbrs_m] * wts_m)
    dL_dth_ref[j] = (T_p2 - T_m2) / (2 * eps2)

print("  j  | myFD (1e-4)   | ref FD (1e-6) | ratio")
for j in range(0, npix, max(1, npix // 30)):
    r = dL_dth_myFD[j] / dL_dth_ref[j] if abs(dL_dth_ref[j]) > 1e-10 else float('nan')
    print(f"  {j:3d} | {dL_dth_myFD[j]:13.6g} | {dL_dth_ref[j]:13.6g} | {r:.4f}")


print()
print("=" * 65)
print("TEST C: full backward vs FD of sum(T_lensed)")
print("=" * 65)
# Compute T_lensed at current phi
T_lensed0 = np.sum(T_np[neighbors] * weights, axis=0)
L0_lens = np.sum(T_lensed0)

# Full backward (upstream=ones)
g = np.ones(npix)
ph_p2 = phi_lensed + eps_angle
ph_m2 = phi_lensed - eps_angle
nbrs_pp, wts_pp = hp.get_interp_weights(NSIDE, theta_lensed, ph_p2)
nbrs_pm, wts_pm = hp.get_interp_weights(NSIDE, theta_lensed, ph_m2)
T_pp = np.sum(T_np[nbrs_pp] * wts_pp, axis=0)
T_pm = np.sum(T_np[nbrs_pm] * wts_pm, axis=0)
dL_dph_myFD = (T_pp - T_pm) / (2.0 * eps_angle)

dL_dth_full = np.zeros(npix)
dL_dth_full[pix] = dL_dth_myFD
dL_dph_full = np.zeros(npix)
dL_dph_full[pix] = dL_dph_myFD
g_phi_auto = _deflection_adjoint(dL_dth_full, dL_dph_full, NSIDE, LMAX)

# FD of sum(T_lensed) w.r.t. phi_packed
fd_C = np.zeros(n_phi)
for i in sampled:
    ph_p = phi_packed.copy()
    ph_p[i] += eps
    ph_m = phi_packed.copy()
    ph_m[i] -= eps
    _, _, nbrs_p2, wts_p2, _, _ = _bilinear_weight_grads(ph_p, NSIDE, LMAX, pix)
    _, _, nbrs_m2, wts_m2, _, _ = _bilinear_weight_grads(ph_m, NSIDE, LMAX, pix)
    lp2 = np.sum(T_np[nbrs_p2] * wts_p2)
    lm2 = np.sum(T_np[nbrs_m2] * wts_m2)
    fd_C[i] = (lp2 - lm2) / (2 * eps)

print("  i | auto (full)   | FD (sum Tlens)| ratio")
for i in sampled[:10]:
    r = g_phi_auto[i] / fd_C[i] if abs(fd_C[i]) > 1e-10 else float('nan')
    print(f"  {i:3d} | {g_phi_auto[i]:13.6g} | {fd_C[i]:13.6g} | {r:.4f}")

print("\nDone.")
