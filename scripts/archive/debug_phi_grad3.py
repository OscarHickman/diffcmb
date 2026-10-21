"""Debug: verify adjoint + bilinear-FD with CURRENT code (m>0 factor, eps=1e-7)."""
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

# Same phi as the failing test (seed=42, amplitude=1e-4)
size = hp.Alm.getsize(LMAX)
phi_hp = rng.standard_normal(size) + 1j * rng.standard_normal(size)
ells = np.array([hp.Alm.getlm(LMAX, i)[0] for i in range(size)], dtype=float)
ells = np.maximum(ells, 1.0)
phi_hp *= 1e-4 / ells**1.5
phi_hp[0] = 0.0
if LMAX >= 2:
    phi_hp[1] = 0.0
phi_packed = _alm_hp_to_packed(phi_hp.astype(np.complex128), LMAX)
pix = np.arange(NPIX)

print("=" * 65)
print("TEST A: _deflection_adjoint vs FD (current code, m>0 fix applied)")
print("=" * 65)
g_theta = rng.standard_normal(NPIX)
g_phi_pix = rng.standard_normal(NPIX)
adj = _deflection_adjoint(g_theta, g_phi_pix, NSIDE, LMAX)

phi_alm_hp0 = _alm_packed_to_hp(phi_packed, LMAX)
d_theta0, d_phi0 = deflection_field(phi_alm_hp0, NSIDE, LMAX)

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
print("TEST B: bilinear FD eps=1e-7 vs reference FD eps=1e-8 on theta'")
print("=" * 65)
T_np = rng.standard_normal(NPIX) * 50.0
_, _, neighbors, weights, theta_lensed, phi_lensed = _bilinear_weight_grads(
    phi_packed, NSIDE, LMAX, pix)

eps_bfd = 1e-7
th_p = np.clip(theta_lensed + eps_bfd, 1e-12, np.pi - 1e-12)
th_m = np.clip(theta_lensed - eps_bfd, 1e-12, np.pi - 1e-12)
nbrs_tp, wts_tp = hp.get_interp_weights(NSIDE, th_p, phi_lensed)
nbrs_tm, wts_tm = hp.get_interp_weights(NSIDE, th_m, phi_lensed)
dL_dth_bfd = (np.sum(T_np[nbrs_tp]*wts_tp, axis=0) - np.sum(T_np[nbrs_tm]*wts_tm, axis=0)) / (2.0*eps_bfd)

# Reference: eps=1e-8 directly on theta'
eps_ref = 1e-8
dL_dth_ref = np.zeros(NPIX)
check_pix = np.arange(0, NPIX, max(1, NPIX // 30))
for j in check_pix:
    th_jp = min(theta_lensed[j] + eps_ref, np.pi - 1e-12)
    th_jm = max(theta_lensed[j] - eps_ref, 1e-12)
    n_p, w_p = hp.get_interp_weights(NSIDE, np.array([th_jp]), np.array([phi_lensed[j]]))
    n_m, w_m = hp.get_interp_weights(NSIDE, np.array([th_jm]), np.array([phi_lensed[j]]))
    dL_dth_ref[j] = (np.sum(T_np[n_p]*w_p) - np.sum(T_np[n_m]*w_m)) / (2*eps_ref)

print("  j  | bfd (1e-7)    | ref (1e-8)    | ratio")
bad_count = 0
for j in check_pix[:15]:
    r = dL_dth_bfd[j]/dL_dth_ref[j] if abs(dL_dth_ref[j]) > 1e-10 else float('nan')
    flag = " *** BAD" if abs(r-1)>0.05 else ""
    if abs(r-1) > 0.05:
        bad_count += 1
    print(f"  {j:4d} | {dL_dth_bfd[j]:13.6g} | {dL_dth_ref[j]:13.6g} | {r:.4f}{flag}")
print(f"  Bad pixels in check_pix: {bad_count}/{len(check_pix[:15])}")


print()
print("=" * 65)
print("TEST C: full backward (eps=1e-7) vs FD of sum(T_lensed)")
print("=" * 65)
# Bilinear FD for theta and phi  (matches current backward code)
ph_p2 = phi_lensed + eps_bfd
ph_m2 = phi_lensed - eps_bfd
nbrs_pp, wts_pp = hp.get_interp_weights(NSIDE, theta_lensed, ph_p2)
nbrs_pm, wts_pm = hp.get_interp_weights(NSIDE, theta_lensed, ph_m2)
dL_dph_bfd = (np.sum(T_np[nbrs_pp]*wts_pp, axis=0) - np.sum(T_np[nbrs_pm]*wts_pm, axis=0)) / (2.0*eps_bfd)

dL_dth_full = np.zeros(NPIX)
dL_dth_full[pix] = dL_dth_bfd
dL_dph_full = np.zeros(NPIX)
dL_dph_full[pix] = dL_dph_bfd
g_phi_auto = _deflection_adjoint(dL_dth_full, dL_dph_full, NSIDE, LMAX)

# FD of sum(T_lensed)
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
for i in sampled[:12]:
    r = g_phi_auto[i] / fd_C[i] if abs(fd_C[i]) > 1e-10 else float('nan')
    print(f"  {i:3d} | {g_phi_auto[i]:13.6g} | {fd_C[i]:13.6g} | {r:.4f}")

print()
print("=" * 65)
print("TEST D: check dL_dth scale — what does adjoint expect?")
print("=" * 65)
# If adjoint is correct, g_phi_auto should = FD of sum_j g_theta_j * d_theta_j
# where g_theta_j = dL/dtheta'_j. Let's verify by computing what adjoint gives
# when g_theta = dL_dth_bfd and comparing with direct FD.
print("  Summary of dL_dth magnitudes:")
print(f"    max|dL_dth_bfd| = {np.max(np.abs(dL_dth_bfd)):.4g}")
print(f"    max|dL_dth_ref| = {np.max(np.abs(dL_dth_ref[check_pix])):.4g}")
print(f"    mean|dL_dth_bfd| = {np.mean(np.abs(dL_dth_bfd)):.4g}")
print(f"    T values std: {np.std(T_np):.4g}")

print("\nDone.")
