"""Pure-numpy diagnostic: does _deflection_adjoint give the correct gradient
in the LENSING context (same setup as test_phi_grad_deflection_adjoint_vs_fd)?

Bypasses TF entirely to isolate whether the error is in the math or in how TF
handles the custom_gradient backward.

Strategy
--------
1. Replicate the exact test setup (same RNG seed 42).
2. Compute dL/dtheta and dL/dphi from the bilinear backward manually.
3. Compute g_phi_adj = _deflection_adjoint(dL_dth_full, dL_dph_full).
4. Compute g_phi_fd by direct FD of sum(T_lensed) in pure numpy.
5. Compare ratio per component.

This settles whether the adjoint is mathematically correct for the lensing inputs.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import healpy as hp

LMAX = 20
NSIDE = 16
NPIX = 12 * NSIDE * NSIDE
n_real = LMAX * (LMAX + 1) // 2 - 3
n_imag = (LMAX - 2) * (LMAX - 1) // 2
n_packed = n_real + n_imag

print(f"LMAX={LMAX}, NSIDE={NSIDE}, NPIX={NPIX}, n_packed={n_packed}")
print(f"4pi/NPIX = {4*np.pi/NPIX:.6f},  NPIX/4pi = {NPIX/(4*np.pi):.6f}")
print()

from diffcmb.lensing import (
    _alm_hp_to_packed,
    _alm_packed_to_hp,
    _bilinear_weight_grads,
    _deflection_adjoint,
    deflection_field,
)


def _rand_phi_packed(lmax, rng, amplitude=5e-4):
    """Same helper as in test_lensing.py."""
    size = hp.Alm.getsize(lmax)
    phi_hp = rng.standard_normal(size) + 1j * rng.standard_normal(size)
    ells = np.array([hp.Alm.getlm(lmax, i)[0] for i in range(size)], dtype=float)
    ells = np.maximum(ells, 1.0)
    phi_hp *= amplitude / ells**1.5
    phi_hp[0] = 0.0
    if lmax >= 2:
        phi_hp[1] = 0.0
    return _alm_hp_to_packed(phi_hp.astype(np.complex128), lmax)


# Exact same seed as the test
rng = np.random.default_rng(42)
phi0 = _rand_phi_packed(LMAX, rng, amplitude=1e-4)
T_np = rng.standard_normal(NPIX) * 50.0
pix = np.arange(NPIX)

theta0, phi0_ang = hp.pix2ang(NSIDE, pix)

print("=" * 70)
print("TEST A: Pure-numpy adjoint vs FD in the LENSING context")
print("=" * 70)
print("(This is the same computation as test_phi_grad_deflection_adjoint_vs_fd")
print(" but without TF.)")
print()

# Step 1: Compute the bilinear backward inputs (same as the backward pass).
dw_dtheta, dw_dphi, neighbors, weights, theta_lensed, phi_lensed = \
    _bilinear_weight_grads(phi0, NSIDE, LMAX, pix)

T_at_nbrs = T_np[neighbors]  # (4, NPIX)

# upstream = ones (from sum loss)
upstream = np.ones(NPIX)
dL_dth = upstream * np.sum(T_at_nbrs * dw_dtheta, axis=0)  # (NPIX,)
dL_dph = upstream * np.sum(T_at_nbrs * dw_dphi, axis=0)    # (NPIX,)

print(f"dL_dth: max={np.max(np.abs(dL_dth)):.4e}, mean={np.mean(np.abs(dL_dth)):.4e}")
print(f"dL_dph: max={np.max(np.abs(dL_dph)):.4e}, mean={np.mean(np.abs(dL_dph)):.4e}")

# Step 2: Compute the adjoint (this is what the backward function does).
g_phi_adj = _deflection_adjoint(dL_dth, dL_dph, NSIDE, LMAX)
print(f"g_phi_adj[:5] = {g_phi_adj[:5]}")
print()

# Step 3: Compute FD of sum(T_lensed) w.r.t. phi_packed.
def compute_T_lensed_sum(phi_packed):
    phi_hp = _alm_packed_to_hp(phi_packed, LMAX)
    d_th, d_ph = deflection_field(phi_hp, NSIDE, LMAX)
    theta_l = np.clip(theta0 + d_th, 1e-12, np.pi - 1e-12)
    phi_l = phi0_ang + d_ph
    nbrs, wts = hp.get_interp_weights(NSIDE, theta_l, phi_l)
    return np.sum(T_np[nbrs] * wts)

eps = 1e-6
n_check = 20
# Use same sampled indices as the test: arange(0, n_packed, max(1, n_packed // 20))
sampled = np.arange(0, n_packed, max(1, n_packed // 20))[:n_check]

g_phi_fd = np.zeros(n_packed)
print("Computing FD (20 components)...")
for j in sampled:
    phi_p = phi0.copy()
    phi_p[j] += eps
    phi_m = phi0.copy()
    phi_m[j] -= eps
    g_phi_fd[j] = (compute_T_lensed_sum(phi_p) - compute_T_lensed_sum(phi_m)) / (2 * eps)
print()

print("adj vs FD for sampled components:")
print(f"{'j':>5} {'adj':>14} {'FD':>14} {'ratio':>10}")
print("-" * 50)
for j in sampled:
    adj = g_phi_adj[j]
    fd = g_phi_fd[j]
    ratio = adj / fd if abs(fd) > 1e-10 else float('nan')
    print(f"{j:5d} {adj:14.4e} {fd:14.4e} {ratio:10.5f}")

print()

print("=" * 70)
print("TEST B: Same as TEST A but using random (g_theta, g_phi) as in TEST 5")
print("        Should confirm the 4pi/NPIX factor from the debug script")
print("=" * 70)

rng2 = np.random.default_rng(99)
g_theta_rand = rng2.standard_normal(NPIX)
g_phi_rand = rng2.standard_normal(NPIX)

g_packed_rand = _deflection_adjoint(g_theta_rand, g_phi_rand, NSIDE, LMAX)

def compute_deflect_inner(phi_packed):
    phi_hp = _alm_packed_to_hp(phi_packed, LMAX)
    d_th, d_ph = deflection_field(phi_hp, NSIDE, LMAX)
    return np.sum(d_th * g_theta_rand) + np.sum(d_ph * g_phi_rand)

eps2 = 1e-7
g_fd_rand = np.zeros(n_packed)
print("Computing FD for random-input test (10 components)...")
for j in range(10):
    phi_p = phi0.copy()
    phi_p[j] += eps2
    phi_m = phi0.copy()
    phi_m[j] -= eps2
    g_fd_rand[j] = (compute_deflect_inner(phi_p) - compute_deflect_inner(phi_m)) / (2 * eps2)

print()
print("adj vs FD for random inputs (j=0..9):")
print(f"{'j':>5} {'adj':>14} {'FD':>14} {'ratio':>10}")
print("-" * 50)
for j in range(10):
    adj = g_packed_rand[j]
    fd = g_fd_rand[j]
    ratio = adj / fd if abs(fd) > 1e-10 else float('nan')
    print(f"{j:5d} {adj:14.4e} {fd:14.4e} {ratio:10.5f}")

print()

print("=" * 70)
print("TEST C: Cross-check — does _deflect_adjoint(dL_dth, dL_dph) equal")
print("        FD of deflect_inner when g_theta=dL_dth, g_phi=dL_dph?")
print("=" * 70)

g_packed_lensing_inputs = _deflection_adjoint(dL_dth, dL_dph, NSIDE, LMAX)

def compute_deflect_inner_lensing(phi_packed):
    phi_hp = _alm_packed_to_hp(phi_packed, LMAX)
    d_th, d_ph = deflection_field(phi_hp, NSIDE, LMAX)
    return np.sum(d_th * dL_dth) + np.sum(d_ph * dL_dph)

g_fd_lensing = np.zeros(n_packed)
print("Computing FD for deflect_inner with lensing inputs (20 components)...")
for j in sampled:
    phi_p = phi0.copy()
    phi_p[j] += eps
    phi_m = phi0.copy()
    phi_m[j] -= eps
    g_fd_lensing[j] = (compute_deflect_inner_lensing(phi_p) - compute_deflect_inner_lensing(phi_m)) / (2 * eps)

print()
print("adj vs FD of deflect_inner(dL_dth, dL_dph):")
print("(If TEST A ratio ≈ C ratio: issue is in chain rule chain, not in adjoint.)")
print("(If TEST C ratio ≈ 4pi/NPIX and TEST A ratio ≈ 171: FD_lensing ≠ FD_deflect !!!)")
print(f"{'j':>5} {'adj':>14} {'FD_deflect':>14} {'ratio':>10}")
print("-" * 60)
for j in sampled:
    adj = g_packed_lensing_inputs[j]
    fd = g_fd_lensing[j]
    ratio = adj / fd if abs(fd) > 1e-10 else float('nan')
    print(f"{j:5d} {adj:14.4e} {fd:14.4e} {ratio:10.5f}")

print()

print("=" * 70)
print("TEST D: Is FD_lensing == FD_deflect(dL_dth, dL_dph)?")
print("        Compare the two FD estimates component by component.")
print("=" * 70)
print()
print("FD_lensing vs FD_deflect ratio (should be 1.0 by chain rule):")
print(f"{'j':>5} {'FD_lensing':>14} {'FD_deflect':>14} {'ratio':>10}")
print("-" * 60)
for j in sampled:
    fd_l = g_phi_fd[j]
    fd_d = g_fd_lensing[j]
    ratio = fd_l / fd_d if abs(fd_d) > 1e-10 else float('nan')
    print(f"{j:5d} {fd_l:14.4e} {fd_d:14.4e} {ratio:10.5f}")

print()
print("Done.")
