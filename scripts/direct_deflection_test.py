"""Run deflection_field directly (no pytest) to isolate the crash."""
import faulthandler
import sys

faulthandler.enable()

import numpy as np

try:
    import healpy as hp
except ImportError:
    hp = None

try:
    import tensorflow as tf
    print(f"TF {tf.__version__} loaded")
except ImportError:
    tf = None
    print("TF not available")

sys.path.insert(0, 'diffcmb')
from diffcmb.lensing import deflection_field

LMAX = 20
NSIDE = 16

print(f"lmax={LMAX}, nside={NSIDE}")
print(f"alm size = {LMAX*(LMAX+1)//2}, hp.Alm.getsize(LMAX-1) = {hp.Alm.getsize(LMAX-1)}")

# Test 1: zeros
print("\nTest 1: zero phi")
phi_zeros = np.zeros(LMAX*(LMAX+1)//2, dtype=np.complex128)
d_theta, d_phi = deflection_field(phi_zeros, NSIDE, LMAX)
print(f"  PASSED: max deflection = {max(np.max(np.abs(d_theta)), np.max(np.abs(d_phi))):.2e}")

# Test 2: small non-zero phi
print("\nTest 2: non-zero phi")
rng = np.random.default_rng(42)
phi_packed = rng.normal(0, 5e-4, LMAX*(LMAX+1)//2) + 0j
phi_packed[:4] = 0  # zero monopole/dipole
d_theta2, d_phi2 = deflection_field(phi_packed, NSIDE, LMAX)
print(f"  PASSED: max deflection = {max(np.max(np.abs(d_theta2)), np.max(np.abs(d_phi2))):.4e} rad")

print("\nAll direct tests passed.")
