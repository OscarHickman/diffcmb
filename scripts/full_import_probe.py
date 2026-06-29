"""Test: does importing the full diffcmb stack (including model/CAMB) cause the crash?"""
import faulthandler
import sys

faulthandler.enable()
import numpy as np

# Mirror what tests/test_lensing.py imports at module level
try:
    import healpy as hp
except ImportError:
    hp = None

try:
    import tensorflow as tf
    print(f"TF {tf.__version__}")
except ImportError:
    tf = None

sys.path.insert(0, 'diffcmb')
from diffcmb.lensing import (
    _alm_hp_to_packed,
    _alm_packed_to_hp,
    _deflection_adjoint,
    apply_lensing_tf,
    deflection_field,
    lens_map_phi_diff_tf,
    lens_map_tf,
    precompute_lensing,
    psi_lensed,
)

print("lensing imports OK")

from diffcmb.model import CosmologyAdvancedSampling

print("model import OK")

LMAX = 20
NSIDE = 16

print(f"\nTesting deflection_field with zero phi (lmax={LMAX}, nside={NSIDE})...")
phi = np.zeros(LMAX*(LMAX+1)//2, dtype=np.complex128)
d_theta, d_phi = deflection_field(phi, NSIDE, LMAX)
print(f"  PASSED: max={max(np.max(np.abs(d_theta)), np.max(np.abs(d_phi))):.2e}")

print("\nTesting with non-zero phi...")
rng = np.random.default_rng(0)
phi_nz = (rng.normal(0, 5e-4, LMAX*(LMAX+1)//2) * np.array([0]*4 + [1]*(LMAX*(LMAX+1)//2-4))).astype(complex)
d_theta2, d_phi2 = deflection_field(phi_nz, NSIDE, LMAX)
print(f"  PASSED: max={max(np.max(np.abs(d_theta2)), np.max(np.abs(d_phi2))):.4e}")

print("\nAll passed.")
