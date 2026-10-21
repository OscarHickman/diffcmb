"""Minimal reproduction: does importing TF before healpy cause alm2map_spin to crash?"""
import sys

NSIDE = 16
LMAX = 20

print("=== Test A: healpy only, no TF ===")
import healpy as hp
import numpy as np

alm = np.zeros(hp.Alm.getsize(LMAX-1), dtype=np.complex128)
r = hp.alm2map_spin([alm, alm], NSIDE, 1, LMAX-1)
print(f"  PASSED: shape {r[0].shape}")

print("\n=== Test B: import TF, then healpy alm2map_spin ===")
import importlib

# Force TF import
try:
    import tensorflow as tf
    print(f"  TF version: {tf.__version__}")
except Exception as e:
    print(f"  TF import failed: {e}")
    sys.exit(0)

# Now try alm2map_spin again in the same process
r2 = hp.alm2map_spin([alm, alm], NSIDE, 1, LMAX-1)
print(f"  PASSED: shape {r2[0].shape}")

print("\nAll passed — TF/healpy coexistence OK on this node.")
