"""Where on the sky is the high-L (L >= 30) phi power of a chain: deep in / at the edge of the
observed region and of the mask (ROADMAP T1.2a, the A_phi = 1 trap, 2026-10-20).

Usage: PYTHONPATH=diffcmb .venv/bin/python scripts/diagnostics/diagnose_phi_mask_edge.py LABEL CHAIN.npz [LABEL CHAIN.npz ...]
Prints each region's mean phi^2 over the last third of the chain, relative to the deep observed sky.
"""
import sys

import healpy as hp
import numpy as np

from diffcmb.lensing import _alm_packed_to_hp

LMAX, NSIDE, LCUT = 64, 64, 30
mask = hp.read_map("data/planck_common_mask_int_ns64.fits", dtype=np.float64) > 0.5
sm = hp.smoothing(mask.astype(float), fwhm=np.radians(6.0))
regions = {"obs deep": mask & (sm > 0.9), "obs edge": mask & (sm <= 0.9),
           "masked edge": ~mask & (sm >= 0.1), "masked deep": ~mask & (sm < 0.1)}
print("pixels:", {k: int(v.sum()) for k, v in regions.items()})
filt = np.r_[np.zeros(LCUT), np.ones(LMAX - LCUT)]
def profile(path, n=40):
    ps = np.load(path)["phi_samples"]
    idx = np.linspace(len(ps) * 2 // 3, len(ps) - 1, n).astype(int)
    acc = dict.fromkeys(regions, 0.0)
    for j in idx:
        m = hp.alm2map(hp.almxfl(_alm_packed_to_hp(ps[j], LMAX), filt), NSIDE, lmax=LMAX - 1)
        for k, v in regions.items():
            acc[k] += np.mean(m[v] ** 2) / n
    return acc
for label, path in zip(sys.argv[1::2], sys.argv[2::2]):
    a = profile(path)
    print(f"{label:22s}" + "  ".join(f"{k} {v / a['obs deep']:.2f}" for k, v in a.items()) + f"   (obs deep {a['obs deep']:.3g})")
