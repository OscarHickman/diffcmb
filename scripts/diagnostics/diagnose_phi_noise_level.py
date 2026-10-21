"""L >= 30 phi power of a chain by tercile of the ecliptic noise map, relative to the lowest-noise
tercile (ROADMAP T1.2a: does the A_phi = 1 trapped phi follow the noise? 2026-10-21: no).

Usage: PYTHONPATH=diffcmb .venv/bin/python scripts/diagnostics/diagnose_phi_noise_level.py LABEL CHAIN.npz [LABEL CHAIN.npz ...]
"""
import sys

import healpy as hp
import numpy as np

sys.path.insert(0, "scripts")
from real_sky_setup import ecliptic_noise_map

from diffcmb.lensing import _alm_packed_to_hp

LMAX, NSIDE, LCUT = 64, 64, 30
sig = ecliptic_noise_map(NSIDE, 30.0, 0.5)
edges = np.quantile(sig, [1 / 3, 2 / 3])
terc = {"low noise": sig < edges[0], "mid": (sig >= edges[0]) & (sig < edges[1]), "high noise": sig >= edges[1]}
print("sigma tercile means:", {k: round(float(sig[v].mean()), 1) for k, v in terc.items()})
filt = np.r_[np.zeros(LCUT), np.ones(LMAX - LCUT)]
for label, path in zip(sys.argv[1::2], sys.argv[2::2]):
    ps = np.load(path)["phi_samples"]
    acc = dict.fromkeys(terc, 0.0)
    idx = np.linspace(len(ps) * 2 // 3, len(ps) - 1, 40).astype(int)
    for j in idx:
        m = hp.alm2map(hp.almxfl(_alm_packed_to_hp(ps[j], LMAX), filt), NSIDE, lmax=LMAX - 1)
        for k, v in terc.items():
            acc[k] += np.mean(m[v] ** 2) / len(idx)
    print(f"{label:20s}" + "  ".join(f"{k} {v / acc['low noise']:.2f}" for k, v in acc.items()))
