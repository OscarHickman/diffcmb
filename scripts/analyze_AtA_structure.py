"""Characterise the off-diagonal structure of A^T A for the real ducc0
full-sky HEALPix SHT (ROADMAP.md Phase 0c Step 5): is it banded in (l, m),
low-rank, or diffuse/dense? This determines which scalable correction is
worth building for the messenger-field sampler at production lmax — a
Woodbury/low-rank correction (if compressible) or the more invasive
Gauss-Legendre exact-quadrature switch (if not).

Usage: PYTHONPATH=diffcmb .venv/bin/python scripts/analyze_AtA_structure.py
"""
import numpy as np

from diffcmb.samplers import _packed_to_alm_ho
from diffcmb.sht_ducc import HealpixSHT

LMAX = 16
NSIDE = 16


def build_lm_index(lmax, n_real, n_imag):
    """(L, m, part) for each packed-index row, matching _build_inv_cl_diag's
    layout: real parts (L=2..lmax-1, m=0..L) then imag parts (m=2..L)."""
    lm = []
    for L in range(2, lmax):
        for m in range(L + 1):
            lm.append((L, m, 're'))
    for L in range(2, lmax):
        for m in range(2, L + 1):
            lm.append((L, m, 'im'))
    assert len(lm) == n_real + n_imag
    return lm


def build_dense_J(lmax, nside, n_real, n_imag):
    n_alm = n_real + n_imag
    sht_full = HealpixSHT(nside=nside, lmax=lmax, unmasked_idx=None)
    npix = sht_full.npix
    J = np.empty((npix, n_alm), dtype=np.float64)
    e = np.zeros(n_alm)
    for i in range(n_alm):
        e[:] = 0.0
        e[i] = 1.0
        alm_ho = _packed_to_alm_ho(e, lmax, n_real)
        J[:, i] = sht_full.synthesis_full(alm_ho)
    return J


def main():
    lmax, nside = LMAX, NSIDE
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = (lmax - 2) * (lmax - 1) // 2
    n_alm = n_real + n_imag
    print(f"lmax={lmax}, NSIDE={nside}, n_alm={n_alm}")

    lm = build_lm_index(lmax, n_real, n_imag)
    L_arr = np.array([x[0] for x in lm])
    m_arr = np.array([x[1] for x in lm])

    print("Building dense full-sky J (probing A_action)...")
    J = build_dense_J(lmax, nside, n_real, n_imag)
    AtA = J.T @ J
    diag = np.diag(AtA)
    off = AtA - np.diag(diag)

    frob_off = np.linalg.norm(off)
    frob_diag = np.linalg.norm(diag)
    print(f"\n||off-diag||_F / ||diag||_F = {frob_off / frob_diag:.4e}")
    print(f"max|off-diag| / mean(diag)   = {np.abs(off).max() / diag.mean():.4e}")

    # --- Structure check 1: does off-diagonal magnitude decay with |dL|, |dm|?
    dL = np.abs(L_arr[:, None] - L_arr[None, :])
    dm = np.abs(m_arr[:, None] - m_arr[None, :])
    print("\nMean |off-diag| by |L_i - L_j| (normalised by mean(diag)):")
    for d in range(0, 14):
        mask = (dL == d)
        np.fill_diagonal(mask, False)
        if mask.sum() == 0:
            continue
        val = np.abs(off[mask]).mean() / diag.mean()
        print(f"  |dL|={d}: mean|off-diag|/mean(diag) = {val:.4e}  (n={mask.sum()})")

    print("\nMean |off-diag| by |m_i - m_j| (normalised by mean(diag)):")
    for d in range(0, 14):
        mask = (dm == d)
        np.fill_diagonal(mask, False)
        if mask.sum() == 0:
            continue
        val = np.abs(off[mask]).mean() / diag.mean()
        print(f"  |dm|={d}: mean|off-diag|/mean(diag) = {val:.4e}  (n={mask.sum()})")

    # --- Structure check 2: low-rank compressibility of the off-diagonal part
    print("\nSVD of off-diagonal part (compressibility check):")
    svals = np.linalg.svd(off, compute_uv=False)
    cum_energy = np.cumsum(svals**2) / np.sum(svals**2)
    for frac in (0.5, 0.8, 0.9, 0.95, 0.99):
        k = int(np.searchsorted(cum_energy, frac) + 1)
        print(f"  rank needed for {frac*100:.0f}% of off-diag energy: {k} / {n_alm} ({100*k/n_alm:.1f}%)")

    # --- Structure check 3: banded bandwidth needed to capture most energy
    # (using a single combined "distance" in the packed index itself, which
    # groups by L already given the layout)
    idx = np.arange(n_alm)
    didx = np.abs(idx[:, None] - idx[None, :])
    print("\nEnergy captured by index-bandwidth truncation:")
    total_energy = np.sum(off**2)
    for bw in (1, 2, 4, 8, 16, 32):
        band_mask = didx <= bw
        energy = np.sum(off[band_mask]**2) / total_energy
        print(f"  bandwidth={bw:3d}: {100*energy:.1f}% of off-diag energy")

    # --- Structure check 4: same-m-only bandwidth in dL. If off-diagonal
    # energy is dominated by same-m pairs (as check 1's |dm| breakdown
    # suggests), the natural correction is block-diagonal-per-m with a small
    # bandwidth in L within each block -- O(n_alm * bandwidth) rather than
    # O(n_alm^2/3) for a generic dense/low-rank treatment.
    same_m = (dm == 0)
    np.fill_diagonal(same_m, False)
    energy_same_m = np.sum(off[same_m]**2)
    print(f"\nFraction of total off-diag energy from same-m pairs: {100*energy_same_m/total_energy:.1f}%")

    print("Energy captured by dL-bandwidth truncation, RESTRICTED to same-m pairs:")
    for bw in (2, 4, 6, 8, 10, 12):
        band_mask = same_m & (dL <= bw)
        energy = np.sum(off[band_mask]**2) / total_energy
        print(f"  same-m, |dL|<={bw:3d}: {100*energy:.1f}% of TOTAL off-diag energy")


if __name__ == "__main__":
    main()
