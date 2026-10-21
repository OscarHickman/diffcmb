"""
URGENT diagnostic: the matrix-free ducc0 SHT (sht_ducc.py) is only accuracy-
validated against healpy at lmax=15 (tests/test_sht_ducc.py) and against the
dense sph-matrix path at lmax=12 synthetic data (tests/test_sht_ducc_model_
integration.py). Its production-scale benchmark (scripts/archive/benchmark_sht_ducc.py)
only measures SPEED, never accuracy, at lmax=300/NSIDE=256.

A well-mixed (median ESS 80%) HMC+matrix-free-SHT chain at lmax=300 with
n_burnin=6000 produced C_l systematically LOW by 3-6x vs the trusted Phase 0
dense-matrix reference (results/lmax300_nside256_gibbs_real_double), worse at
higher l -- the same qualitative pattern as messenger's (separately explained,
under-mixing) bias, but now in a chain with excellent ESS, meaning it can't be
a mixing problem this time. This script isolates whether the matrix-free SHT
itself is accurate at production scale by comparing its forward synthesis
directly against healpy.alm2map (an independent reference implementation),
using the REAL Planck mask at lmax=300/NSIDE=256 -- no MCMC, no model, just
the raw SHT operator, so it is cheap and decisive.
"""
import healpy as hp
import numpy as np

from diffcmb.sht_ducc import HealpixSHT

DATA_DIR = "/cosma8/data/dp004/dc-hick2/Plank"
LMAX = 300
NSIDE = 256


def main():
    mask_file = f"{DATA_DIR}/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"
    raw_mask = hp.read_map(mask_file, field=0)
    mask = hp.ud_grade(raw_mask, nside_out=NSIDE)
    unmasked_idx = np.where(mask > 0.9)[0]
    print(f"lmax={LMAX} nside={NSIDE} unmasked={len(unmasked_idx)}/{hp.nside2npix(NSIDE)}")

    sht = HealpixSHT(nside=NSIDE, lmax=LMAX, unmasked_idx=unmasked_idx, nthreads=8)
    print(f"n_alm={sht.n_alm}")

    rng = np.random.default_rng(0)
    alm = rng.standard_normal(sht.n_alm) + 1j * rng.standard_normal(sht.n_alm)
    _, ms = hp.Alm.getlm(LMAX - 1, i=np.arange(sht.n_alm))
    alm[ms == 0] = alm[ms == 0].real
    alm = alm.astype(np.complex128)

    print("\n--- Forward synthesis: ducc0 (matrix-free) vs healpy.alm2map ---")
    map_ducc_full = sht.synthesis_full(alm)
    map_hp_full = hp.alm2map(alm, NSIDE, lmax=LMAX - 1, mmax=LMAX - 1)

    rel_err_full = np.abs(map_ducc_full - map_hp_full) / (np.abs(map_hp_full).max() + 1e-300)
    print(f"full-sky max rel err (vs max|map|): {rel_err_full.max():.3e}")
    print(f"ducc full-sky map: mean={map_ducc_full.mean():.6e} std={map_ducc_full.std():.6e}")
    print(f"healpy full-sky map: mean={map_hp_full.mean():.6e} std={map_hp_full.std():.6e}")
    print(f"std ratio (ducc/healpy): {map_ducc_full.std() / map_hp_full.std():.6f}")

    map_ducc_masked = sht.masked_synthesis(alm)
    map_hp_masked = map_hp_full[unmasked_idx]
    rel_err_masked = np.abs(map_ducc_masked - map_hp_masked) / (np.abs(map_hp_masked).max() + 1e-300)
    print(f"\nmasked-subset max rel err (vs max|map|): {rel_err_masked.max():.3e}")

    print("\n--- Adjoint synthesis round-trip: A^T A should be ~identity-scaled ---")
    # Round-trip a single unit alm mode through synthesis -> adjoint_synthesis,
    # compare the recovered coefficient's magnitude to the analytic NPIX/(4pi) guess.
    e = np.zeros(sht.n_alm, dtype=np.complex128)
    for probe_i in [0, sht.n_alm // 4, sht.n_alm // 2, sht.n_alm - 1]:
        e[:] = 0
        e[probe_i] = 1.0
        m = sht.synthesis_full(e)
        back = sht.adjoint_synthesis_full(m)
        print(f"  alm[{probe_i}]=1 -> forward+adjoint recovers alm[{probe_i}]={back[probe_i].real:.4f}"
              f" (imag={back[probe_i].imag:.2e}), other-mode leakage max={np.abs(np.delete(back, probe_i)).max():.4e}")


if __name__ == "__main__":
    main()
