"""How much of the real SMICA map the lmax = 64 forward model cannot explain (ROADMAP T1.2a).

The real-sky simulation test draws band-limited truths (l < lmax), so its data
contain no power above the band. The real sky does. Whatever reaches the
nside-64 data map from l >= lmax, and whatever a harmonic cut spreads from the
masked region into observed pixels, is signal the likelihood can only absorb
as noise. This script measures both on the Planck 2018 SMICA map, for the
candidate preparations, against the production noise sigma:

  ud_grade   model.py's current real-data path: pixel-average 2048 -> 64. The
             model is credited with the best transfer it could carry
             (5' beam x nside-64 pixel window); the residual is the rest.
  gauss_F    harmonic: deconvolve SMICA's 5' beam and the nside-2048 pixel
             window, apply a Gaussian beam of FWHM F to all l < 3 nside and
             synthesise at nside 64. The model carries the same Gaussian for
             l < lmax; the residual is the l >= lmax part.
  cut_F      as gauss_F, then keep only 2 <= l < lmax (prepare_real_data.py):
             band-limited by construction, so the only residual is masked-region
             content spread into the observed pixels by the cut. It is not a row
             of its own: it is the 'leak' column of the gauss_F row.

The leak is measured twice: from SMICA x (1 - mask) (foreground residuals and
inpainting plus the CMB there) and from a Gaussian CMB realisation of SMICA's
own masked spectrum, x (1 - mask). The CMB part is legitimate signal the model
does describe; the excess of the data leak over the simulation leak bounds the
non-CMB contamination.

Also reported per F: the band-edge signal-to-noise per mode,
F_l^2 C_l / N_l at l = lmax - 1, with N_l = sigma^2 * 4 pi / npix.

Usage:
  PYTHONPATH=diffcmb OMP_NUM_THREADS=4 .venv/bin/python scripts/real_data_bandlimit_budget.py \
      --out results/analysis/realdata_prep/bandlimit_budget.npz
"""

import argparse
import os

import healpy as hp
import numpy as np

PLANCK_DIR = "/cosma8/data/dp004/dc-hick2/Plank"
SMICA = "COM_CMB_IQU-smica_2048_R3.00_full.fits"
MASK = "COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"
SMICA_FWHM_ARCMIN = 5.0
MASK_THRESHOLD = 0.9  # model.py / real_sky_setup.py
FWHMS_ARCMIN = (5, 30, 60, 90, 120, 160, 200, 240, 300, 360)


def deconvolved_alm(m, nside_in, lhi):
    """alm of a SMICA-resolution map to lhi, with the 5' beam and the
    input pixel window divided out."""
    alm = hp.map2alm(m, lmax=lhi, iter=1, use_pixel_weights=True)
    t = hp.gauss_beam(np.radians(SMICA_FWHM_ARCMIN / 60.0), lmax=lhi)
    t = t * hp.pixwin(nside_in, lmax=lhi)
    return hp.almxfl(alm, 1.0 / t)


def band(alm, lhi, lo, hi, fl=None):
    """alm restricted to lo <= l < hi, optionally multiplied by fl."""
    ell = hp.Alm.getlm(lhi)[0]
    f = np.ones(lhi + 1) if fl is None else fl
    return np.where((ell >= lo) & (ell < hi), alm * f[ell], 0.0)


def rms(m, keep):
    return float(np.sqrt(np.mean(m[keep] ** 2)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--planck_dir", default=PLANCK_DIR)
    p.add_argument("--nside", type=int, default=64)
    p.add_argument("--lmax", type=int, default=64, help="model band: l < lmax")
    p.add_argument("--sigma", type=float, default=30.0, help="production noise sigma [muK]")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    nside, lmax = args.nside, args.lmax
    lhi = 3 * nside - 1  # highest l an nside-64 synthesis represents without aliasing
    ld = 4 * nside  # deconvolution / ud_grade reference band
    npix = hp.nside2npix(nside)
    n_l = args.sigma**2 * 4 * np.pi / npix

    print("reading SMICA I and the common mask ...", flush=True)
    sky = hp.read_map(os.path.join(args.planck_dir, SMICA), field=0) * 1e6  # K_CMB -> muK
    nside_in = hp.npix2nside(sky.size)
    mask_in = hp.read_map(os.path.join(args.planck_dir, MASK), field=0)
    keep_in = mask_in > 0.5
    keep = hp.ud_grade(mask_in, nside_out=nside) > MASK_THRESHOLD
    print(f"  nside_in {nside_in}, f_sky(64) {keep.mean():.3f}", flush=True)

    a_sky = deconvolved_alm(sky, nside_in, ld)
    a_out = deconvolved_alm(np.where(keep_in, 0.0, sky), nside_in, ld)  # masked region only
    cl_in = hp.anafast(np.where(keep_in, sky, 0.0), lmax=ld) / keep_in.mean()
    cl_in /= (hp.gauss_beam(np.radians(SMICA_FWHM_ARCMIN / 60.0), lmax=ld) * hp.pixwin(nside_in, lmax=ld)) ** 2
    np.random.seed(args.seed)  # healpy synfast draws from numpy's global state
    sim = hp.alm2map(hp.synalm(cl_in, lmax=ld), nside_in, lmax=ld)
    a_sim_out = hp.map2alm(np.where(keep_in, 0.0, sim), lmax=ld, iter=1, use_pixel_weights=True)
    d_ud = hp.ud_grade(sky, nside_out=nside)
    del sky, sim, mask_in

    # ud_grade path: the in-band model prediction with the best transfer it can carry.
    t_ud = hp.gauss_beam(np.radians(SMICA_FWHM_ARCMIN / 60.0), lmax=ld) * hp.pixwin(nside, lmax=ld)
    m_ud = hp.alm2map(band(a_sky, ld, 0, lmax, t_ud), nside, lmax=ld)
    rows = [("ud_grade", SMICA_FWHM_ARCMIN, rms(d_ud - m_ud, keep), np.nan, np.nan,
             t_ud[lmax - 1] ** 2 * cl_in[lmax - 1] / n_l)]

    for f in FWHMS_ARCMIN:
        g = hp.gauss_beam(np.radians(f / 60.0), lmax=ld)
        oob = rms(hp.alm2map(band(a_sky, ld, lmax, lhi + 1, g), nside, lmax=ld), keep)
        leak = rms(hp.alm2map(band(a_out, ld, 2, lmax, g), nside, lmax=ld), keep)
        leak_sim = rms(hp.alm2map(band(a_sim_out, ld, 2, lmax, g), nside, lmax=ld), keep)
        snr = g[lmax - 1] ** 2 * cl_in[lmax - 1] / n_l
        rows.append((f"gauss_{f}", f, oob, leak, leak_sim, snr))

    sig_in = rms(hp.alm2map(band(a_sky, ld, 2, lmax), nside, lmax=ld), keep)
    print(f"\nin-band (2 <= l < {lmax}) signal rms on the observed sky: {sig_in:.1f} muK; "
          f"sigma = {args.sigma:g} muK; N_l = {n_l:.3g} muK^2\n")
    print(f"{'prep':<12}{'FWHM':>6}{'oob rms':>10}{'/sigma':>8}{'leak':>8}{'leak_sim':>10}"
          f"{'S/N@l=' + str(lmax - 1):>10}")
    for name, f, oob, leak, leak_sim, snr in rows:
        print(f"{name:<12}{f:>6.0f}{oob:>10.2f}{oob / args.sigma:>8.3f}{leak:>8.2f}{leak_sim:>10.2f}{snr:>10.2f}")
    print("\noob = l >= lmax residual the model cannot explain (gauss_F, ud_grade); "
          "cut_F has oob = 0 and the same leak as gauss_F.\n"
          "leak = masked-region content reaching observed pixels after the l < lmax cut; "
          "leak_sim = the same for a pure-CMB realisation.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, prep=np.array([r[0] for r in rows]), fwhm_arcmin=np.array([r[1] for r in rows]),
             oob_rms=np.array([r[2] for r in rows]), leak_rms=np.array([r[3] for r in rows]),
             leak_sim_rms=np.array([r[4] for r in rows]), snr_band_edge=np.array([r[5] for r in rows]),
             signal_rms=sig_in, sigma=args.sigma, nside=nside, lmax=lmax, f_sky=keep.mean(),
             cl_smica_masked=cl_in)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
