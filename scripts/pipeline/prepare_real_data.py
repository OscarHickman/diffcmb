"""Band-limited Planck CMB temperature map for the lmax = 64 sampler (ROADMAP T1.2a).

Writes the 2 <= l < lmax part of a Planck CMB map (default SMICA R3.00; the
SEVEM R3.01 and PR4 R4.00 maps also carry a 5' beam at nside 2048), its 5'
beam and nside-2048 pixel window divided out and the model's own transfer (Gaussian beam x nside pixel
window, power.beam_pixwin_transfer) applied, at --nside, in muK. This is the
data the forward model describes, so the run on it is the T1.2a simulation
pipeline with only the sky changed. coverage_ensemble_chain.py --data_map
reads it, checks the header against its own beam/nside/lmax, and adds the
noise realisation its likelihood assumes (SMICA's own noise is <= 0.26 muK
rms in this band, against sigma = 30).

Usage:
  PYTHONPATH=diffcmb OMP_NUM_THREADS=4 .venv/bin/python scripts/pipeline/prepare_real_data.py \
      --out results/analysis/realdata_prep/smica_l64_ns64_b5.fits
"""

import argparse
import os
import sys

import healpy as hp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # scripts/
import _paths  # noqa: E402,F401  (puts every script folder on sys.path)
from real_sky_setup import bandlimited_map  # noqa: E402

from diffcmb.power import beam_pixwin_transfer  # noqa: E402

PLANCK_DIR = "/cosma8/data/dp004/dc-hick2/Plank"
SMICA = "COM_CMB_IQU-smica_2048_R3.00_full.fits"
PLANCK_FWHM_ARCMIN = 5.0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--planck_dir", default=PLANCK_DIR)
    p.add_argument("--map", default=SMICA, help="file in --planck_dir; field 0 is read")
    p.add_argument("--nside", type=int, default=64)
    p.add_argument("--lmax", type=int, default=64, help="model band: l < lmax")
    p.add_argument("--beam_fwhm_arcmin", type=float, default=5.0,
                   help="the model's beam (coverage_ensemble_chain --beam_fwhm_arcmin)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    sky = hp.read_map(os.path.join(args.planck_dir, args.map), field=0) * 1e6  # K_CMB -> muK
    transfer = beam_pixwin_transfer(args.lmax, args.beam_fwhm_arcmin, args.nside)
    m = bandlimited_map(sky, args.nside, args.lmax, transfer, fwhm_in_arcmin=PLANCK_FWHM_ARCMIN)
    print(f"{args.map} 2 <= l < {args.lmax} at nside {args.nside}: rms {m.std():.1f} muK, "
          f"transfer {args.beam_fwhm_arcmin:g}' beam x pixwin")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    hp.write_map(args.out, m, dtype=np.float64, overwrite=True, column_units="uK_CMB", extra_header=[
        ("DATASRC", args.map[:60], "Planck CMB map, intensity"),
        ("METHOD", "harmonic band limit", "deconvolve 5' + pixwin2048, x transfer, l<LMAX"),
        ("LMAX", args.lmax, "model band l < LMAX"),
        ("BEAMFWHM", args.beam_fwhm_arcmin, "model beam [arcmin], x pixwin(NSIDE)"),
    ])
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
