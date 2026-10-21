"""Real-sky ingredients for the pipeline-on-a-simulation test (ROADMAP T1.2a).

The truth is still drawn from the model's own priors; only the observing
geometry is real or realistic: the Planck common temperature mask, the beam and
pixel window, and a spatially varying noise map. Planck's per-pixel noise
covariance is not in this repo, so the default noise map is an ecliptic-latitude
surrogate (more hits near the ecliptic poles, so lower noise there) and is
labelled as such wherever it is saved.
"""
import healpy as hp
import numpy as np

MASK_THRESHOLD = 0.9  # the same cut model.py applies to the real-data mask


def load_mask(path, nside, threshold=MASK_THRESHOLD):
    """Boolean per-pixel keep-mask (True = observed) at ``nside``."""
    raw = hp.read_map(path, field=0)
    keep = hp.ud_grade(raw, nside_out=nside) > threshold
    if not keep.any():
        raise ValueError(f"mask {path} leaves no pixels at nside={nside}")
    return keep


def ecliptic_noise_map(nside, sigma, amplitude=0.5):
    """sigma * (1 + amplitude * cos^2 beta) per pixel, beta = ecliptic latitude.

    ``sigma`` is the noise at the ecliptic poles, ``amplitude`` the fractional
    increase at the ecliptic plane (0 gives uniform noise).
    """
    if amplitude < 0:
        raise ValueError("amplitude must be >= 0")
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    rot = hp.Rotator(coord=["G", "E"])
    theta_e, _ = rot(theta, phi)
    return sigma * (1.0 + amplitude * np.sin(theta_e) ** 2)  # cos(beta) = sin(colatitude)


def bandlimited_map(sky, nside_out, lmax, transfer, fwhm_in_arcmin=5.0, iters=3):
    """The l < lmax part of a full-resolution map, as the model's forward
    operator describes it, synthesised at ``nside_out``.

    The map's own Gaussian beam (``fwhm_in_arcmin``; SMICA's is 5') and pixel
    window are divided out, l = 0, 1 are dropped and ``transfer`` (per l,
    length lmax, e.g. power.beam_pixwin_transfer of the model) is applied.
    Power at l >= lmax is
    dropped in harmonic space: pixel-averaging instead (hp.ud_grade) leaves the
    real sky's l >= 64 power in an nside-64 map at ~2x the production noise
    (scripts/diagnostics/real_data_bandlimit_budget.py), which the band-limited model can
    only misread.
    """
    transfer = np.asarray(transfer, dtype=np.float64)
    if transfer.shape != (lmax,):
        raise ValueError(f"transfer must have length lmax={lmax}, got {transfer.shape}")
    nside_in = hp.npix2nside(len(sky))
    alm = hp.map2alm(sky, lmax=lmax - 1, iter=iters)
    t_in = hp.gauss_beam(np.radians(fwhm_in_arcmin / 60.0), lmax=lmax - 1) * hp.pixwin(nside_in, lmax=lmax - 1)
    f = transfer / t_in
    f[:2] = 0.0  # the sampler has no monopole or dipole
    return hp.alm2map(hp.almxfl(alm, f), nside_out, lmax=lmax - 1)


def read_data_map(path, nside, lmax, beam_fwhm_arcmin):
    """A prepare_real_data.py map, refused unless it was made for this model:
    the same nside, band and beam (the transfer it carries is the model's)."""
    m, header = hp.read_map(path, field=0, h=True, dtype=np.float64)
    h = dict(header)
    if hp.npix2nside(m.size) != nside or int(h.get("LMAX", -1)) != lmax:
        raise ValueError(f"{path} is nside {hp.npix2nside(m.size)}, LMAX {h.get('LMAX')}; "
                         f"the model is nside {nside}, lmax {lmax}")
    if beam_fwhm_arcmin is None or not np.isclose(float(h.get("BEAMFWHM", np.nan)), beam_fwhm_arcmin):
        raise ValueError(f"{path} carries a {h.get('BEAMFWHM')}' beam; the model's is "
                         f"{beam_fwhm_arcmin}")
    return m


def apply_mask(model, keep):
    """Observe ``model`` only where ``keep`` is True. Call before
    ``model._ensure_tf_tensors()``, which builds the SHT and Ninv_masked from
    unmasked_idx and Ninv. Shared by the chain script and the likelihood replay
    so the two cannot disagree about the geometry."""
    keep = np.asarray(keep, dtype=bool)
    if keep.shape != (model.NPIX,):
        raise ValueError(f"mask has {keep.size} pixels, model has {model.NPIX}")
    model.unmasked_idx = np.where(keep)[0]
    model.Ninv = np.where(keep, model.Ninv, 0.0)
