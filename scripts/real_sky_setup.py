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
