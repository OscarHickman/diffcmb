import numpy as np

try:
    import healpy as hp
    import scipy as sp
except Exception:
    hp = None
    sp = None


def noisemapfunc(_map: np.ndarray, _var: float):
    """Add Gaussian noise with standard deviation `_var` to every pixel of `_map`.

    Returns a tuple: (noisy_map, noise_vector)
    """
    _noisevec = np.random.normal(0, _var, len(_map))
    _newmap = np.array(_map) + _noisevec
    return _newmap, _noisevec


def sphharm(m: int, ell: int, _pixno: int, _NSIDE: int):
    """Return spherical harmonic value for a pixel (wrapper around healpy/scipy)."""
    if hp is None or sp is None:
        raise ImportError("healpy and scipy are required for sphharm")
    _theta, _phi = hp.pix2ang(nside=_NSIDE, ipix=_pixno)
    return sp.special.sph_harm(m, ell, _phi, _theta)
