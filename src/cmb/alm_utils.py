import functools

import numpy as np

try:
    import healpy as hp
    import scipy as sp
except Exception:
    hp = None
    sp = None


@functools.lru_cache(maxsize=8)
def _ordering_indices(lmax):
    """Precompute ho<->mo index permutations for a given lmax (cached)."""
    n = lmax * (lmax + 1) // 2
    ho_to_mo = np.empty(n, dtype=np.intp)
    mo_to_ho = np.empty(n, dtype=np.intp)
    ho_idx = 0
    for m in range(lmax):
        for L in range(m, lmax):
            mo_idx = L * (L + 1) // 2 + m
            ho_idx_val = m * lmax - m * (m - 1) // 2 + (L - m)
            ho_to_mo[ho_idx] = mo_idx
            mo_to_ho[mo_idx] = ho_idx_val
            ho_idx += 1
    ho_to_mo.flags.writeable = False
    mo_to_ho.flags.writeable = False
    return ho_to_mo, mo_to_ho


def cltoalm(_cls, _NSIDE, _lmax):
    """Generate alms from cls (original implementation - may be experimental)."""
    _alms = []
    _count = 0
    for L in range(_lmax):
        if _cls[L] > 0:
            _alms.append(complex(np.random.normal(0, _cls[L]), 0))
        else:
            _alms.append(complex(0, 0))

        for m in range(L + 1):
            if _cls[L] > 0 and _cls[m] > 0:
                _alms.append(
                    complex(
                        np.random.normal(0, 0.5 * _cls[L]),
                        np.random.normal(0, 0.5 * _cls[m]),
                    )
                )
            if _cls[L] > 0 and _cls[m] <= 0:
                _alms.append(complex(np.random.normal(0, 0.5 * _cls[L]), 0))
            if _cls[L] <= 0 and _cls[m] > 0:
                _alms.append(complex(0, np.random.normal(0, 0.5 * _cls[m])))
            else:
                _alms.append(complex(0, 0))
    return _alms


def hpcltoalm(_cls, _NSIDE, _lmax):
    if hp is None:
        raise ImportError("healpy is required for hpcltoalm")
    return hp.synalm(_cls, _lmax - 1, new=True)


def cltomap(_cls, _NSIDE, _lmax):
    _alm = cltoalm(_cls, _NSIDE, _lmax)
    return almtomap(_alm, _NSIDE, _lmax)


def hpcltomap(_cls, _NSIDE, _lmax):
    if hp is None:
        raise ImportError("healpy is required for hpcltomap")
    return hp.synfast(_cls, _NSIDE, _lmax - 1, new=True)


def hpmaptocl(_map, _NSIDE, _lmax):
    if hp is None:
        raise ImportError("healpy is required for hpmaptocl")
    return hp.anafast(_map, lmax=_lmax - 1)


def maptoalm(_map):
    if sp is None:
        raise ImportError("scipy is required for maptoalm")
    _omegp = (4 * np.pi) / len(_map)
    _lmax = int(np.sqrt(len(_map) * (3 / 4)))
    _NSIDE = int(_lmax / 3)
    _alm = []
    for L in range(_lmax):
        for m in range(L + 1):
            _TpYlm = []
            for i in range(len(_map)):
                _TpYlm.append(_map[i] * np.conjugate(sphharm(m, L, i, _NSIDE)))
            _alm.append(_omegp * sum(_TpYlm))

    return np.array(_alm)


def hpmaptoalm(_map, _lmax):
    if hp is None:
        raise ImportError("healpy is required for hpmaptoalm")
    return hp.map2alm(_map, _lmax - 1)


def almtocl(_alm, lmax):
    _l = np.arange(lmax)
    _scaling = 1 / (2 * _l + 1)
    count = 0
    _new = []
    _cl = []
    for L in range(lmax):
        _new.append([])
        for m in range(L):
            if m == 0:
                _new[L].append(np.absolute(_alm[count]) ** 2)
                count = count + 1

            if m > 0:
                _new[L].append(2 * np.absolute(_alm[count]) ** 2)
                count = count + 1

    for i in range(len(_new)):
        _cl.append(_scaling[i] * sum(_new[i]))

    return _cl


def hpalmtocl(_alms, _lmax):
    if hp is None:
        raise ImportError("healpy is required for hpalmtocl")
    return hp.alm2cl(_alms, lmax=_lmax - 1)


def almtomap(_alm, _NSIDE, _lmax):
    """alm -> map using original ordering"""
    if sp is None:
        raise ImportError("scipy is required for almtomap")
    _map = []
    _Npix = 12 * (_NSIDE) ** 2

    for i in range(_Npix):
        _sum = []
        _count = 0
        for L in np.arange(0, _lmax):
            for m in np.arange(0, L + 1):
                if m == 0:
                    _sum.append(_alm[_count] * sphharm(m, L, i, _NSIDE))
                    _count = _count + 1
                else:
                    _sum.append(
                        2
                        * (
                            np.real(_alm[_count]) * np.real(sphharm(m, L, i, _NSIDE))
                            - np.imag(_alm[_count]) * np.imag(sphharm(m, L, i, _NSIDE))
                        )
                    )
                    _count = _count + 1
        _map.append(sum(_sum))

    return np.real(_map)


def hpalmtomap(_alms, _NSIDE, _lmax):
    if hp is None:
        raise ImportError("healpy is required for hpalmtomap")
    return hp.alm2map(_alms, _NSIDE, _lmax - 1)


def hpmapsmooth(_map, _lmax):
    if hp is None:
        raise ImportError("healpy is required for hpmapsmooth")
    return hp.smoothing(_map, lmax=_lmax)


def hpalmsmooth(_alms):
    if hp is None:
        raise ImportError("healpy is required for hpalmsmooth")
    return hp.smoothalm(_alms, fwhm=0.0)


def singulartosplitalm(_alm):
    _realalm, _imagalm = _alm.real, _alm.imag
    return [_realalm, _imagalm]


def splittosingularalm(_realalm, _imagalm, lmax):
    _alm = []
    _ralmcount = 0
    _ialmcount = 0
    for L in range(lmax):
        for m in range(L + 1):
            if L == 0 or L == 1:
                _alm.append(complex(0, 0))
            else:
                if m == 0 or m == 1:
                    _alm.append(complex(_realalm[_ralmcount], 0))
                    _ralmcount = _ralmcount + 1
                else:
                    _alm.append(complex(_realalm[_ralmcount], _imagalm[_ialmcount]))
                    _ralmcount = _ralmcount + 1
                    _ialmcount = _ialmcount + 1

    return _alm


@functools.lru_cache(maxsize=8)
def _alm_scatter_indices(lmax):
    """Precompute scatter indices for splittosingularalm_tf (cached by lmax)."""
    len_alm = lmax * (lmax + 1) // 2
    real_indices = np.arange(3, len_alm, dtype=np.intp)[:, np.newaxis]
    imag_indices = np.array(
        [L * (L + 1) // 2 + m for L in range(2, lmax) for m in range(2, L + 1)],
        dtype=np.intp,
    )[:, np.newaxis]
    real_indices.flags.writeable = False
    imag_indices.flags.writeable = False
    return real_indices, imag_indices, len_alm


def splittosingularalm_tf(_realalm, _imagalm, lmax):
    try:
        import tensorflow as tf
    except Exception as exc:
        raise ImportError("tensorflow is required for splittosingularalm_tf") from exc
    real_idx, imag_idx, len_alm = _alm_scatter_indices(lmax)
    real_out = tf.scatter_nd(real_idx, _realalm, shape=[len_alm])
    imag_out = tf.scatter_nd(imag_idx, _imagalm, shape=[len_alm])
    return tf.complex(real_out, imag_out)


def sphharm(m, ell, _pixno, _NSIDE):
    if hp is None or sp is None:
        raise ImportError("healpy and scipy are required for sphharm")
    _theta, _phi = hp.pix2ang(nside=_NSIDE, ipix=_pixno)
    return sp.special.sph_harm(m, ell, _phi, _theta)


def almtomap_tf(_alm, _NSIDE, _lmax, _sph, _weights=None):
    try:
        import tensorflow as tf
    except Exception as exc:
        raise ImportError("tensorflow is required for almtomap_tf") from exc

    if _weights is not None:
        _alm = _weights * _alm
    else:
        # Fallback for backward compatibility or when weights aren't precomputed
        _w = np.ones(len(_alm), dtype=np.complex128)
        _count = 0
        for L in range(_lmax):
            for m in range(L + 1):
                if m == 0:
                    _w[_count] = complex(0.5, 0)
                _count = _count + 1
        _alm = tf.convert_to_tensor(_w) * _alm

    # Compute 2 * real(sph @ alm)
    # This avoids splitting _sph into real/imag parts, saving significant memory
    return 2.0 * tf.math.real(tf.linalg.matvec(_sph, _alm))


def almmotho(_moalm, _lmax):
    """Change alm ordering from author's ordering to healpy's ordering."""
    ho_to_mo, _ = _ordering_indices(_lmax)
    return np.asarray(_moalm)[ho_to_mo]


def almhotmo(_hoalm, _lmax):
    """Change alm ordering from healpy to author's ordering."""
    _, mo_to_ho = _ordering_indices(_lmax)
    return np.asarray(_hoalm)[mo_to_ho]


def alminit(_alms, _lmax):
    _count = 0
    for L in range(_lmax):
        for _ in range(L + 1):
            if L == 0 or L == 1:
                _alms[_count] = complex(0, 0)
                _count = _count + 1
    _count = 0
    for L in range(_lmax):
        for m in range(L + 1):
            if m == 0 or m == 1:
                _alms[_count] = complex(np.real(_alms[_count]), 0)
                _count = _count + 1
            else:
                _count = _count + 1
    return _alms


def hpalminit(_alms, _lmax):
    _count = 0
    for L in range(_lmax):
        for _ in range(L + 1):
            _count = _count + 1
            if _count == 1 or _count == 2 or _count == _lmax + 1:
                _alms[_count - 1] = complex(0, 0)
    _count = 0
    for _ in range(2 * _lmax - 1):
        _alms[_count] = complex(np.real(_alms[_count]), 0)
        _count = _count + 1
    return _alms
