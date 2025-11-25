import numpy as np

try:
    import healpy as hp
    import scipy as sp
except Exception:
    hp = None
    sp = None


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


def splittosingularalm_tf(_realalm, _imagalm, lmax):
    try:
        import tensorflow as tf
    except Exception as exc:
        raise ImportError("tensorflow is required for splittosingularalm_tf") from exc
    _zero = tf.zeros(1, dtype=np.float64)
    _count = 0
    for _ in range(3):
        _realalm = tf.concat([_zero, _realalm], axis=0)
    for L in range(lmax):
        for m in range(L + 1):
            if m == 0 or m == 1:
                if L == 0:
                    _imagalm = tf.concat([_zero, _imagalm], axis=0)
                else:
                    _front = _imagalm[:_count]
                    _back = _imagalm[_count:]
                    _term = tf.concat([_zero, _back], axis=0)
                    _imagalm = tf.concat([_front, _term], axis=0)
            _count = _count + 1
    return tf.complex(_realalm, _imagalm)


def sphharm(m, ell, _pixno, _NSIDE):
    if hp is None or sp is None:
        raise ImportError("healpy and scipy are required for sphharm")
    _theta, _phi = hp.pix2ang(nside=_NSIDE, ipix=_pixno)
    return sp.special.sph_harm(m, ell, _phi, _theta)


def almtomap_tf(_alm, _NSIDE, _lmax, _sph):
    try:
        import tensorflow as tf
    except Exception as exc:
        raise ImportError("tensorflow is required for almtomap_tf") from exc
    _ones = np.ones(len(_alm), dtype=np.complex128)
    _count = 0
    for L in range(_lmax):
        for m in range(L + 1):
            if m == 0:
                _ones[_count] = complex(0.5, 0)
            _count = _count + 1
    _ones = tf.convert_to_tensor(_ones)
    _alm = _ones * _alm
    _ralm = tf.math.real(_alm)
    _ialm = tf.math.imag(_alm)
    _rsph = tf.math.real(_sph)
    _isph = tf.math.imag(_sph)

    _map1 = tf.linalg.matvec(_rsph, _ralm)
    _map2 = tf.linalg.matvec(_isph, _ialm)
    _map = 2 * (_map1 - _map2)
    return _map
    # almtomap_tf2 was removed - it relied on a module-level _sph and


def almmotho(_moalm, _lmax):
    """Change alm ordering from author's ordering to healpy's ordering."""
    _hoalm = []
    _count4 = []
    _count5 = 0
    for i in np.arange(2, _lmax + 2):
        _count4.append(_count5)
        _count5 = _count5 + i
    for i in range(_lmax):
        _count1 = 0
        for j in np.arange(i + 1, _lmax + 1):
            _hoalm.append(_moalm[_count1 + _count4[i]])
            _count1 = _count1 + j
    return np.array(_hoalm)


def almhotmo(_hoalm, _lmax):
    """Change alm ordering from healpy to author's ordering."""
    _moalm = np.zeros(sum(np.arange(_lmax + 1)), dtype=complex)
    _count4 = []
    _count5 = 0
    for i in np.arange(2, _lmax + 2):
        _count4.append(_count5)
        _count5 = _count5 + i
    _count1 = 0
    for i in range(_lmax):
        _count2 = 0
        for j in np.arange(i + 1, _lmax + 1):
            _moalm[_count2 + _count4[i]] = _hoalm[_count1]
            _count1 = _count1 + 1
            _count2 = _count2 + j
    return np.array(_moalm)


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
