import numpy as np

from .alm import noisemapfunc
from .alm_utils import (
    almhotmo,
    almmotho,
    almtomap_tf,
    hpalminit,
    hpalmtocl,
    hpalmtomap,
    hpcltoalm,
    hpmapsmooth,
    hpmaptoalm,
    splittosingularalm,
    splittosingularalm_tf,
)

try:
    import healpy as hp
    import scipy as sp
    import tensorflow as tf
except Exception:
    tf = None
    hp = None
    sp = None


class CosmologyAdvancedSampling:
    """A lightweight port of the notebook class into a testable class.

    This class is a direct translation and keeps behaviour; further
    refactors can split responsibilities.
    """

    def __init__(self, _lmax, _NSIDE, _noisesig):
        lcdm_parameters = np.array([67.74, 0.0486, 0.2589, 0.06, 0.0, 0.066])

        NPIX = 12 * (_NSIDE**2)
        n = np.linspace(_noisesig, _noisesig, NPIX)
        Ninv = [1.0 / (ni**2) for ni in n]

        lcdm_cls = None
        try:
            from .power import call_CAMB_map

            lcdm_cls = call_CAMB_map(lcdm_parameters, _lmax)
        except Exception:
            # call_CAMB_map may be unavailable (missing camb) — keep None
            lcdm_cls = np.zeros(_lmax)

        notpad_lcdm_alms = hpcltoalm(lcdm_cls, _NSIDE, _lmax)
        pad_lcdm_alms = hpalminit(notpad_lcdm_alms, _lmax)
        pad_lcdm_map = hpalmtomap(pad_lcdm_alms, _NSIDE, _lmax)
        pad_lcdm_map = hpmapsmooth(pad_lcdm_map, _lmax)
        notpad_prior_map = noisemapfunc(pad_lcdm_map, n[0])[0]
        notpad_prior_halms = hpmaptoalm(notpad_prior_map, _lmax)
        pad_prior_halms = hpalminit(notpad_prior_halms, _lmax)
        pad_prior_map = hpalmtomap(pad_prior_halms, _NSIDE, _lmax)
        pad_prior_alms = almhotmo(pad_prior_halms, _lmax)
        pad_prior_cls = hpalmtocl(pad_prior_halms, _lmax)

        # Defer TensorFlow/heavy dependency work to runtime (lazy creation).
        # Creating spherical harmonics and the multtensor requires TensorFlow,
        # healpy and scipy; do this only when needed to allow lightweight
        # import of the module.
        self.sph = None
        self.shape = None
        r_alms_init = pad_prior_alms.real
        i_alms_init = pad_prior_alms.imag
        x0 = []

        for i in range(_lmax - 2):
            if pad_prior_cls[i + 2] > 0:
                x0.append(np.log(pad_prior_cls[i + 2]))
            else:
                x0.append(0)

        _count = 0
        for L in range(_lmax):
            for _ in range(L + 1):
                if L == 0 or L == 1:
                    _count = _count + 1
                else:
                    x0.append(r_alms_init[_count])
                    _count = _count + 1

        _count = 0
        for L in range(_lmax):
            for m in range(L + 1):
                if m == 0 or m == 1:
                    _count = _count + 1
                else:
                    x0.append(i_alms_init[_count])
                    _count = _count + 1

        self.lmax = _lmax
        self.NSIDE = _NSIDE
        self.noisesig = _noisesig
        self.Ninv = Ninv
        self.NPIX = NPIX

        self.lcdm_cls = lcdm_cls
        self.lcdm_alms = pad_lcdm_alms
        self.lcdm_map = pad_lcdm_map
        self.prior_cls = pad_prior_cls
        self.prior_alms = pad_prior_alms
        self.prior_map = pad_prior_map

        # shape and sph will be created lazily by _ensure_tf_tensors()
        self.shape = None
        self.sph = None
        self.x0 = x0

    def _ensure_tf_tensors(self):
        """Create TensorFlow-dependent tensors (self.sph, self.shape) on demand.

        Raises ImportError if TensorFlow (or required libs) are not available.
        """
        # already created
        if self.sph is not None and self.shape is not None:
            return

        if tf is None:
            raise ImportError("tensorflow is required for tf-dependent features")

        if hp is None or sp is None:
            raise ImportError(
                "healpy and scipy are required to build spherical harmonics"
            )

        # build spherical harmonics tensor
        # Try the Rust extension (parallel Holmes-Featherstone recurrence) first;
        # fall back to vectorized scipy calls if cmb_sph is not built.
        NPIX = int(self.NSIDE**2 * 12)
        len_alm = int(self.lmax * (self.lmax + 1) / 2)
        thetas, phis = hp.pix2ang(nside=self.NSIDE, ipix=np.arange(NPIX))
        try:
            from cmb_sph import compute_sph as _rust_compute_sph
            _sph = _rust_compute_sph(thetas, phis, self.lmax)
        except ImportError:
            _sph = np.empty((NPIX, len_alm), dtype=np.complex128)
            col = 0
            for L in range(self.lmax):
                for m in range(L + 1):
                    vals = sp.special.sph_harm(m, L, phis, thetas)
                    if L == 0:
                        vals = vals.real.astype(np.complex128)
                    _sph[:, col] = vals
                    col += 1
        self.sph = tf.convert_to_tensor(_sph, dtype=np.complex128)

        # create multtensor via helper (import here to avoid dependency)
        try:
            from .tf_helpers import multtensor

            self.shape = multtensor(self.lmax, int(self.lmax * (self.lmax + 1) / 2))
        except Exception as e:
            # cleanup and re-raise to keep state consistent
            self.sph = None
            self.shape = None
            raise ImportError("failed to create TF multtensor: %s" % e) from e

    def prior_parameters_tf(self):
        return tf.convert_to_tensor(self.x0)

    # psi and psi_tf are intentionally kept as thin wrappers referencing
    # the existing functions moved to helper modules; these can be
    # refactored further when needed.

    def psi(self, _params):
        # kept minimal - original code used many globals; here we reuse
        # class attributes to compute psi
        _params = self.x0
        _lmax = self.lmax
        _NSIDE = self.NSIDE
        _map = self.prior_map
        _Ninv = self.Ninv
        _lncl, _realalm, _imagalm = [0, 0], [], []
        for i in range(_lmax - 2):
            _lncl.append(_params[i])
        for i in range(int(_lmax * (_lmax + 1) / 2) - 3):
            _realalm.append(_params[i + _lmax - 2])
        for i in range(int(_lmax * (_lmax + 1) / 2) - (2 * _lmax - 1)):
            _imagalm.append(_params[i + _lmax - 2 + int(_lmax * (_lmax + 1) / 2) - 3])

        _d = _map
        _a = splittosingularalm(_realalm, _imagalm, _lmax)
        _Ya = hpalmtomap(almmotho(_a, _lmax), _NSIDE, _lmax)
        _BYa = _Ya

        _elem, _psi1, _psi2, _psi3 = [], [], [], []

        for i in range(len(_d)):
            _elem.append(_d[i] - _BYa[i])
            _psi1.append(0.5 * (_elem[i] ** 2) * _Ninv[i])

        _l = np.arange(_lmax)
        for i in range(len(_lncl)):
            _psi2.append((_l[i] + 0.5) * (_lncl[i]))

        _a = np.absolute(np.array(_a)) ** 2
        _as = np.matmul(self.shape.numpy(), _a)
        _psi3 = 0.5 * _as / np.exp(np.array(_lncl))

        _psi = sum(_psi1) + sum(_psi2) + sum(_psi3)
        return _psi

    def _psi_tf_raw(self, _params):
        """Core computation for psi_tf; accessed via psi_tf for the compiled version."""
        _map, _lmax, _NSIDE, _Ninv = self.prior_map, self.lmax, self.NSIDE, self.Ninv
        _lnclstart = tf.zeros(2, np.float64)
        _lncl = tf.concat([_lnclstart, _params[: (_lmax - 2)]], axis=0)
        _realalm = _params[_lmax - 2 : (int(_lmax * (_lmax + 1) / 2) - 3 + _lmax - 2)]
        _imagalm = _params[(int(_lmax * (_lmax + 1) / 2) - 3 + _lmax - 2) :]

        _d = _map
        _a = splittosingularalm_tf(_realalm, _imagalm, _lmax)
        _Ya = almtomap_tf(_a, _NSIDE, _lmax, self.sph)
        _BYa = _Ya
        _elem = _d - _BYa
        _psi1 = 0.5 * (_elem**2) * _Ninv
        _l = tf.range(_lmax, dtype=np.float64)
        _psi2 = (_l + 0.5) * _lncl

        _a = tf.math.abs(_a) ** 2
        _as = tf.linalg.matvec(self.shape, _a)
        _psi3 = 0.5 * _as / tf.math.exp(_lncl)

        return tf.reduce_sum(_psi1) + tf.reduce_sum(_psi2) + tf.reduce_sum(_psi3)

    def psi_tf(self, _params):
        if self.sph is None:
            self._ensure_tf_tensors()
        if not hasattr(self, "_compiled_psi_tf"):
            self._compiled_psi_tf = tf.function(self._psi_tf_raw)
        return self._compiled_psi_tf(_params)
