import gc
import os
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

    def __init__(self, _lmax, _NSIDE, _noisesig, data_mode='synthetic', data_dir=None):
        lcdm_parameters = np.array([67.74, 0.0486, 0.2589, 0.06, 0.0, 0.066])

        if data_mode == 'synthetic':
            print("Generating synthetic data...")
            NPIX = 12 * (_NSIDE**2)
            n = np.linspace(_noisesig, _noisesig, NPIX)
            Ninv = [1.0 / (ni**2) for ni in n]

            lcdm_cls = None
            try:
                from .power import call_CAMB_map

                lcdm_cls = call_CAMB_map(lcdm_parameters, _lmax)
            except Exception:
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
            
            self.prior_map = pad_prior_map
            self.NPIX = NPIX
            self.Ninv = Ninv
            self.prior_alms = pad_prior_alms
            self.prior_cls = pad_prior_cls

        elif data_mode == 'real':
            print(f"Loading real Planck data from {data_dir}...")
            # Load SMICA map and mask
            map_file = os.path.join(data_dir, 'COM_CMB_IQU-smica_2048_R3.00_full.fits')
            mask_file = os.path.join(data_dir, 'COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits')
            
            # Load and downsample to _NSIDE
            raw_map = hp.read_map(map_file, field=0)
            self.prior_map = hp.ud_grade(raw_map, nside_out=_NSIDE)
            self.NPIX = hp.nside2npix(_NSIDE)
            
            # Load mask
            raw_mask = hp.read_map(mask_file, field=0)
            mask = hp.ud_grade(raw_mask, nside_out=_NSIDE)
            # Mask out invalid pixels (set to 0 for simplicity in this implementation)
            self.prior_map[mask < 0.9] = 0.0 
            
            # Approximate noise from Half-Mission maps if available, or uniform
            # For now, start with uniform noise as placeholder for real-world integration
            self.Ninv = np.ones(self.NPIX) / (_noisesig**2)
            
            # Initial estimate of alms
            self.prior_alms = hp.map2alm(self.prior_map, lmax=_lmax)
            self.prior_cls = hp.anafast(self.prior_map, lmax=_lmax)
            
        else:
            raise ValueError("data_mode must be 'synthetic' or 'real'")

        self.sph = None
        self.shape = None
        self.alm_weights = None

        r_alms_init = self.prior_alms.real
        i_alms_init = self.prior_alms.imag
        x0 = []

        for i in range(_lmax - 2):
            if self.prior_cls[i + 2] > 0:
                x0.append(np.log(self.prior_cls[i + 2]))
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
        self.x0 = x0

    def _ensure_tf_tensors(self):
        """Create TensorFlow-dependent tensors on demand with memory optimizations."""
        if self.sph is not None and self.shape is not None:
            return

        if tf is None:
            raise ImportError("tensorflow is required for tf-dependent features")

        if hp is None or sp is None:
            raise ImportError(
                "healpy and scipy are required to build spherical harmonics"
            )

        NPIX = int(self.NSIDE**2 * 12)
        len_alm = int(self.lmax * (self.lmax + 1) / 2)
        thetas, phis = hp.pix2ang(nside=self.NSIDE, ipix=np.arange(NPIX))
        
        try:
            from cmb_sph import compute_sph as _rust_compute_sph
            _sph_np = _rust_compute_sph(thetas, phis, self.lmax)
        except ImportError:
            _sph_np = np.empty((NPIX, len_alm), dtype=np.complex128)
            col = 0
            for L in range(self.lmax):
                for m in range(L + 1):
                    vals = sp.special.sph_harm(m, L, phis, thetas)
                    if L == 0:
                        vals = vals.real.astype(np.complex128)
                    _sph_np[:, col] = vals
                    col += 1
        
        self.sph = tf.convert_to_tensor(_sph_np, dtype=np.complex128)
        
        # Critical memory optimization: delete large numpy array immediately
        del _sph_np
        gc.collect()

        # Precompute weights for almtomap_tf to avoid repeat allocations
        _w = np.ones(len_alm, dtype=np.complex128)
        _count = 0
        for L in range(self.lmax):
            for m in range(L + 1):
                if m == 0:
                    _w[_count] = complex(0.5, 0)
                _count = _count + 1
        self.alm_weights = tf.convert_to_tensor(_w, dtype=np.complex128)

        try:
            from .tf_helpers import multtensor
            self.shape = multtensor(self.lmax, len_alm)
        except Exception as e:
            self.sph = None
            self.shape = None
            raise ImportError("failed to create TF multtensor: %s" % e) from e

    def prior_parameters_tf(self):
        return tf.convert_to_tensor(self.x0)

    def psi(self, _params):
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
        
        # Use optimized almtomap_tf with precomputed weights
        _Ya = almtomap_tf(_a, _NSIDE, _lmax, self.sph, _weights=self.alm_weights)
        
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
