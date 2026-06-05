import gc
import os

import numpy as np

from .alm import noisemapfunc
from .alm_utils import (
    almhotmo,
    hpalminit,
    hpalmtocl,
    hpalmtomap,
    hpcltoalm,
    hpmapsmooth,
    hpmaptoalm,
    splittosingularalm_tf,
)

try:
    import healpy as hp
    import scipy as sp
    import tensorflow as tf
    if tf is not None:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception as e:
                    print(f"Warning: Could not set memory growth for {gpu}: {e}")
except Exception:
    tf = None
    hp = None
    sp = None


def matvec_on_device(sph, a):
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("tensorflow is required for matvec_on_device")

    @tf.custom_gradient
    def _matvec_custom(matrix, vector):
        dev = getattr(matrix, "device", None)
        if dev:
            with tf.device(dev):
                val = tf.linalg.matvec(matrix, tf.cast(vector, matrix.dtype))
        else:
            val = tf.linalg.matvec(matrix, tf.cast(vector, matrix.dtype))

        def grad(dy):
            if dev:
                with tf.device(dev):
                    grad_x = tf.linalg.matvec(matrix, tf.cast(dy, matrix.dtype), adjoint_a=True)
            else:
                grad_x = tf.linalg.matvec(matrix, tf.cast(dy, matrix.dtype), adjoint_a=True)
            return None, tf.cast(grad_x, vector.dtype)

        return val, grad

    return _matvec_custom(sph, a)


class CosmologyAdvancedSampling:
    """A lightweight port of the notebook class into a testable class.

    This class is a direct translation and keeps behaviour; further
    refactors can split responsibilities.
    """

    def __init__(self, _lmax, _NSIDE, _noisesig, data_mode='synthetic', data_dir=None, parameterization='centered'):
        lcdm_parameters = np.array([67.74, 0.0486, 0.2589, 0.06, 0.0, 0.066])
        self.parameterization = parameterization

        if data_mode == 'synthetic':
            print("Generating synthetic data...")
            NPIX = 12 * (_NSIDE**2)
            n = np.linspace(_noisesig, _noisesig, NPIX)
            self.Ninv = np.array([1.0 / (ni**2) for ni in n])

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
            self.prior_alms = pad_prior_alms
            self.prior_cls = pad_prior_cls
            self.unmasked_idx = np.arange(NPIX)

        elif data_mode == 'real':
            print(f"Loading real Planck data from {data_dir}...")
            map_file = os.path.join(data_dir, 'COM_CMB_IQU-smica_2048_R3.00_full.fits')
            mask_file = os.path.join(data_dir, 'COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits')

            print("  Reading map...")
            raw_map = hp.read_map(map_file, field=0)
            self.prior_map = hp.ud_grade(raw_map, nside_out=_NSIDE) * 1e6
            self.NPIX = hp.nside2npix(_NSIDE)

            print("  Reading mask...")
            raw_mask = hp.read_map(mask_file, field=0)
            mask = hp.ud_grade(raw_mask, nside_out=_NSIDE)

            self.unmasked_idx = np.where(mask > 0.9)[0]
            print(f"  Unmasked fraction: {len(self.unmasked_idx)/self.NPIX:.3f}")

            self.prior_map[mask < 0.9] = 0.0

            full_Ninv = np.ones(self.NPIX) / (_noisesig**2)
            full_Ninv[mask < 0.9] = 0.0
            self.Ninv = full_Ninv

            print("  Initial anafast...")
            self.prior_alms = hp.map2alm(self.prior_map, lmax=_lmax-1)
            self.prior_cls = hp.anafast(self.prior_map, lmax=_lmax-1)
            if len(self.prior_cls) < _lmax:
                self.prior_cls = np.pad(self.prior_cls, (0, _lmax - len(self.prior_cls)))

        else:
            raise ValueError("data_mode must be 'synthetic' or 'real'")

        self.sph = None
        self.sph1 = None
        self.sph2 = None
        self.shape = None
        self.alm_weights = None
        self.multi_device = False

        r_alms_init = self.prior_alms.real
        i_alms_init = self.prior_alms.imag
        x0 = []

        for i in range(_lmax - 2):
            val = self.prior_cls[i + 2]
            if val > 0:
                x0.append(np.log(val))
            else:
                x0.append(-20.0)

        _count = 0
        for L in range(_lmax):
            for _ in range(L + 1):
                if L == 0 or L == 1:
                    _count = _count + 1
                else:
                    if _count < len(r_alms_init):
                        x0.append(r_alms_init[_count])
                    else:
                        x0.append(0.0)
                    _count = _count + 1

        _count = 0
        for L in range(_lmax):
            for m in range(L + 1):
                if m == 0 or m == 1:
                    _count = _count + 1
                else:
                    if _count < len(i_alms_init):
                        x0.append(i_alms_init[_count])
                    else:
                        x0.append(0.0)
                    _count = _count + 1

        self.lmax = _lmax
        self.NSIDE = _NSIDE
        self.noisesig = _noisesig
        self.x0 = x0

        if parameterization == 'non-centered':
            print("Initializing Non-Centered Parameterization...")
            _lnclstart = np.zeros(2)
            _lncl = np.concatenate([_lnclstart, np.array(x0[: (_lmax - 2)])])
            _sqrt_cl = np.sqrt(np.exp(_lncl))

            _count_r = 0
            _count_i = 0
            for L in range(_lmax):
                for m in range(L + 1):
                    if L < 2:
                        continue
                    scl = _sqrt_cl[L] if _sqrt_cl[L] > 1e-10 else 1.0
                    if m < 2:
                        x0[_lmax - 2 + _count_r] /= scl
                        _count_r += 1
                    else:
                        x0[_lmax - 2 + _count_r] /= scl
                        x0[(int(_lmax * (_lmax + 1) / 2) - 3 + _lmax - 2) + _count_i] /= scl
                        _count_r += 1
                        _count_i += 1
            self.x0 = x0

    def _ensure_tf_tensors(self):
        """Create TensorFlow-dependent tensors with matrix splitting to avoid 24GB allocation limit."""
        if self.sph1 is not None:
            return

        if tf is None:
            raise ImportError("tensorflow is required for tf-dependent features")

        if hp is None or sp is None:
            raise ImportError(
                "healpy and scipy are required to build spherical harmonics"
            )

        NPIX = int(self.NSIDE**2 * 12)
        len_alm = int(self.lmax * (self.lmax + 1) / 2)

        thetas_full, phis_full = hp.pix2ang(nside=self.NSIDE, ipix=np.arange(NPIX))
        thetas = thetas_full[self.unmasked_idx]
        phis = phis_full[self.unmasked_idx]
        NPIX_CROP = len(thetas)

        print(f"Pre-computing {len_alm} spherical harmonics for {NPIX_CROP} unmasked pixels...")
        try:
            from cmb_sph import compute_sph as _rust_compute_sph
            _sph_np = _rust_compute_sph(thetas, phis, self.lmax)
            print("  Using Rust extension.")
        except ImportError:
            print("  Rust extension not found, using slow Scipy fallback...")
            _sph_np = np.empty((NPIX_CROP, len_alm), dtype=np.complex64)
            col = 0
            for L in range(self.lmax):
                for m in range(L + 1):
                    vals = sp.special.sph_harm(m, L, phis, thetas)
                    if L == 0:
                        vals = vals.real.astype(np.complex64)
                    _sph_np[:, col] = vals
                    col += 1

        # Split matrix into 2 parts (by pixel) to fit in GPU allocator bins
        mid = NPIX_CROP // 2
        print(f"  Splitting matrix: Part 1 ({mid}), Part 2 ({NPIX_CROP-mid})")

        matrix_size_gb = (NPIX_CROP * len_alm * 8) / (1024**3)
        print(f"  Total matrix size: {matrix_size_gb:.2f} GB")

        # Check available GPUs
        gpus = tf.config.list_physical_devices('GPU')
        print(f"  Available GPUs: {len(gpus)}")

        if len(gpus) >= 2 and matrix_size_gb > 12.0:
            print("  Large matrix & multiple GPUs: placing Part 1 on GPU 0 and Part 2 on GPU 1")
            self.multi_device = True
            with tf.device('/GPU:0'):
                self.sph1 = tf.convert_to_tensor(_sph_np[:mid], dtype=np.complex64)
            with tf.device('/GPU:1'):
                self.sph2 = tf.convert_to_tensor(_sph_np[mid:], dtype=np.complex64)
        elif len(gpus) == 1 and matrix_size_gb > 12.0:
            print("  Large matrix & single GPU: placing Part 1 on GPU 0 and offloading Part 2 to CPU")
            self.multi_device = True
            with tf.device('/GPU:0'):
                self.sph1 = tf.convert_to_tensor(_sph_np[:mid], dtype=np.complex64)
            with tf.device('/CPU:0'):
                self.sph2 = tf.convert_to_tensor(_sph_np[mid:], dtype=np.complex64)
        else:
            self.multi_device = False
            self.sph1 = tf.convert_to_tensor(_sph_np[:mid], dtype=np.complex64)
            self.sph2 = tf.convert_to_tensor(_sph_np[mid:], dtype=np.complex64)

        del _sph_np
        gc.collect()

        _w = np.ones(len_alm, dtype=np.complex64)
        _l_idx = np.empty(len_alm, dtype=np.int32)
        _count = 0
        for L in range(self.lmax):
            for m in range(L + 1):
                if m == 0:
                    _w[_count] = complex(0.5, 0)
                _l_idx[_count] = L
                _count = _count + 1
        self.alm_weights = tf.convert_to_tensor(_w, dtype=np.complex64)
        self.l_indices = tf.convert_to_tensor(_l_idx, dtype=np.int32)

        _lw = np.ones(len_alm, dtype=np.float32)
        _count = 0
        for L in range(self.lmax):
            for m in range(L + 1):
                if m > 0:
                    _lw[_count] = 2.0
                _count = _count + 1
        self.l_weights = tf.convert_to_tensor(_lw, dtype=np.float32)

        _imag_mask = []
        for L in range(2, self.lmax):
            for m in range(2, L + 1):
                _imag_mask.append(L * (L + 1) // 2 + m)
        self.imag_indices = tf.convert_to_tensor(_imag_mask, dtype=np.int32)

        # Split data and Ninv to match split matrix
        self.prior_map1_tf = tf.convert_to_tensor(self.prior_map[self.unmasked_idx[:mid]], dtype=tf.float64)
        self.prior_map2_tf = tf.convert_to_tensor(self.prior_map[self.unmasked_idx[mid:]], dtype=tf.float64)
        self.Ninv1_tf = tf.convert_to_tensor(self.Ninv[self.unmasked_idx[:mid]], dtype=tf.float64)
        self.Ninv2_tf = tf.convert_to_tensor(self.Ninv[self.unmasked_idx[mid:]], dtype=tf.float64)
        self.sph = self.sph1 # For backward compatibility in utils

    def prior_parameters_tf(self):
        return tf.convert_to_tensor(self.x0, dtype=tf.float64)

    def psi(self, _params):
        return 0.0

    def _psi_tf_raw(self, _params):
        """Optimized log-posterior with Matrix Splitting."""
        _lmax, _NSIDE = self.lmax, self.NSIDE

        _lnclstart = tf.zeros(2, tf.float64)
        _lncl = tf.concat([_lnclstart, tf.cast(_params[: (_lmax - 2)], tf.float64)], axis=0)
        _real_p = tf.cast(_params[_lmax - 2 : (int(_lmax * (_lmax + 1) / 2) - 3 + _lmax - 2)], tf.float64)
        _imag_p = tf.cast(_params[(int(_lmax * (_lmax + 1) / 2) - 3 + _lmax - 2) :], tf.float64)

        if self.parameterization == 'non-centered':
            _cl_per_alm = tf.gather(tf.math.exp(_lncl), self.l_indices)
            _sqrt_cl = tf.math.sqrt(_cl_per_alm)
            _sqrt_cl_real = _sqrt_cl[3:]
            _realalm = _real_p * _sqrt_cl_real
            _sqrt_cl_imag = tf.gather(_sqrt_cl, self.imag_indices)
            _imagalm = _imag_p * _sqrt_cl_imag
            _psi_prior_alm = 0.5 * (tf.reduce_sum(_real_p**2) + tf.reduce_sum(_imag_p**2))
        else:
            _realalm, _imagalm = _real_p, _imag_p
            _a_tmp = splittosingularalm_tf(_realalm, _imagalm, _lmax)
            _abs_a2 = tf.cast(tf.math.abs(_a_tmp), tf.float32) ** 2
            _as = tf.math.unsorted_segment_sum(_abs_a2 * self.l_weights, self.l_indices, num_segments=_lmax)
            _psi_prior_alm = 0.5 * tf.reduce_sum(tf.cast(_as, tf.float64) / tf.math.exp(_lncl))

        _a = splittosingularalm_tf(_realalm, _imagalm, _lmax)
        _a_c64 = self.alm_weights * tf.cast(_a, tf.complex64)

        # Matrix splitting matvecs
        _Ya1 = 2.0 * tf.math.real(matvec_on_device(self.sph1, _a_c64))
        _Ya2 = 2.0 * tf.math.real(matvec_on_device(self.sph2, _a_c64))

        _psi_lik = 0.5 * (tf.reduce_sum((self.prior_map1_tf - tf.cast(_Ya1, tf.float64))**2 * self.Ninv1_tf) +
                          tf.reduce_sum((self.prior_map2_tf - tf.cast(_Ya2, tf.float64))**2 * self.Ninv2_tf))

        _l = tf.range(_lmax, dtype=tf.float64)
        _psi_cl = tf.reduce_sum((_l + 0.5) * _lncl)

        return _psi_lik + _psi_cl + _psi_prior_alm

    def psi_tf(self, _params):
        if self.sph1 is None:
            self._ensure_tf_tensors()
        if not hasattr(self, "_compiled_psi_tf"):
            use_jit = not getattr(self, "multi_device", False)
            print(f"Compiling psi_tf with jit_compile={use_jit}...")
            self._compiled_psi_tf = tf.function(self._psi_tf_raw, jit_compile=use_jit)
        return self._compiled_psi_tf(_params)
