import gc
import os

import numpy as np

from .alm import noisemapfunc
from .alm_utils import (
    _ordering_indices,
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

    dev = getattr(sph, "device", None)

    @tf.custom_gradient
    def _matvec_custom(vector):
        if dev:
            with tf.device(dev):
                val = tf.linalg.matvec(sph, tf.cast(vector, sph.dtype))
        else:
            val = tf.linalg.matvec(sph, tf.cast(vector, sph.dtype))

        def grad(dy):
            # Mathematically: sph^H * dy = conj(sph^T * conj(dy))
            # This avoids conjugating the massive sph matrix, saving 11GB of temp memory.
            dy_c = tf.math.conj(tf.cast(dy, sph.dtype))
            if dev:
                with tf.device(dev):
                    grad_x = tf.math.conj(tf.linalg.matvec(sph, dy_c, transpose_a=True))
            else:
                grad_x = tf.math.conj(tf.linalg.matvec(sph, dy_c, transpose_a=True))
            # When sph_parts live on different GPUs, TF's autodiff must sum
            # per-part contributions to the shared alm gradient. Leaving
            # grad_x on its part's device (dev) makes that cross-device
            # accumulation silently wrong; moving it to a single common
            # device first makes the accumulation an ordinary same-device
            # sum. Verified against a linearity/symmetry check of A p :=
            # grad(psi)(p) - grad(psi)(0), which only holds with this fix
            # when parts are split across >1 GPU.
            with tf.device('/CPU:0'):
                grad_x = tf.identity(tf.cast(grad_x, vector.dtype))
            return grad_x


        return val, grad

    return _matvec_custom(a)



class CosmologyAdvancedSampling:
    """A lightweight port of the notebook class into a testable class.

    This class is a direct translation and keeps behaviour; further
    refactors can split responsibilities.
    """

    def __init__(self, _lmax, _NSIDE, _noisesig, data_mode='synthetic', data_dir=None, parameterization='centered', dtype=None, use_matrixfree_sht=False, sht_nthreads=0):
        if dtype is None:
            dtype = tf.complex64 if tf is not None else None
        self.dtype = dtype
        lcdm_parameters = np.array([67.74, 0.0486, 0.2589, 0.06, 0.0, 0.066])
        self.parameterization = parameterization
        # Phase 1.5 (ROADMAP.md): matrix-free ducc0 SHT in place of the dense
        # `sph` matrix. Opt-in only — the dense path stays the default so
        # existing production runs/checkpoints are unaffected. See
        # diffcmb/sht_ducc.py and tests/test_sht_ducc.py for validation.
        self.use_matrixfree_sht = use_matrixfree_sht
        self.sht_nthreads = sht_nthreads
        self._sht = None

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

    def build_mass_sqrt_diag(self):
        """Diagonal sqrt(M) for HMC preconditioning, ordered to match x0.

        For centered:  alm[l,m] ~ N(0, Cl[l])  → M = 1/Cl[l], scale = 1/sqrt(Cl[l])
        For non-centered: u[l,m] ~ N(0,1)       → M = 1,       scale = 1
        For lnCl[l]:  posterior width ~ 1/sqrt(2l+1) → M = 2l+1, scale = sqrt(2l+1)

        Extreme values are capped at 1000x the 1st-percentile to keep the
        condition number manageable when individual Cl estimates are near zero.
        """
        lmax = self.lmax
        cls = np.asarray(self.prior_cls, dtype=np.float64)
        mass_sqrt = np.empty(len(self.x0), dtype=np.float64)
        idx = 0

        # lnCl parameters: l = 2 .. lmax-1
        for i in range(lmax - 2):
            mass_sqrt[idx] = np.sqrt(2.0 * (i + 2) + 1.0)
            idx += 1

        if self.parameterization == 'centered':
            # real alm: L=2..lmax-1, m=0..L
            for L in range(2, lmax):
                cl = max(abs(float(cls[L])) if L < len(cls) else 0.0, 1e-30)
                scale = 1.0 / np.sqrt(cl)
                for _ in range(L + 1):
                    mass_sqrt[idx] = scale
                    idx += 1
            # imaginary alm: L=2..lmax-1, m=2..L
            for L in range(2, lmax):
                cl = max(abs(float(cls[L])) if L < len(cls) else 0.0, 1e-30)
                scale = 1.0 / np.sqrt(cl)
                for m in range(L + 1):
                    if m >= 2:
                        mass_sqrt[idx] = scale
                        idx += 1
        else:
            # non-centered: u params have N(0,1) prior → identity mass
            mass_sqrt[idx:] = 1.0
            idx = len(self.x0)

        assert idx == len(self.x0), f"mass_sqrt size mismatch: {idx} vs {len(self.x0)}"

        # Cap to limit condition number: clip anything above 1000x the 1st percentile
        lo = np.percentile(mass_sqrt, 1)
        mass_sqrt = np.clip(mass_sqrt, lo * 0.1, lo * 1000.0)
        return mass_sqrt

    def _ensure_tf_tensors(self):
        """Create TensorFlow-dependent tensors with matrix splitting to avoid 24GB allocation limit."""
        if self.sph1 is not None or self._sht is not None:
            return

        if tf is None:
            raise ImportError("tensorflow is required for tf-dependent features")

        if hp is None or sp is None:
            raise ImportError(
                "healpy and scipy are required to build spherical harmonics"
            )

        NPIX = int(self.NSIDE**2 * 12)
        len_alm = int(self.lmax * (self.lmax + 1) / 2)
        np_dtype = np.complex128 if self.dtype == tf.complex128 else np.complex64

        # alm_weights / l_indices / l_weights / imag_indices don't depend on
        # which SHT backend is used (dense matrix vs matrix-free), so build
        # them once up front.
        _w = np.ones(len_alm, dtype=np_dtype)
        _l_idx = np.empty(len_alm, dtype=np.int32)
        _count = 0
        for L in range(self.lmax):
            for m in range(L + 1):
                if m == 0:
                    _w[_count] = complex(0.5, 0)
                _l_idx[_count] = L
                _count = _count + 1
        self.alm_weights = tf.convert_to_tensor(_w, dtype=self.dtype)
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

        if self.use_matrixfree_sht:
            from .sht_ducc import HealpixSHT

            print(f"Building matrix-free ducc0 SHT for {len_alm} alm, {len(self.unmasked_idx)} unmasked pixels...")
            self._sht = HealpixSHT(
                nside=self.NSIDE, lmax=self.lmax, unmasked_idx=self.unmasked_idx,
                nthreads=self.sht_nthreads,
            )
            self.Ninv_masked = tf.convert_to_tensor(self.Ninv[self.unmasked_idx], dtype=tf.float64)
            self.prior_map_masked = tf.convert_to_tensor(self.prior_map[self.unmasked_idx], dtype=tf.float64)
            self.multi_device = False
            # `splittosingularalm_tf` (and this codebase's dense `sph` matrix)
            # use "author ordering" (row-major by (L, m)); ducc0/healpy use
            # "healpy ordering" (column-major by m) — see CLAUDE.md. Precompute
            # the gather index once so _psi_tf_raw / samplers.py can convert.
            ho_to_mo, _ = _ordering_indices(self.lmax)
            self._alm_mo_to_ho_idx = tf.convert_to_tensor(ho_to_mo, dtype=tf.int32)
            return

        thetas_full, phis_full = hp.pix2ang(nside=self.NSIDE, ipix=np.arange(NPIX))
        thetas = thetas_full[self.unmasked_idx]
        phis = phis_full[self.unmasked_idx]
        NPIX_CROP = len(thetas)

        bytes_per_val = 16 if self.dtype == tf.complex128 else 8

        # Dynamic splitting to fit in GPU/allocator limits (e.g. 10GB per part)
        MAX_PART_GB = 10.0
        pix_per_part = int((MAX_PART_GB * 1024**3) / (len_alm * bytes_per_val))
        num_parts = int(np.ceil(NPIX_CROP / pix_per_part))

        # Check available GPUs
        gpus = tf.config.list_physical_devices('GPU')
        print(f"Pre-computing {len_alm} spherical harmonics for {NPIX_CROP} unmasked pixels...")
        print(f"  Splitting matrix into {num_parts} parts of max {MAX_PART_GB} GB each")
        print(f"  Available GPUs: {len(gpus)}")

        try:
            from cmb_sph import compute_sph as _rust_compute_sph
            use_rust = True
            print("  Using Rust extension (chunk-by-chunk to save RAM).")
        except ImportError:
            use_rust = False
            print("  Rust extension not found, using slow Scipy fallback...")

        self.sph_parts = []
        self.prior_map_parts = []
        self.Ninv_parts = []

        for i in range(num_parts):
            start = i * pix_per_part
            end = min((i + 1) * pix_per_part, NPIX_CROP)

            if use_rust:
                _chunk_np = _rust_compute_sph(thetas[start:end], phis[start:end], self.lmax).astype(np_dtype)
            else:
                _chunk_np = np.empty((end - start, len_alm), dtype=np_dtype)
                col = 0
                for L in range(self.lmax):
                    for m in range(L + 1):
                        vals = sp.special.sph_harm(m, L, phis[start:end], thetas[start:end])
                        if L == 0:
                            vals = vals.real.astype(np_dtype)
                        _chunk_np[:, col] = vals
                        col += 1

            # Place first part on GPU, rest on CPU
            dev = f'/GPU:{i}' if i < len(gpus) else '/CPU:0'
            print(f"  Part {i+1}/{num_parts}: pixels {start}-{end} on {dev}")
            with tf.device(dev):
                self.sph_parts.append(tf.convert_to_tensor(_chunk_np, dtype=self.dtype))
                self.prior_map_parts.append(tf.convert_to_tensor(self.prior_map[self.unmasked_idx[start:end]], dtype=tf.float64))
                self.Ninv_parts.append(tf.convert_to_tensor(self.Ninv[self.unmasked_idx[start:end]], dtype=tf.float64))

            del _chunk_np
            gc.collect()

        self.multi_device = (num_parts > 1 or len(gpus) > 0)
        self.sph1 = self.sph_parts[0]  # Compatibility
        self.sph2 = self.sph_parts[1] if num_parts > 1 else self.sph_parts[0]  # Compatibility

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
            # Added a small epsilon in denominator to prevent division by zero and NaNs during MAP estimation/optimization
            _psi_prior_alm = 0.5 * tf.reduce_sum(tf.cast(_as, tf.float64) / (tf.math.exp(_lncl) + 1e-30))


        _a = splittosingularalm_tf(_realalm, _imagalm, _lmax)

        if self.use_matrixfree_sht:
            from .sht_ducc import masked_synthesis_tf

            _a_ho = tf.gather(_a, self._alm_mo_to_ho_idx)
            _Ya = masked_synthesis_tf(tf.cast(_a_ho, tf.complex128), self._sht)
            _psi_lik = 0.5 * tf.reduce_sum((self.prior_map_masked - _Ya) ** 2 * self.Ninv_masked)
        else:
            _a_c64 = self.alm_weights * tf.cast(_a, self.dtype)
            # Matrix splitting matvecs (loop over parts)
            _psi_lik = 0.0
            for i in range(len(self.sph_parts)):
                _Ya = 2.0 * tf.math.real(matvec_on_device(self.sph_parts[i], _a_c64))
                _psi_lik += 0.5 * tf.reduce_sum((self.prior_map_parts[i] - tf.cast(_Ya, tf.float64))**2 * self.Ninv_parts[i])

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

    # ------------------------------------------------------------------
    # Gibbs sampler support
    # ------------------------------------------------------------------

    def compute_sl_np(self, alm_flat_np):
        """Compute S_l = sum_{m=-l}^{l} |a_{lm}|^2 for l=0..lmax-1.

        alm_flat_np: 1-D numpy array = x0[lmax-2:] (real parts then imaginary parts).
        """
        lmax = self.lmax
        n_real = lmax * (lmax + 1) // 2 - 3
        real_p = alm_flat_np[:n_real]
        imag_p = alm_flat_np[n_real:]
        S = np.zeros(lmax)
        r_idx = 0
        i_idx = 0
        for L in range(2, lmax):
            for m in range(L + 1):
                re = real_p[r_idx]
                r_idx += 1
                im = imag_p[i_idx] if m >= 2 else 0.0
                if m >= 2:
                    i_idx += 1
                if m == 0:
                    S[L] += re * re
                else:
                    S[L] += 2.0 * (re * re + im * im)
        return S

    def sample_cl_given_alm(self, alm_flat_np, rng=None):
        """Sample ln(C_l) | alm for l=2..lmax-1 from the exact inverse-Gamma conditional.

        The log-posterior implied by psi_tf gives:
            C_l | alm ~ InvGamma(alpha=l-0.5, beta=S_l/2)
        where S_l = sum_{m=-l}^{l} |a_{lm}|^2.

        Returns lncl array of shape (lmax-2,).
        """
        if rng is None:
            rng = np.random.default_rng()
        lmax = self.lmax
        if np.any(~np.isfinite(alm_flat_np)):
            raise ValueError("Non-finite values (NaNs/Infs) detected in alm_flat_np during sample_cl_given_alm!")
        S = self.compute_sl_np(alm_flat_np)
        lncl = np.empty(lmax - 2)
        for i in range(lmax - 2):
            l = i + 2
            alpha = float(l) - 0.5
            s_val = S[l]
            if not np.isfinite(s_val) or s_val < 0.0:
                s_val = 0.0
            beta = max(s_val * 0.5, 1e-60)
            g = rng.gamma(alpha, scale=1.0)
            val_cl = beta / max(g, 1e-300)
            # Clip between exp(-15) and exp(15) to keep log-scale calculations stable
            val_cl = np.clip(val_cl, 3e-7, 3e6)
            lncl[i] = np.log(val_cl)
        return lncl

    def build_posterior_mass_sqrt(self, cl_full):
        """Diagonal sqrt(posterior Hessian) for HMC preconditioning of the alm block.

        Uses the diagonal approximation of the posterior precision in alm space:
            H[l,m] ≈ factor * (1/C_l + Ninv_eff)
        where Ninv_eff = f_sky * mean(Ninv_unmasked) * Npix / (4π), and
        factor = 1 for m=0 (real dof, variance C_l) or 2 for m>0 (each of
        Re/Im has variance C_l/2, so precision 2/C_l; same 2x applies to the
        noise term since d(map)/d(re_lm) carries a factor of 2 for m>0 in
        the real-map synthesis convention used by _psi_tf_raw). Matches the
        l_weights/alm_weights factors used in _psi_tf_raw and the S_l sum in
        compute_sl_np.

        For high-S/N CMB problems this nearly diagonalises the posterior,
        giving a condition number close to 1 in the whitened space.

        Returns a 1-D float64 array of length n_real_alm + n_imag_alm.
        """
        lmax = self.lmax
        n_unmasked = len(self.unmasked_idx)
        Ninv_mean = float(np.mean(self.Ninv[self.unmasked_idx])) if n_unmasked > 0 else 1.0
        f_sky = n_unmasked / self.NPIX
        Ninv_eff = f_sky * Ninv_mean * self.NPIX / (4.0 * np.pi)

        n_real = lmax * (lmax + 1) // 2 - 3
        n_imag = (lmax - 2) * (lmax - 1) // 2
        mass_sqrt = np.empty(n_real + n_imag, dtype=np.float64)
        idx = 0
        for L in range(2, lmax):
            cl = max(float(cl_full[L]) if L < len(cl_full) else 1e-30, 1e-30)
            base = 1.0 / cl + Ninv_eff
            for m in range(L + 1):
                factor = 1.0 if m == 0 else 2.0
                mass_sqrt[idx] = np.sqrt(factor * base)
                idx += 1
        for L in range(2, lmax):
            cl = max(float(cl_full[L]) if L < len(cl_full) else 1e-30, 1e-30)
            scale = np.sqrt(2.0 * (1.0 / cl + Ninv_eff))
            for m in range(L + 1):
                if m >= 2:
                    mass_sqrt[idx] = scale
                    idx += 1
        assert idx == n_real + n_imag
        return mass_sqrt
