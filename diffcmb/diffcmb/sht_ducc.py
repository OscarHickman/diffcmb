"""Matrix-free spin-0 SHT via ducc0 (Phase 1.5).

Replaces the dense `sph` matrix (O(lmax^2 * Npix), ~570 GB at lmax=300 /
NSIDE=256, 96% of which falls back to CPU matvecs — see ROADMAP.md Phase
1.5) with ducc0's C++ synthesis/adjoint_synthesis, which operate in
O(Npix) memory and run the same transform in milliseconds on CPU.

Convention
----------
alm are standard healpy-ordered (triangular, m=0..lmax-1 major, column-major
by m — `hp.Alm.getidx` compatible), spin-0, UNweighted (i.e. NOT
pre-multiplied by the 0.5-at-m=0 `alm_weights` factor used elsewhere in this
codebase for the dense `2*real(Y @ alm_weighted)` trick). `ducc_synthesis`
already returns the correct real map directly — verified against
`healpy.alm2map` to ~1e-13 relative error (see tests/test_sht_ducc.py).

The public entry point is `masked_synthesis_tf`, a `tf.custom_gradient`
mapping (real/imag alm coefficient tensors) -> real map on a chosen subset
of unmasked pixels, drop-in for the masked-pixel restriction of
`Y @ alm` used throughout `model.py`. Its gradient is derived from the
adjoint identity

    sum(Synthesis(a) * m) == sum_lm  w_lm * Re(conj(a_lm) * AdjointSynthesis(m)_lm)

with w_lm = 2 for m>0, 1 for m=0 (the same convention as this codebase's
`alm_weights`, scaled by 2 — see docstring derivation in
`_masked_synthesis_grad`) — verified numerically to 1e-16 relative error
against the <Ya, m> = <a, Y^T m> pairing, and cross-checked against
`tf.test.compute_gradient`-style finite differences in
tests/test_sht_ducc.py.
"""

import numpy as np

try:
    import ducc0
except ImportError:
    ducc0 = None

try:
    import healpy as hp
except ImportError:
    hp = None

try:
    import tensorflow as tf
except ImportError:
    tf = None


def _require_deps():
    if ducc0 is None:
        raise ImportError("ducc0 is required for sht_ducc")
    if hp is None:
        raise ImportError("healpy is required for sht_ducc")
    if tf is None:
        raise ImportError("tensorflow is required for sht_ducc")


class HealpixSHT:
    """Caches the ring geometry and alm-weight vector for one (nside, lmax)."""

    def __init__(self, nside, lmax, unmasked_idx=None, nthreads=0):
        _require_deps()
        self.nside = nside
        self.lmax = lmax
        self.mmax = lmax - 1
        self.npix = hp.nside2npix(nside)
        self.nthreads = nthreads
        self.unmasked_idx = (
            np.asarray(unmasked_idx, dtype=np.int64)
            if unmasked_idx is not None
            else np.arange(self.npix, dtype=np.int64)
        )
        hb = ducc0.healpix.Healpix_Base(nside=nside, scheme="RING")
        self._info = hb.sht_info()

        n_alm = hp.Alm.getsize(self.mmax, self.mmax)
        ls, ms = hp.Alm.getlm(self.mmax, i=np.arange(n_alm))
        self.n_alm = n_alm
        # w_lm = 2 for m>0, 1 for m=0 — see module docstring derivation.
        self._w = np.where(ms == 0, 1.0, 2.0)

    def synthesis_full(self, alm_ho):
        """Complex healpy-ordered alm -> real full-sky map (Npix,)."""
        alm_ho = np.ascontiguousarray(alm_ho, dtype=np.complex128)
        out = ducc0.sht.synthesis(
            alm=alm_ho[np.newaxis, :], spin=0, lmax=self.lmax - 1, mmax=self.mmax,
            nthreads=self.nthreads, **self._info,
        )
        return out[0]

    def adjoint_synthesis_full(self, map_full):
        """Real full-sky map (Npix,) -> complex healpy-ordered alm."""
        map_full = np.ascontiguousarray(map_full, dtype=np.float64)
        out = ducc0.sht.adjoint_synthesis(
            map=map_full[np.newaxis, :], spin=0, lmax=self.lmax - 1, mmax=self.mmax,
            nthreads=self.nthreads, **self._info,
        )
        return out[0]

    def masked_synthesis(self, alm_ho):
        """Complex healpy-ordered alm -> real map on unmasked_idx only."""
        return self.synthesis_full(alm_ho)[self.unmasked_idx]

    def masked_adjoint_synthesis(self, map_masked):
        """Real map on unmasked_idx -> complex healpy-ordered alm (adjoint of masked_synthesis)."""
        map_full = np.zeros(self.npix, dtype=np.float64)
        map_full[self.unmasked_idx] = map_masked
        return self.adjoint_synthesis_full(map_full)


def masked_synthesis_tf(alm_ho_tf, sht: HealpixSHT):
    """tf.custom_gradient wrapping HealpixSHT.masked_synthesis.

    alm_ho_tf: complex128 tensor, healpy-ordered, shape (sht.n_alm,), UNweighted.
    Returns: float64 tensor, shape (len(sht.unmasked_idx),).

    Uses `tf.py_function` (not a bare `.numpy()` call) for the ducc0 escape
    hatch, so this op can be embedded inside a `tf.function`-traced graph
    (e.g. samplers.py's `_grad_fn`/`_jt_v_fn`, or `psi_tf`'s compiled
    wrapper) rather than only working in pure eager mode — ducc0 is a
    foreign C++ library with no TF op, so some escape hatch is unavoidable,
    but `tf.py_function` is the graph-compatible way to do it (unlike plain
    `.numpy()`, which raises during tracing because the tensor is symbolic).
    """
    _require_deps()

    def _forward_np(alm_np):
        return sht.masked_synthesis(alm_np.numpy()).astype(np.float64)

    def _backward_np(dy_np):
        g = sht.masked_adjoint_synthesis(dy_np.numpy().astype(np.float64))
        # NOT conjugated: unlike matvec_on_device (whose complex output still
        # needs an external 2*real() step, so downstream gets its sign
        # convention from tf.math.real's gradient rule), this op fuses the
        # real-part extraction internally, so the raw (unconjugated) h = w*g
        # is what correctly backprops through the real/imag split in
        # splittosingularalm_tf's tf.complex(re, im) reconstruction —
        # verified empirically against the dense path in
        # tests/test_sht_ducc_model_integration.py (conj(h) matches the real
        # component but flips the sign of the imaginary component).
        return (sht._w * g).astype(np.complex128)

    @tf.custom_gradient
    def _fn(alm_tf):
        map_tf = tf.py_function(func=_forward_np, inp=[alm_tf], Tout=tf.float64)
        map_tf.set_shape([len(sht.unmasked_idx)])

        def grad(dy):
            grad_alm = tf.py_function(func=_backward_np, inp=[dy], Tout=tf.complex128)
            grad_alm.set_shape([sht.n_alm])
            return tf.cast(grad_alm, alm_tf.dtype)

        return map_tf, grad

    return _fn(alm_ho_tf)
