import os
import time

import numpy as np

try:
    import tensorflow as tf
except Exception:
    tf = None

try:
    import tensorflow_probability as tfp
except Exception:
    tfp = None

try:
    from .alm_utils import (
        _alm_scatter_indices,
        almhotmo,
        almmotho,
        packed_length,
        packed_sizes,
        splittosingularalm,
        splittosingularalm_tf,
    )
    from .model import matvec_on_device
except Exception:
    _alm_scatter_indices = None
    almhotmo = None
    almmotho = None
    packed_length = None
    packed_sizes = None
    splittosingularalm = None
    splittosingularalm_tf = None
    matvec_on_device = None


PACKING_VERSION = 2


class WrapperResults:
    """A memory-efficient wrapper that mimics TFP kernel_results.

    This avoids keeping large tensors (like gradients and state histories) in GPU
    memory during sample_chain, reducing memory footprint by ~9.6 GB.
    """
    def __init__(self, target_log_prob, is_accepted, new_step_size):
        self.target_log_prob = target_log_prob
        self.is_accepted = is_accepted
        self.new_step_size = new_step_size

    @property
    def inner_results(self):
        return self

    @property
    def accepted_results(self):
        return self


def find_map_estimate(model, n_steps=500, learning_rate=0.0002, print_every=50):
    """Find the MAP estimate by minimising psi_tf with Adam.

    psi_tf is the negative log-posterior, so its minimum is the MAP.
    Requires model._ensure_tf_tensors() to have been called first.

    Returns a float64 tensor of the same shape as model.prior_parameters_tf()
    that can be passed directly as initial_state to run_chain_hmc / run_chain_nut.
    """
    if tf is None:
        raise ImportError("tensorflow is required for find_map_estimate")

    x0 = model.prior_parameters_tf()
    psi_at_x0 = float(model.psi_tf(x0))

    params = tf.Variable(x0)
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    # Wrap a single optimisation step so TF can compile it
    @tf.function
    def _step():
        with tf.GradientTape() as tape:
            loss = model._psi_tf_raw(params)
        grads = tape.gradient(loss, params)
        # Clip gradients by norm to preserve direction while preventing numeric explosion
        grads_clipped = tf.clip_by_norm(grads, 100.0)
        optimizer.apply_gradients([(grads_clipped, params)])

        # Clip lncl parameters to prevent runaway underflow/overflow
        lmax = model.lmax
        lncl_clipped = tf.clip_by_value(params[:lmax-2], -12.0, 12.0)
        params.assign(tf.concat([lncl_clipped, params[lmax-2:]], axis=0))
        return loss

    print(f"Finding MAP estimate ({n_steps} Adam steps, lr={learning_rate})...")
    print(f"  initial psi = {psi_at_x0:.6g}")
    t0 = time.time()
    loss_val = None
    for i in range(n_steps):
        loss_val = _step()
        loss_float = float(loss_val)
        if not np.isfinite(loss_float):
            raise ValueError(f"MAP optimization failed: loss became {loss_float} at step {i}")
        if i % print_every == 0 or i == n_steps - 1:
            print(f"  step {i:4d}: psi = {loss_float:.6g}  ({time.time()-t0:.1f}s)")

    improvement = psi_at_x0 - float(loss_val)
    print(f"MAP complete: psi {psi_at_x0:.6g} → {float(loss_val):.6g}  (Δ={improvement:.4g})")
    return tf.constant(params.numpy(), dtype=x0.dtype)


def run_chain_hmc(
    modelparams,
    initial_state,
    _step_size=0.1,
    num_results=1000,
    num_burnin_steps=0,
    _n_lfs=10,
    mass_sqrt_diag=None,
):
    """Run Hamiltonian Monte Carlo sampler with dual-averaging step size adaptation.

    mass_sqrt_diag: 1-D tensor of sqrt(M) values (one per parameter).  When
    provided, momentum is drawn from N(0, diag(mass_sqrt_diag^2)) so that each
    parameter is explored on its natural posterior scale.  Build it via
    model.build_mass_sqrt_diag().

    Returns the desired walks through parameter space.
    """
    if tfp is None:
        raise ImportError("tensorflow_probability is required for run_chain_hmc")
    if tf is None:
        raise ImportError("tensorflow is required for run_chain_hmc")

    # psi_tf is the negative log-posterior; negate to give TFP the log-posterior.
    def log_prob_fn(params):
        return -modelparams.psi_tf(params)

    if mass_sqrt_diag is not None:
        # Preconditioning via whitening: sample u = theta / mass_sqrt so that
        # each dimension has near-unit posterior scale.  Samples are un-whitened
        # before returning, keeping the external interface unchanged.
        mass_sqrt = tf.cast(mass_sqrt_diag, initial_state.dtype)
        def effective_log_prob(u):
            return log_prob_fn(u * mass_sqrt)
        effective_initial_state = initial_state / mass_sqrt
    else:
        mass_sqrt = None
        effective_log_prob = log_prob_fn
        effective_initial_state = initial_state

    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=effective_log_prob,
        step_size=_step_size,
        num_leapfrog_steps=_n_lfs,
    )
    adaptive_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        hmc_kernel,
        num_adaptation_steps=max(1, num_burnin_steps),
        target_accept_prob=0.65,
    )

    def trace_fn(current_state, kernel_results):
        inner = getattr(kernel_results, "inner_results", kernel_results)
        if hasattr(inner, "accepted_results"):
            target_log_prob = inner.accepted_results.target_log_prob
        else:
            target_log_prob = getattr(inner, "target_log_prob", tf.constant(0.0, dtype=tf.float64))
        is_accepted = getattr(inner, "is_accepted", tf.constant(False, dtype=tf.bool))
        step_size = getattr(kernel_results, "new_step_size", tf.constant(0.0, dtype=tf.float64))
        return (
            target_log_prob,
            is_accepted,
            step_size,
        )

    samples, trace_results = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=effective_initial_state,
        kernel=adaptive_kernel,
        trace_fn=trace_fn,
    )

    # Un-whiten: map samples back from u-space to original theta-space
    if mass_sqrt is not None:
        samples = samples * mass_sqrt

    return samples, WrapperResults(*trace_results)


def _build_inv_cl_diag(lmax, cl_full, n_real, n_imag):
    """Per-alm 1/C_l diagonal matching the real+imag alm parameter layout.

    A complex a_lm (m>0) has Re/Im parts each drawn from N(0, Cl/2), i.e.
    precision 2/Cl per dof (matches the l_weights=2.0 factor used in
    _psi_prior_alm and the S_l = ... + 2*(re^2+im^2) sum in compute_sl_np).
    The m=0 dof is purely real with variance Cl, precision 1/Cl.
    """
    inv_cl = np.empty(n_real + n_imag, dtype=np.float64)
    idx = 0
    for L in range(2, lmax):
        cl = max(float(cl_full[L]), 1e-30)
        for m in range(L + 1):
            inv_cl[idx] = (1.0 if m == 0 else 2.0) / cl
            idx += 1
    for L in range(2, lmax):
        cl = max(float(cl_full[L]), 1e-30)
        for _m in range(1, L + 1):
            inv_cl[idx] = 2.0 / cl
            idx += 1
    return inv_cl


def sample_alm_cg(model, lncl_np, rng, n_pcg_iter=50, tol=1e-6, verbose_pcg=False):
    """Exact Gaussian draw from p(alm | C_l, d) via preconditioned CG (Wandelt+2004).

    Solves A x = b_sample where:
        A        = diag(1/C_l per alm)  +  J^T N^{-1} J
        b_sample = J^T N^{-1} d  +  C_l^{-1/2} ω₁  +  J^T N^{-1/2} ω₂

    The matvec A p is obtained via ∇_alm ψ(p) − ∇_alm ψ(0) using TF autodiff.
    Diagonal preconditioner P = factor * (1/C_l + Ninv_eff), factor=1 (m=0) or
    2 (m>0)  (≈ mass_sqrt², see build_posterior_mass_sqrt) nearly diagonalises
    A for high-S/N, near-full-sky CMB; convergence expected in O(10–50) PCG
    iterations there. For a masked sky (f_sky << 1) the diagonal approximation
    misses significant off-diagonal mode-coupling from J^T N^{-1} J, and PCG
    may need many more iterations or a better preconditioner to converge.

    No accept/reject — the solution is an exact sample from the conditional.

    Returns
    -------
    alm_np : 1-D float64 array, length n_real + n_imag
    residual_norms : list of ||r|| per PCG iteration
    """
    if tf is None:
        raise ImportError("tensorflow is required for sample_alm_cg")
    if splittosingularalm_tf is None or matvec_on_device is None:
        raise ImportError("alm_utils.splittosingularalm_tf and model.matvec_on_device are required")

    lmax = model.lmax
    n_real, n_imag = packed_sizes(lmax)
    n_alm = n_real + n_imag

    lncl_full = np.zeros(lmax)
    lncl_full[2:] = lncl_np
    cl_full = np.exp(lncl_full)
    inv_cl_diag = _build_inv_cl_diag(lmax, cl_full, n_real, n_imag)
    mass_sq = model.build_posterior_mass_sqrt(cl_full) ** 2

    # Compiled ∇_alm ψ(lncl, alm) — traced once per model, reused every Gibbs step
    if not hasattr(model, "_cg_grad_fn"):
        @tf.function(jit_compile=False)
        def _grad_fn(lncl_tf, alm_tf):
            with tf.GradientTape() as tape:
                tape.watch(alm_tf)
                full_params = tf.concat([lncl_tf, alm_tf], axis=0)
                val = model._psi_tf_raw(full_params)
            return tape.gradient(val, alm_tf)
        model._cg_grad_fn = _grad_fn

    lncl_tf_c = tf.constant(lncl_np, dtype=tf.float64)

    def alm_grad(alm_np):
        return model._cg_grad_fn(lncl_tf_c, tf.constant(alm_np, dtype=tf.float64)).numpy()

    # Compiled J^T v via autodiff of ∑_i (Ya)_i v_i — traced once per model
    if not hasattr(model, "_cg_jt_v_fn"):
        _n_real_cap = n_real
        _lmax_cap = lmax

        if getattr(model, "use_matrixfree_sht", False):
            from .sht_ducc import masked_synthesis_tf

            @tf.function(jit_compile=False)
            def _jt_v_fn(v_concat, alm_zero):
                with tf.GradientTape() as tape:
                    tape.watch(alm_zero)
                    _rp = alm_zero[:_n_real_cap]
                    _ip = alm_zero[_n_real_cap:]
                    _a = splittosingularalm_tf(_rp, _ip, _lmax_cap)
                    _a_ho = tf.gather(_a, model._alm_mo_to_ho_idx)
                    _Ya = masked_synthesis_tf(tf.cast(_a_ho, tf.complex128), model._sht)
                    inner = tf.reduce_sum(_Ya * v_concat)
                return tape.gradient(inner, alm_zero)
        else:
            _part_sizes = [int(sph_p.shape[0]) for sph_p in model.sph_parts]

            @tf.function(jit_compile=False)
            def _jt_v_fn(v_concat, alm_zero):
                v_parts = tf.split(v_concat, _part_sizes)
                with tf.GradientTape() as tape:
                    tape.watch(alm_zero)
                    _rp = alm_zero[:_n_real_cap]
                    _ip = alm_zero[_n_real_cap:]
                    _a = splittosingularalm_tf(_rp, _ip, _lmax_cap)
                    _a_c = model.alm_weights * tf.cast(_a, model.dtype)
                    inner = tf.zeros((), dtype=tf.float64)
                    for i, sph_p in enumerate(model.sph_parts):
                        _Ya = 2.0 * tf.math.real(matvec_on_device(sph_p, _a_c))
                        inner = inner + tf.reduce_sum(tf.cast(_Ya, tf.float64) * v_parts[i])
                return tape.gradient(inner, alm_zero)

        model._cg_jt_v_fn = _jt_v_fn

    # Noise terms for exact sampling
    # Term 1: C_l^{-1/2} ω₁  (prior-space white noise)
    omega1 = rng.standard_normal(n_alm)
    noise_prior = np.sqrt(inv_cl_diag) * omega1

    # Term 2: J^T N^{-1/2} ω₂  (pixel-space noise projected through adjoint)
    if getattr(model, "use_matrixfree_sht", False):
        Ninv_np = model.Ninv_masked.numpy()
    else:
        Ninv_np = np.concatenate([model.Ninv_parts[i].numpy() for i in range(len(model.sph_parts))])
    omega2 = rng.standard_normal(len(Ninv_np))
    v_pix = np.sqrt(np.maximum(Ninv_np, 0.0)) * omega2
    noise_pix = model._cg_jt_v_fn(
        tf.constant(v_pix, dtype=tf.float64),
        tf.zeros(n_alm, dtype=tf.float64),
    ).numpy()

    noise_target = noise_prior + noise_pix

    # PCG: find x such that  ∇_alm ψ(x) = noise_target
    # Residual r = ∇_alm ψ(x) − noise_target = A x − b_sample
    zeros = np.zeros(n_alm, dtype=np.float64)
    minus_b_data = alm_grad(zeros)          # = ∇_alm ψ(0) = −b_data

    x = zeros.copy()
    r = minus_b_data - noise_target         # = −b_sample  at x=0
    z = r / mass_sq
    p = -z.copy()
    rz = float(np.dot(r, z))
    residual_norms = [float(np.linalg.norm(r))]

    for _it in range(n_pcg_iter):
        if residual_norms[-1] < tol:
            break
        # A p = ∇_alm ψ(p) − ∇_alm ψ(0)
        Ap = alm_grad(p) - minus_b_data
        pAp = float(np.dot(p, Ap))
        if abs(pAp) < 1e-300:
            break
        alpha = rz / pAp
        x = x + alpha * p
        r = r + alpha * Ap
        z = r / mass_sq
        rz_new = float(np.dot(r, z))
        residual_norms.append(float(np.linalg.norm(r)))
        beta = rz_new / rz
        if verbose_pcg:
            print(
                f"      it={_it:3d}  |r|={residual_norms[-1]:.3e}  alpha={alpha:.3e}  "
                f"pAp={pAp:.3e}  rz={rz:.3e}  beta={beta:.3e}"
            )
        p = -z + beta * p
        rz = rz_new

    if residual_norms[-1] > tol:
        print(f"    PCG: |r|={residual_norms[-1]:.3e} after {len(residual_norms)-1} iters (tol={tol:.0e})")

    return x, residual_norms


def _packed_to_alm_ho(packed, lmax, n_real):
    """Packed real+imag alm vector (x0/sample_alm_cg layout) -> complex healpy-ordered alm."""
    real_p, imag_p = packed[:n_real], packed[n_real:]
    alm_mo = np.array(splittosingularalm(real_p, imag_p, lmax), dtype=np.complex128)
    return almmotho(alm_mo, lmax)


def _alm_ho_to_packed(alm_ho, lmax):
    """Complex healpy-ordered alm -> packed real+imag vector (x0/sample_alm_cg layout)."""
    alm_mo = almhotmo(alm_ho, lmax)
    real_idx, imag_idx, _ = _alm_scatter_indices(lmax)
    real_p = alm_mo[real_idx.ravel()].real
    imag_p = alm_mo[imag_idx.ravel()].imag
    return np.concatenate([real_p, imag_p])


def _build_full_sky_norm_diag(lmax, n_real, n_imag, base_norm_const):
    """Analytic per-alm diagonal of A^T A for the full-sky HEALPix SHT
    operator, in the continuum-quadrature limit: base_norm_const scaled by
    the m=0-vs-m>0 factor-of-2 weight also used by _build_inv_cl_diag (each
    m>0 mode's real part reconstructs a "2*Re(a_lm)*Re(Y_lm)" real-map
    component with twice the map-space norm of the m=0 mode's single real
    dof).

    This analytic value is only ~1-2% accurate for a real (quadrature-
    approximate) HEALPix grid — and the error is systematically biased
    for the m=0 (zonal) modes specifically, not just noisy — which produces
    a detectable, non-random bias in the messenger sampler's posterior mean
    for exactly those modes (found via scripts/debug_messenger_masksky.py's
    per-mode diagnostic, z-scores 40-80 sigma for m=0 vs <8 sigma elsewhere).
    _calibrate_full_sky_norm_diag replaces this with the empirically probed
    diagonal, which _sample_alm_messenger actually uses; this analytic
    version is kept only as a cheap fallback / cross-check.
    """
    w = np.empty(n_real + n_imag, dtype=np.float64)
    idx = 0
    for L in range(2, lmax):
        for m in range(L + 1):
            w[idx] = 1.0 if m == 0 else 2.0
            idx += 1
    for L in range(2, lmax):
        for _m in range(1, L + 1):
            w[idx] = 2.0
            idx += 1
    return base_norm_const * w


def _calibrate_full_sky_norm_diag(sht_full, lmax, n_real, n_imag, progress_every=None):
    """Empirical diagonal of A^T A, probed mode-by-mode through the actual
    full-sky synthesis operator (see _build_full_sky_norm_diag's docstring
    for why the analytic NPIX/(4pi)*w_lm approximation is not accurate
    enough, particularly for m=0). Off-diagonal A^T A terms are small
    (~1% of the diagonal, verified in scripts/debug_messenger_masksky.py) so
    a diagonal calibration removes the dominant error.

    O(n_alm) full-sky synthesis calls — a one-time cost, cache the result
    on the model (see sample_alm_messenger).
    """
    n_alm = n_real + n_imag
    norm_diag = np.empty(n_alm, dtype=np.float64)
    e = np.zeros(n_alm, dtype=np.float64)
    for i in range(n_alm):
        e[i - 1 if i > 0 else 0] = 0.0
        e[i] = 1.0
        map_i = sht_full.synthesis_full(_packed_to_alm_ho(e, lmax, n_real))
        norm_diag[i] = float(np.sum(map_i ** 2))
        if progress_every and i % progress_every == 0:
            print(f"    calibrating messenger norm_diag: {i}/{n_alm}")
    return norm_diag


def _alm_index_lm(lmax, n_real, n_imag):
    """L and m for each packed-index position, matching _build_inv_cl_diag's
    layout: real parts (L=2..lmax-1, m=0..L) then imag parts (m=1..L)."""
    L_arr = np.empty(n_real + n_imag, dtype=np.int64)
    m_arr = np.empty(n_real + n_imag, dtype=np.int64)
    idx = 0
    for L in range(2, lmax):
        for m in range(L + 1):
            L_arr[idx] = L
            m_arr[idx] = m
            idx += 1
    for L in range(2, lmax):
        for m in range(1, L + 1):
            L_arr[idx] = L
            m_arr[idx] = m
            idx += 1
    return L_arr, m_arr


def _calibrate_block_AtA(sht_full, lmax, n_real, n_imag, progress_every=None, m_group_size=1):
    """Empirical block-diagonal-by-m A^T A, exact within each block (see
    messenger.sample_s_given_t_block's docstring for why same-m dominates
    >99% of the total off-diagonal energy — scripts/analyze_AtA_structure.py
    — making this a much closer approximation than the pure diagonal
    calibration while staying far cheaper than the full dense A^T A).

    m_group_size: number of consecutive m values combined into one block
    (default 1 = one block per m, exact for the dominant same-m coupling
    only). The residual cross-m coupling this discards is small in
    aggregate (~0.3% of total off-diagonal energy) but decays with |dm|
    rather than vanishing exactly (scripts/analyze_AtA_structure.py), so
    widening the block to a small window of neighbouring m's captures
    more of it at a modest extra cost — a size/accuracy knob, not a
    correctness one (m_group_size=1 already reproduces the exact same-m
    coupling within its own blocks).

    O(n_alm) full-sky synthesis calls (one map per basis vector, same cost
    as _calibrate_full_sky_norm_diag) plus O(n_alm * max_block_size) memory
    to hold one block's maps at a time — NOT O(n_alm * Npix) total, since
    maps are only held one block at a time and discarded. Still a real
    per-model, per-(lmax,NSIDE) one-time cost; unlike the diagonal
    calibration, the block Cholesky itself (build_block_cholesky) must be
    redone whenever inv_cl_diag/tau2 change (see that function's docstring),
    i.e. once per outer Gibbs sweep — the AtA_blocks returned here are
    reusable across those since they don't depend on C_l or tau2.

    Returns a list of (idx, AtA_block) pairs, idx sorted by increasing m then
    L, suitable for messenger.build_block_cholesky.
    """
    L_arr, m_arr = _alm_index_lm(lmax, n_real, n_imag)
    n_alm = n_real + n_imag
    blocks = []
    unique_m = np.unique(m_arr)
    n_done = 0
    for g in range(0, len(unique_m), m_group_size):
        m_group = unique_m[g:g + m_group_size]
        idx = np.where(np.isin(m_arr, m_group))[0]
        map_list = []
        e = np.zeros(n_alm, dtype=np.float64)
        for i in idx:
            e[:] = 0.0
            e[i] = 1.0
            map_list.append(sht_full.synthesis_full(_packed_to_alm_ho(e, lmax, n_real)))
        M = np.stack(map_list, axis=0)  # (block_size, npix)
        AtA_block = M @ M.T
        blocks.append((idx, AtA_block))
        n_done += len(idx)
        if progress_every and n_done % progress_every < len(idx):
            print(f"    calibrating messenger block A^T A: {n_done}/{n_alm} alm probed")
    return blocks


def sample_alm_messenger(
    model, lncl_np, rng, n_messenger_iter=100, tau2=None, s0=None,
    norm_diag_safety_margin=1.02, AtA=None, use_block_correction=False,
    m_group_size=1,
):
    """Exact Gaussian draw from p(alm | C_l, d) via the messenger-field Gibbs
    sampler (Elsner & Wandelt 2013), replacing sample_alm_cg's diagonal-
    preconditioned PCG, which Phase 0b (ROADMAP.md) showed cannot converge on
    the real masked sky within any tractable iteration budget.

    Requires model.use_matrixfree_sht=True (uses the ducc0 SHT wrapper,
    diffcmb.sht_ducc.HealpixSHT, for both the masked forward operator implicit
    in the data and a lazily-built FULL-SKY instance for the messenger field's
    generative reparametrisation — see messenger.py's module docstring).

    tau2: messenger covariance; must satisfy tau2 <= min(N_ii) over observed
    pixels. Defaults to 0.9 * that bound if not given.

    s0: optional packed alm vector (same layout as the return value) to warm-
    start the messenger chain from, e.g. the previous Gibbs sweep's alm state.

    norm_diag_safety_margin: safety factor (>1) inflating the calibrated
    A^T A diagonal used in the s|t harmonic-space step. A^T A is only
    diagonal to ~1-2% accuracy for a real (quadrature-approximate) HEALPix
    SHT (see _calibrate_full_sky_norm_diag); using the *exact* calibrated
    diagonal with no margin under-states the true precision by that ~1-2%
    off-diagonal coupling, which is enough to make the messenger Gibbs
    sampler's effective single-step transition operator have spectral
    radius > 1 on a masked sky — not just biased, but genuinely divergent
    (found via scripts/debug_messenger_masksky.py: |alm| grew geometrically,
    doubling roughly every ~20 outer Gibbs steps, once real Ninv=0 masking
    was applied). Inflating the diagonal is a standard majorisation trick
    (Gershgorin-style): it over-states the precision slightly, guaranteeing
    contraction at the cost of a little extra shrinkage/bias. There is a
    real stability-vs-bias trade-off here: at lmax=10/NSIDE=8, margins below
    ~1.002 still diverge (slowly), while a generous margin (1.1) stabilises
    but introduces a large bias (mean error grew to >100 posterior SEs in
    scripts/debug_messenger_masksky.py). 1.02 was chosen empirically as a
    compromise (stable with headroom, smaller bias than 1.1) but this
    trade-off is not yet resolved and is not validated at production lmax —
    see ROADMAP.md Phase 0c Step 5 / "known limitation" note. The clean fix
    is an exactly-orthonormal full-sky SHT (e.g. Gauss-Legendre/Driscoll-Healy
    quadrature via ducc0, rather than HEALPix's approximate quadrature) so
    A^T A is exactly diagonal and no margin is needed at all.

    AtA: optional (n_alm, n_alm) dense A^T A override — see
    messenger.sample_s_given_t_dense. Bypasses norm_diag_safety_margin
    entirely (exact update, no majorisation needed). Only tractable while
    n_alm is small (validation use); not a production-scale option at
    lmax=300 (O(n_alm^3) per draw). Ignored if use_block_correction=True.

    use_block_correction: use the block-diagonal-by-m A^T A correction
    (_calibrate_block_AtA / messenger.sample_s_given_t_block) instead of the
    plain diagonal approximation — captures >99% of the off-diagonal energy
    a real HEALPix SHT carries (scripts/analyze_AtA_structure.py) at
    O(n_alm * max_block_size) rather than AtA's O(n_alm^3); the scalable
    candidate fix for ROADMAP.md Phase 0c Step 5. Recalibrated (Cholesky
    factors rebuilt) every call since it depends on the current C_l draw.

    Returns alm_np : 1-D float64 array, length n_real + n_imag (same layout as
    sample_alm_cg's return value).
    """
    from .messenger import build_block_cholesky, run_messenger_gibbs
    from .sht_ducc import HealpixSHT

    if not getattr(model, "use_matrixfree_sht", False):
        raise ValueError("sample_alm_messenger requires model.use_matrixfree_sht=True")
    if model._sht is None:
        model._ensure_tf_tensors()

    lmax = model.lmax
    n_real, n_imag = packed_sizes(lmax)

    if getattr(model, "_sht_full", None) is None:
        model._sht_full = HealpixSHT(
            nside=model.NSIDE, lmax=lmax, unmasked_idx=None,
            nthreads=getattr(model, "sht_nthreads", 0),
        )
    sht_full = model._sht_full

    lncl_full = np.zeros(lmax)
    lncl_full[2:] = lncl_np
    cl_full = np.exp(lncl_full)
    inv_cl_diag = _build_inv_cl_diag(lmax, cl_full, n_real, n_imag)

    # A^T A is diagonal (to good approximation, see
    # _build_full_sky_norm_diag's docstring), calibrated empirically once per
    # model and cached (the analytic NPIX/(4pi)*w_lm guess is ~1-2% off,
    # systematically for m=0, which is enough to bias the sampler — see
    # _calibrate_full_sky_norm_diag's docstring).
    if getattr(model, "_messenger_norm_diag", None) is None:
        print(f"  Calibrating messenger A^T A diagonal ({n_real + n_imag} modes)...")
        model._messenger_norm_diag = _calibrate_full_sky_norm_diag(
            sht_full, lmax, n_real, n_imag,
            progress_every=5000 if n_real + n_imag > 5000 else None,
        )
    norm_const = model._messenger_norm_diag * norm_diag_safety_margin

    Ninv_full = np.asarray(model.Ninv, dtype=np.float64)
    d_full = np.asarray(model.prior_map, dtype=np.float64)

    if tau2 is None:
        Ninv_obs = Ninv_full[Ninv_full > 0]
        tau2 = 0.9 / float(Ninv_obs.max())

    def A_action(s):
        return sht_full.synthesis_full(_packed_to_alm_ho(s, lmax, n_real))

    def At_action(t):
        alm_ho = sht_full._w * sht_full.adjoint_synthesis_full(t)
        return _alm_ho_to_packed(alm_ho, lmax)

    block_chol = None
    if use_block_correction:
        if getattr(model, "_messenger_AtA_blocks_cache", None) is None:
            model._messenger_AtA_blocks_cache = {}
        if m_group_size not in model._messenger_AtA_blocks_cache:
            print(f"  Calibrating messenger block A^T A ({n_real + n_imag} modes, m_group_size={m_group_size})...")
            model._messenger_AtA_blocks_cache[m_group_size] = _calibrate_block_AtA(
                sht_full, lmax, n_real, n_imag,
                progress_every=5000 if n_real + n_imag > 5000 else None,
                m_group_size=m_group_size,
            )
        block_chol = build_block_cholesky(
            model._messenger_AtA_blocks_cache[m_group_size], inv_cl_diag, tau2,
        )

    return run_messenger_gibbs(
        d_full, Ninv_full, inv_cl_diag, tau2, A_action, At_action, rng,
        n_messenger_iter, s0=s0, norm_const=norm_const, AtA=AtA,
        block_chol=block_chol,
    )


def build_phi_prior_mass_sqrt(lmax, cl_phiphi_full):
    """Diagonal sqrt(prior precision) for HMC preconditioning of the phi block.

    Same real/imag alm layout as `model.build_posterior_mass_sqrt`, but keyed
    off `cl_phiphi_full` instead of the CMB C_l. Only the Gaussian prior
    curvature 1/C_l^phiphi is used — the lensing likelihood's curvature w.r.t.
    phi has no cheap diagonal estimate yet, so this is an approximate
    (prior-only) preconditioner.
    """
    n_real, n_imag = packed_sizes(lmax)
    mass_sqrt = np.empty(n_real + n_imag, dtype=np.float64)
    idx = 0
    for L in range(2, lmax):
        cl = max(float(cl_phiphi_full[L]) if L < len(cl_phiphi_full) else 1e-30, 1e-30)
        scale = np.sqrt(1.0 / cl)
        for _ in range(L + 1):
            mass_sqrt[idx] = scale
            idx += 1
    for L in range(2, lmax):
        cl = max(float(cl_phiphi_full[L]) if L < len(cl_phiphi_full) else 1e-30, 1e-30)
        scale = np.sqrt(1.0 / cl)
        for _m in range(1, L + 1):
            mass_sqrt[idx] = scale
            idx += 1
    assert idx == n_real + n_imag
    return mass_sqrt


def build_phi_posterior_mass_sqrt(lmax, cl_phiphi_full, diag_fisher_per_L):
    """Like build_phi_prior_mass_sqrt, but adds a per-L likelihood-curvature
    estimate (diag_fisher_per_L, e.g. from estimate_phi_diag_fisher in
    lensing.py) to the prior precision 1/cl_phiphi before taking the sqrt --
    the same '1/C_l + Ninv_eff' pattern model.build_posterior_mass_sqrt uses
    for the alm block. diag_fisher_per_L=0 reduces exactly to
    build_phi_prior_mass_sqrt.
    """
    n_real, n_imag = packed_sizes(lmax)
    mass_sqrt = np.empty(n_real + n_imag, dtype=np.float64)
    idx = 0
    for L in range(2, lmax):
        cl = max(float(cl_phiphi_full[L]) if L < len(cl_phiphi_full) else 1e-30, 1e-30)
        base = 1.0 / cl + float(diag_fisher_per_L[L])
        scale = np.sqrt(base)
        for _ in range(L + 1):
            mass_sqrt[idx] = scale
            idx += 1
    for L in range(2, lmax):
        cl = max(float(cl_phiphi_full[L]) if L < len(cl_phiphi_full) else 1e-30, 1e-30)
        base = 1.0 / cl + float(diag_fisher_per_L[L])
        scale = np.sqrt(base)
        for _m in range(1, L + 1):
            mass_sqrt[idx] = scale
            idx += 1
    assert idx == n_real + n_imag
    return mass_sqrt


def build_phi_block_mass_chol(lmax, cl_phiphi_full, diag_fisher_per_L, block_hessian):
    """Per-m-block Cholesky factors of the phi HMC precision matrix, adding
    lensing.py::estimate_phi_block_hessian's cross-L Nystrom correction on
    top of the diagonal prior+Fisher precision -- the structurally motivated
    fix for the cross-L Hessian coupling that no diagonal mass matrix
    ('prior' or 'fisher') can represent (see estimate_phi_block_hessian's
    docstring; ROADMAP.md/achievements.md 2026-08-17).

    Mirrors messenger.py::build_block_cholesky's block-by-m pattern (same
    codebase precedent for the analogous alm Block-2 problem): each block's
    precision = diag(1/cl_phiphi[L] + diag_fisher_per_L[L]) + H_nystrom,
    symmetrized and lightly jittered before a dense Cholesky.

    block_hessian : dict from estimate_phi_block_hessian, keyed
        (channel, m) -> (idx, H_nystrom). A missing key (block_size < 2,
        skipped there) falls back to the pure-diagonal precision for that
        block, so this degrades gracefully to build_phi_posterior_mass_sqrt
        wherever no cross-L estimate is available.

    Returns
    -------
    block_chol : list of (idx, R) pairs, idx a sorted int array of
        packed-phi positions, R the (K, K) lower-triangular Cholesky factor
        (R @ R.T = block precision), ordered by increasing m within each
        channel (real blocks first, then imag), matching
        estimate_phi_block_hessian's dict iteration order.
    """
    n_real, n_imag = packed_sizes(lmax)
    L_arr, m_arr = _alm_index_lm(lmax, n_real, n_imag)
    channel_arr = np.array(["real"] * n_real + ["imag"] * n_imag)

    def diag_precision(idx):
        out = np.empty(len(idx), dtype=np.float64)
        for k, i in enumerate(idx):
            L = int(L_arr[i])
            cl = max(float(cl_phiphi_full[L]) if L < len(cl_phiphi_full) else 1e-30, 1e-30)
            out[k] = 1.0 / cl + float(diag_fisher_per_L[L])
        return out

    block_chol = []
    for channel in ("real", "imag"):
        chan_mask = channel_arr == channel
        for m in sorted(np.unique(m_arr[chan_mask])):
            idx = np.where(chan_mask & (m_arr == m))[0]
            if len(idx) == 0:
                continue
            precision = np.diag(diag_precision(idx))
            key = (channel, int(m))
            if key in block_hessian:
                idx_h, H = block_hessian[key]
                assert np.array_equal(idx_h, idx), "block_hessian index ordering mismatch"
                precision = precision + H
            precision = 0.5 * (precision + precision.T)
            jitter = 1e-10 * np.trace(precision) / max(len(idx), 1)
            R = np.linalg.cholesky(precision + jitter * np.eye(len(idx)))
            block_chol.append((idx, R))
    return block_chol


def _dense_grad_gather(u, idx, n):
    """tf.gather(u, idx), but with a forced-dense gradient w.r.t. u (via
    scatter_nd) instead of the IndexedSlices tf.gather's default gradient
    produces. TFP's leapfrog integrator (tfp.mcmc.HamiltonianMonteCarlo)
    compares the nested structure of the gradient it gets on different
    calls within the same traced while_loop, and a gather feeding into a
    triangular_solve inside PhiWhitener.unwhiten_tf was found to
    nondeterministically switch between an IndexedSlices and a dense
    Tensor gradient across those calls, raising "structures don't have the
    same nested structure" (ValueError) at trace time for phi_mass_matrix
    ='block'. Forcing scatter_nd's inherently-dense gradient sidesteps
    this without changing any numerics (scatter_nd's gradient w.r.t. the
    updates argument is exactly the gather this replaces).
    """

    @tf.custom_gradient
    def _inner(u_):
        y = tf.gather(u_, idx)

        def grad(dy):
            return tf.scatter_nd(idx[:, tf.newaxis], dy, [n])

        return y, grad

    return _inner(u)


class PhiWhitener:
    """Applies the phi HMC/NUTS/MCLMC preconditioning transform (phi =
    unwhiten(u), u = whiten(phi)) behind one interface, so run_gibbs_chain's
    integrator-facing code doesn't need to branch between the elementwise
    'prior'/'fisher' diagonal and the block-by-m 'block' preconditioner
    (build_phi_block_mass_chol) -- see ROADMAP.md/achievements.md
    2026-08-17's cross-L Hessian coupling finding for why 'block' exists.

    Diagonal mode wraps a plain per-coordinate mass_sqrt array (sqrt
    precision; phi = u / mass_sqrt, unchanged from the pre-existing
    behaviour). Block mode wraps per-m-block Cholesky factors R (R @ R.T =
    block precision); phi = R^-T u per block via a triangular solve,
    matching the diagonal case exactly when R happens to be diagonal (its
    initial value before the block-Hessian warmup below runs).
    """

    def __init__(self, mass_sqrt_np=None, block_chol=None, n=None):
        if block_chol is not None:
            self.mode = "block"
            self.n = n if n is not None else int(sum(len(idx) for idx, _ in block_chol))
            self.block_chol = block_chol
            if tf is not None:
                self.block_chol_tf = [
                    (tf.constant(idx, dtype=tf.int32), tf.Variable(tf.constant(R, dtype=tf.float64)))
                    for idx, R in block_chol
                ]
            else:
                self.block_chol_tf = None
        else:
            self.mode = "diag"
            self.n = len(mass_sqrt_np)
            self.mass_sqrt_np = np.asarray(mass_sqrt_np, dtype=np.float64)
            if tf is not None:
                self.mass_sqrt_var = tf.Variable(tf.constant(self.mass_sqrt_np, dtype=tf.float64))
            else:
                self.mass_sqrt_var = None

    def whiten_np(self, phi_np):
        if self.mode == "diag":
            return phi_np * self.mass_sqrt_np
        u = np.empty(self.n, dtype=np.float64)
        for idx, R in self.block_chol:
            u[idx] = R.T @ phi_np[idx]
        return u

    def unwhiten_np(self, u_np):
        if self.mode == "diag":
            return u_np / self.mass_sqrt_np
        phi = np.empty(self.n, dtype=np.float64)
        for idx, R in self.block_chol:
            phi[idx] = np.linalg.solve(R.T, u_np[idx])
        return phi

    def unwhiten_tf(self, u_tf):
        if self.mode == "diag":
            return u_tf / self.mass_sqrt_var
        idx_pieces, val_pieces = [], []
        for idx_tf, R_var in self.block_chol_tf:
            u_blk = _dense_grad_gather(u_tf, idx_tf, self.n)
            x = tf.linalg.triangular_solve(
                tf.transpose(R_var), u_blk[:, tf.newaxis], lower=False
            )[:, 0]
            idx_pieces.append(idx_tf)
            val_pieces.append(x)
        flat_idx = tf.concat(idx_pieces, axis=0)[:, tf.newaxis]
        flat_val = tf.concat(val_pieces, axis=0)
        return tf.scatter_nd(flat_idx, flat_val, [self.n])

    def update_diag(self, mass_sqrt_np):
        assert self.mode == "diag"
        self.mass_sqrt_np = np.asarray(mass_sqrt_np, dtype=np.float64)
        self.mass_sqrt_var.assign(tf.constant(self.mass_sqrt_np, dtype=tf.float64))

    def update_block(self, block_chol_new):
        assert self.mode == "block"
        for (idx_old, _), (idx_new, R_new), (_, R_var) in zip(
            self.block_chol, block_chol_new, self.block_chol_tf
        ):
            assert np.array_equal(idx_old, idx_new), "block_chol index ordering changed"
            R_var.assign(tf.constant(R_new, dtype=tf.float64))
        self.block_chol = block_chol_new


def build_phi_mclmc_step(phi_log_prob_whitened, phi_step_size_var, phi_mclmc_l, phi_n_lfs):
    """Build the traced Block-3 MCLMC per-sweep step closure.

    Mirrors ``phi_hmc_bootstrap_and_step``'s single-@tf.function-per-sweep
    contract (mclmc.py's eager tf.py_function memory-leak note): the whole
    trajectory -- gradient evaluations and all -- is traced once and reused
    every sweep, not called eagerly. ``phi_log_prob_whitened`` is the same
    closure the HMC path already builds (over lncl_var/state_var or
    cg_lncl_var/cg_alm_var, depending on alm_sampler); only the integrator
    driving Block 3 changes here, not the target itself.
    """
    from .mclmc import mclmc_trajectory

    def phi_grad_fn(u):
        with tf.GradientTape() as tape:
            tape.watch(u)
            lp = phi_log_prob_whitened(u)
        return lp, tape.gradient(lp, u)

    @tf.function
    def phi_mclmc_bootstrap_and_step(state, velocity):
        return mclmc_trajectory(
            state, velocity, phi_grad_fn, phi_step_size_var, phi_mclmc_l, phi_n_lfs
        )

    return phi_mclmc_bootstrap_and_step


def run_gibbs_chain(
    model,
    n_samples=1000,
    n_burnin=500,
    hmc_step_size=0.05,
    n_lfs=20,
    target_accept=0.65,
    seed=None,
    initial_params=None,
    checkpoint_path=None,
    checkpoint_every=100,
    alm_sampler='hmc',
    n_pcg_iter=50,
    n_messenger_iter=100,
    messenger_use_block_correction=False,
    messenger_m_group_size=1,
    cl_phiphi_full=None,
    phi_initial=None,
    phi_hmc_step_size=0.05,
    phi_n_lfs=20,
    phi_target_accept=0.65,
    phi_mass_matrix='prior',
    phi_fisher_warmup_iter=20,
    phi_fisher_n_probes=8,
    phi_block_n_probes=6,
    sample_cl_phiphi=False,
    cl_phiphi_prior_nu=None,
    phi_rescale_move=False,
    phi_rescale_proposal_scale=0.1,
    phi_sampler='hmc',
    phi_mclmc_L=1.0,
    phi_nuts_max_tree_depth=10,
    phi_nuts_max_energy_diff=1000.0,
):
    """Gibbs sampler alternating exact C_l | alm (inverse-Gamma) + alm | C_l steps.

    Step 1 – C_l | alm: exact inverse-Gamma sample (O(lmax), no MCMC error).
    Step 2 – alm | C_l: controlled by `alm_sampler`:
        'hmc'  – one HMC accept/reject with diagonal mass M = sqrt(1/C_l + Ninv_eff).
                 Requires burn-in and step-size tuning; IAT grows with multipole.
        'cg'   – exact Gaussian draw via preconditioned CG (Wandelt+2004).
                 No accept/reject; IAT = 1 at all multipoles by construction.
                 n_burnin is still respected but step-size arguments are ignored.
        'messenger' – exact-in-the-limit Gaussian draw via the messenger-field
                 Gibbs sampler (Elsner & Wandelt 2013), warm-started from the
                 previous alm state each Gibbs step. Unlike 'cg', converges on
                 a masked sky (ROADMAP.md Phase 0c) — no accept/reject; requires
                 model.use_matrixfree_sht=True. n_burnin is still respected but
                 step-size arguments are ignored. messenger_use_block_correction
                 and messenger_m_group_size select the block-diagonal-by-m A^T A
                 correction validated in Phase 0c Step 5/6 (see
                 sample_alm_messenger) instead of the plain diagonal
                 approximation, which is biased on a masked sky.

    Returns (samples, logp, accepts, final_step_size) where samples shape is
    (n_samples, n_params) with the same x0 layout as the rest of the codebase.

    If checkpoint_path is provided, saves state every checkpoint_every collected samples
    and resumes from that file if it already exists (skipping burnin on resume).

    Phase 2, Block 3 (opt-in): if `cl_phiphi_full` is provided, a third Gibbs
    step samples `phi | alm, C_l, d` via HMC against `log_prob_phi_block`
    (lensing.py), using the lensed likelihood in place of the unlensed one for
    Block 2 as well. This changes the return value to a 5-tuple
    (samples, phi_samples, logp, accepts, final_step_size); phi_samples has
    shape (n_samples, n_real_alm + n_imag_alm), same packed layout as alm.
    Requires `alm_sampler='hmc'` (the CG path assumes an unlensed Gaussian
    conditional, which no longer holds once phi is in the model).

    `phi_mass_matrix` selects the Block 3 HMC preconditioner: 'prior'
    (default, unchanged behaviour) uses build_phi_prior_mass_sqrt (prior
    curvature 1/cl_phiphi only). 'fisher' additionally folds in a
    likelihood-curvature estimate (lensing.py::estimate_phi_diag_fisher,
    build_phi_posterior_mass_sqrt) computed once at burn-in iteration
    `phi_fisher_warmup_iter` (so the chain state has moved past the literal
    warm-start point first) using `phi_fisher_n_probes` coordinate samples
    per L -- addresses the prior-only mass matrix leaving the phi block
    barely mixing at production lmax (ROADMAP.md's "Simulation validation"
    entry, achievements.md).

    'block' (2026-08-18, ROADMAP.md/achievements.md) additionally folds in
    lensing.py::estimate_phi_block_hessian's cross-L Nystrom correction,
    computed at the same `phi_fisher_warmup_iter` burn-in point using
    `phi_block_n_probes` coordinate samples per m-block -- the structurally
    motivated fix for the cross-L Hessian coupling
    diagnose_phi_hessian_coupling.py found significant (mean coupling ratios
    0.263/0.621 between L-bins [60,lmax) and [10,30), up to 0.97 at
    individual chain points), which no diagonal-in-L mass matrix ('prior' or
    'fisher') can represent regardless of tuning. Applied via
    PhiWhitener/build_phi_block_mass_chol's per-m block Cholesky rather than
    an elementwise mass_sqrt. `diag_fisher_per_L`/`block_hessian` (the two
    gradient-eval-expensive quantities) are burn-in-only, frozen thereafter --
    but ARE supported with `sample_cl_phiphi=True` (below), unlike 'fisher':
    both are estimates of LIKELIHOOD curvature, independent of cl_phiphi_full,
    so freezing them is exact regardless of how many times Block 4 resamples
    the spectrum afterwards; only the diagonal 1/cl_phiphi(L) prior-precision
    term the new spectrum feeds is rebuilt every sweep (cheap: a per-m-block
    Cholesky, no new gradient evals) -- see the `sample_cl_phiphi` entry below
    (ROADMAP.md, 2026-08-23 -- lmax=64 GO used Block 4 off; this combination
    has not yet itself been gated).

    `sample_cl_phiphi` (opt-in, default False, requires `cl_phiphi_full` to
    also be given): adds a fourth Gibbs step, `C_L^phiphi | phi`, an exact
    inverse-Gamma draw with the same structure as Block 1 (see
    lensing.py::compute_sl_phi_np / sample_cl_phiphi_given_phi). When enabled,
    `cl_phiphi_full` is used only as the initial spectrum; from then on it is
    resampled every sweep from the current phi state (same pattern as
    `current_lncl`/`mass_sqrt_np` for the alm block). The phi HMC mass matrix
    is rebuilt from the new spectrum each sweep: for `phi_mass_matrix='prior'`
    (default) via build_phi_prior_mass_sqrt; for `'block'`, via
    build_phi_block_mass_chol reusing the frozen diag_fisher_per_L/
    block_hessian curvature estimates from burn-in (see above) with the new
    spectrum's diagonal term. Either way phi's whitened HMC state is rescaled
    to match -- mirroring exactly how `state_var`/`mass_sqrt_var` are rebuilt
    every sweep for the alm block. Returns an additional `cl_phiphi_samples`
    array (shape (n_samples, lmax-2), log-spectrum per sample) as the last
    element of the return tuple. Still not supported together with
    `phi_mass_matrix='fisher'` (raises ValueError) -- the same
    frozen-diagonal-Fisher + rebuilt-diagonal-prior trick used for 'block'
    above would apply equally to 'fisher' in principle, but that combination
    is out of scope for this change (not requested, not tested) and remains
    disallowed rather than silently untested.

    `phi_sampler` (opt-in, default 'hmc', requires `cl_phiphi_full` to also be
    given): selects the Block 3 integrator. 'hmc' is the validated production
    path (unchanged). 'mclmc' is a spike (ROADMAP.md, 2026-08-07) -- a hand
    -implemented microcanonical Langevin integrator (Robnik, De Luca,
    Silverstein & Seljak arXiv:2212.08549; see mclmc.py) that reuses the same
    `log_prob_phi_block` target and gradient unchanged, driven by `phi_n_lfs`
    (trajectory length, same meaning as for HMC) and `phi_mclmc_L` (momentum
    decoherence scale). MCLMC is unadjusted (no Metropolis-Hastings step) --
    `phi_hmc_step_size` is used as a FIXED step size (no Robbins-Monro
    adaptation; this is a bounded spike, not a new adaptive scheme), and
    `phi_accepts_out`/checkpointing's accept-rate bookkeeping is repurposed to
    track "trajectory did not diverge" instead of an M-H accept.
    `phi_mass_matrix='fisher'` is now also allowed with `phi_sampler='mclmc'`
    (2026-08-14 geometry-fix spike, ROADMAP.md) as a diagnostic: the
    fisher-warmup code path (see `estimate_phi_diag_fisher` below) is
    generic across both integrators, keyed only on `sample_phi and not
    phi_fisher_done`, and does not special-case which sampler consumes the
    resulting `phi_mass_sqrt_np`/`phi_state_var`. Still requires
    `sample_cl_phiphi=False` (Block 4 off), same reasoning as the
    HMC+fisher+Block4 guard above.

    `phi_sampler='nuts'` (2026-08-17 spike, ROADMAP.md -- every fixed-step-size
    tuning attempt on the phi block, HMC and MCLMC alike, has failed the
    equilibration gate regardless of lmax/window/trajectory-length; this
    tests whether a dynamic trajectory length fixes it rather than another
    fixed-schedule tuning pass) drives Block 3 with `tfp.mcmc.NoUTurnSampler`
    (no `phi_n_lfs` -- NUTS picks its own trajectory length per step via the
    no-U-turn criterion, controlled instead by `phi_nuts_max_tree_depth`/
    `phi_nuts_max_energy_diff`). Deliberately reuses the *same* external
    Robbins-Monro step-size adaptation already validated for the HMC path
    (the `phi_step_float`/`phi_target_accept` loop below), rather than TFP's
    own `DualAveragingStepSizeAdaptation` wrapper -- that wrapper's internal
    adaptation state (step count, running average) would need to persist
    across sweeps to work as intended, but this Gibbs loop calls
    `bootstrap_results` fresh every sweep (the target log-prob's alm/C_l
    conditioning changes every sweep, so a stale `kernel_results` would carry
    a wrong `target_log_prob` -- same reason the HMC branch does this), which
    would silently reset dual-averaging's adaptation state every single sweep
    and never let it converge. The externally-tracked scalar `phi_step_float`
    has no such problem: it is plain Python state, carried across sweeps (and
    checkpoints) explicitly.
    """
    if alm_sampler not in ('hmc', 'cg', 'messenger'):
        raise ValueError(f"alm_sampler must be 'hmc', 'cg', or 'messenger', got {alm_sampler!r}")
    sample_phi = cl_phiphi_full is not None
    if sample_phi and alm_sampler not in ('hmc', 'cg'):
        raise ValueError("cl_phiphi_full (Block 3) requires alm_sampler='hmc' or 'cg'")
    if sample_cl_phiphi and not sample_phi:
        raise ValueError("sample_cl_phiphi=True (Block 4) requires cl_phiphi_full to be given")
    if cl_phiphi_prior_nu is not None and not sample_cl_phiphi:
        raise ValueError(
            "cl_phiphi_prior_nu is only meaningful with sample_cl_phiphi=True "
            "(Block 4); with Block 4 off, cl_phiphi_full is already a fixed "
            "proper prior."
        )
    # Snapshot the fiducial C_L^phiphi that the proper Block 4 prior is centred
    # on. This MUST be taken before the sweep loop, which rebinds
    # cl_phiphi_full to the freshly drawn spectrum every sweep -- centring the
    # prior on that moving value would make it chase the chain and exert no
    # restoring force at all, silently reproducing the improper-prior
    # behaviour it exists to fix.
    cl_phiphi_prior_fid = (
        np.array(cl_phiphi_full, dtype=np.float64, copy=True)
        if cl_phiphi_prior_nu is not None else None
    )
    if phi_mass_matrix not in ('prior', 'fisher', 'block'):
        raise ValueError(f"phi_mass_matrix must be 'prior', 'fisher', or 'block', got {phi_mass_matrix!r}")
    if phi_rescale_move and not sample_cl_phiphi:
        raise ValueError(
            "phi_rescale_move=True requires sample_cl_phiphi=True (Block 4). "
            "The move rescales phi and C_L^phiphi jointly along the funnel "
            "axis; with a fixed spectrum there is no funnel to slide along "
            "and rescaling phi alone would target the wrong distribution."
        )
    if phi_rescale_move and phi_rescale_proposal_scale < 0.0:
        raise ValueError(
            f"phi_rescale_proposal_scale must be >= 0, got "
            f"{phi_rescale_proposal_scale}"
        )
    if sample_cl_phiphi and phi_mass_matrix == 'fisher':
        raise ValueError(
            f"sample_cl_phiphi=True is not supported together with "
            f"phi_mass_matrix={phi_mass_matrix!r} -- the Fisher-informed "
            f"mass matrix is estimated once at burn-in and frozen thereafter, "
            f"which would be inconsistent with a spectrum that keeps changing "
            f"every sweep"
        )
    if phi_sampler not in ('hmc', 'mclmc', 'nuts'):
        raise ValueError(f"phi_sampler must be 'hmc', 'mclmc', or 'nuts', got {phi_sampler!r}")
    if phi_sampler in ('mclmc', 'nuts') and not sample_phi:
        raise ValueError(f"phi_sampler={phi_sampler!r} requires cl_phiphi_full (Block 3) to be given")
    if phi_sampler == 'mclmc' and phi_mass_matrix == 'fisher' and sample_cl_phiphi:
        raise ValueError(
            "phi_sampler='mclmc' + phi_mass_matrix='fisher' is only "
            "validated with sample_cl_phiphi=False (Block 4 off) -- the "
            "fisher mass matrix is a one-time burn-in estimate, frozen "
            "thereafter, which is inconsistent with Block 4 resampling "
            "cl_phiphi_full (and rebuilding the mass matrix from it) every "
            "sweep. This is the same constraint as the HMC+fisher+Block4 "
            "guard above, now also applied to MCLMC "
            "(ROADMAP.md, 2026-08-14 geometry-fix spike)."
        )
    if alm_sampler == 'hmc' and (tf is None or tfp is None):
        raise ImportError("tensorflow and tensorflow_probability are required")
    if alm_sampler in ('cg', 'messenger') and tf is None:
        raise ImportError("tensorflow is required for the CG/messenger samplers")
    # alm_sampler='cg' + sample_phi is the Phase 2 gate-2 exactness experiment
    # (ROADMAP.md Section 1): Block 2 draws an exact Gaussian alm sample against
    # the *unlensed* operator (sample_alm_cg's target is model._psi_tf_raw,
    # which has no phi dependence) while Block 3 still applies the correct
    # lensed likelihood (log_prob_phi_block) to phi. This is deliberately an
    # approximation being tested for bias, not a claimed-correct sampler.

    rng = np.random.default_rng(seed)
    # `rng` covers only the numpy-side draws (Block 1, Block 4, mass-matrix
    # probes, CG/messenger noise). The HMC/NUTS blocks get their momenta and
    # Metropolis uniforms from TFP, and none of the `one_step` calls below
    # pass a `seed=`, so without this they draw from TensorFlow's PROCESS-
    # GLOBAL RNG stream -- making a nominally seeded chain depend on however
    # many TF random ops happened to run earlier in the same process.
    # Measured directly: consuming 5 tf.random.normal draws before an
    # otherwise identical seed=7 chain moved the recovered C_L^phiphi by 5x
    # (4.12e-7 -> 8.32e-8), which is what made
    # tests/test_samplers.py::test_gibbs_chain_sample_cl_phiphi_recovers_known_spectrum
    # pass alone and fail in a full-suite run. Seeding the global stream here
    # makes `seed` mean what it claims for both halves of the sampler.
    if seed is not None and tf is not None:
        tf.random.set_seed(seed)
    lmax = model.lmax
    n_lncl = lmax - 2
    n_real, n_imag = packed_sizes(lmax)
    n_phi = n_real + n_imag

    # --- Resume from checkpoint or initialise fresh ---
    samples_out = []
    logp_out = []
    accepts_out = []
    step_float = hmc_step_size
    resuming = False

    phi_samples_out = []
    phi_accepts_out = []
    phi_step_float = phi_hmc_step_size
    cl_phiphi_out = []

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = np.load(checkpoint_path, allow_pickle=True)
        ckpt_version = int(ckpt["packing_version"]) if "packing_version" in ckpt.files else 1
        if ckpt_version != PACKING_VERSION:
            raise ValueError(
                f"Checkpoint packing version {ckpt_version} does not match current code PACKING_VERSION {PACKING_VERSION}. "
                f"Layout changed; checkpoint cannot be resumed safely."
            )
        current_alm_np = ckpt["alm_state"].copy()
        if len(current_alm_np) != n_phi:
            raise ValueError(
                f"Checkpoint alm_state length {len(current_alm_np)} does not match current model packed length {n_phi} (lmax={lmax})."
            )
        samples_out = list(ckpt["samples"])
        logp_out = list(ckpt["logp"])
        accepts_out = list(ckpt["accepts"].tolist())
        current_lncl = ckpt["lncl_state"].copy()
        mass_sqrt_np = ckpt["mass_sqrt"].copy()
        step_float = float(ckpt["step_size"])
        resuming = True
        if sample_phi:
            phi_current_np = ckpt["phi_state"].copy()
            if len(phi_current_np) != n_phi:
                raise ValueError(
                    f"Checkpoint phi_state length {len(phi_current_np)} does not match current model packed length {n_phi} (lmax={lmax})."
                )
            phi_samples_out = list(ckpt["phi_samples"])
            phi_accepts_out = list(ckpt["phi_accepts"].tolist())
            phi_step_float = float(ckpt["phi_step_size"])
            if phi_sampler == 'mclmc':
                phi_velocity_np = ckpt["phi_velocity"].copy()
        if sample_cl_phiphi:
            # "cl_phiphi_state"/"cl_phiphi_samples" are only present in
            # checkpoints written by a sample_cl_phiphi=True run -- fall back
            # to the caller-supplied cl_phiphi_full for checkpoints from
            # before this feature existed, so old checkpoints stay loadable.
            if "cl_phiphi_state" in ckpt.files:
                cl_phiphi_full = ckpt["cl_phiphi_state"].copy()
                cl_phiphi_out = list(ckpt["cl_phiphi_samples"])
            else:
                cl_phiphi_full = np.array(cl_phiphi_full, dtype=np.float64).copy()
        print(f"Resumed from checkpoint: {len(samples_out)}/{n_samples} samples collected")
    else:
        if initial_params is not None:
            if np.any(~np.isfinite(initial_params)):
                raise ValueError("initial_params contains NaNs or Infs at start of Gibbs chain!")
            x0 = np.array(initial_params, dtype=np.float64).ravel()
        else:
            x0 = np.array(model.x0, dtype=np.float64)
        current_lncl = x0[:n_lncl].copy()
        current_alm_np = x0[n_lncl:].copy()
        cl_full = np.zeros(lmax)
        cl_full[2:] = np.exp(current_lncl)
        mass_sqrt_np = model.build_posterior_mass_sqrt(cl_full)
        if sample_phi:
            phi_current_np = (
                np.array(phi_initial, dtype=np.float64).ravel()
                if phi_initial is not None
                else np.zeros(n_phi, dtype=np.float64)
            )
            if phi_sampler == 'mclmc':
                _v0 = rng.standard_normal(n_phi)
                phi_velocity_np = _v0 / np.linalg.norm(_v0)
        if sample_cl_phiphi:
            # Local mutable working copy -- resampled every sweep from here
            # on, never mutating the caller's original array.
            cl_phiphi_full = np.array(cl_phiphi_full, dtype=np.float64).copy()

    phi_fisher_done = phi_mass_matrix not in ('fisher', 'block')  # already "done" -- no-op

    n_collected = len(samples_out)
    n_samples_remaining = n_samples - n_collected
    burnin_remaining = 0 if resuming else n_burnin

    if n_samples_remaining <= 0:
        print("All samples already collected from checkpoint.")
        if sample_phi:
            if sample_cl_phiphi:
                return (
                    np.array(samples_out, dtype=np.float64),
                    np.array(phi_samples_out, dtype=np.float64),
                    np.array(logp_out, dtype=np.float64),
                    np.array(accepts_out, dtype=bool),
                    step_float,
                    np.array(cl_phiphi_out, dtype=np.float64),
                )
            return (
                np.array(samples_out, dtype=np.float64),
                np.array(phi_samples_out, dtype=np.float64),
                np.array(logp_out, dtype=np.float64),
                np.array(accepts_out, dtype=bool),
                step_float,
            )
        return (
            np.array(samples_out, dtype=np.float64),
            np.array(logp_out, dtype=np.float64),
            np.array(accepts_out, dtype=bool),
            step_float,
        )

    cl_full = np.zeros(lmax)
    cl_full[2:] = np.exp(current_lncl)

    if sample_phi:
        from .lensing import (
            estimate_phi_block_hessian,
            estimate_phi_diag_fisher,
            log_prob_phi_block,
            psi_lensed,
            sample_cl_phiphi_given_phi,
        )

        model._ensure_tf_tensors()
        if phi_mass_matrix == 'block':
            # Starts diagonal (prior-only, zero Fisher) -- identical to
            # 'prior' -- and is upgraded in-place via PhiWhitener.update_block
            # once the burn-in warmup below computes the cross-L Nystrom
            # correction, exactly mirroring how 'fisher' starts from the
            # same prior-only mass_sqrt and is upgraded at phi_fisher_done.
            phi_whitener = PhiWhitener(
                block_chol=build_phi_block_mass_chol(
                    lmax, cl_phiphi_full, np.zeros(lmax, dtype=np.float64), {}
                )
            )
        else:
            phi_whitener = PhiWhitener(mass_sqrt_np=build_phi_prior_mass_sqrt(lmax, cl_phiphi_full))
        phi_state_var = tf.Variable(
            tf.constant(phi_whitener.whiten_np(phi_current_np), dtype=tf.float64)
        )
        phi_step_size_var = tf.Variable(phi_step_float, dtype=tf.float64)
        # Mirrors mass_sqrt_var/lncl_var below: a tf.Variable twin of the
        # plain-numpy cl_phiphi_full, so the phi-block log-prob target can be
        # traced (see hmc_bootstrap_and_step) while Block 4's per-sweep
        # spectrum update (Step 3a) still reaches it via .assign(), rather
        # than the update being silently invisible to a graph that baked in
        # the first sweep's value. The numpy cl_phiphi_full itself is left
        # untouched — cl_phiphi_out/checkpointing still read it directly.
        cl_phiphi_full_var = tf.Variable(tf.constant(cl_phiphi_full, dtype=tf.float64))
        if phi_sampler == 'mclmc':
            # MCLMC's velocity carries information between sweeps (unlike
            # HMC's momentum, which TFP refreshes internally every call), so
            # it must be a persistent Variable, not local trajectory state.
            phi_velocity_var = tf.Variable(tf.constant(phi_velocity_np, dtype=tf.float64))

    # --- Sampler-specific setup ---
    if alm_sampler == 'hmc':
        mass_sqrt_var = tf.Variable(tf.constant(mass_sqrt_np, dtype=tf.float64))
        lncl_var = tf.Variable(
            tf.constant(np.concatenate([np.zeros(2), current_lncl]), dtype=tf.float64)
        )
        state_var = tf.Variable(tf.constant(current_alm_np * mass_sqrt_np, dtype=tf.float64))
        step_size_var = tf.Variable(step_float, dtype=tf.float64)

        def log_prob_whitened(u):
            alm = u / mass_sqrt_var
            full_params = tf.concat([lncl_var[2:], alm], axis=0)
            if sample_phi:
                phi_packed = phi_whitener.unwhiten_tf(phi_state_var)
                return -psi_lensed(model, full_params, phi_packed)
            return -model.psi_tf(full_params)

        hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=log_prob_whitened,
            step_size=step_size_var,
            num_leapfrog_steps=n_lfs,
        )

        # bootstrap_results must be called fresh every sweep (the Gibbs
        # conditioning -- lncl_var/mass_sqrt_var, and phi_state_var/
        # phi_mass_sqrt_var when sample_phi -- changes between blocks, so a
        # cached kernel-results struct from a previous sweep would carry a
        # stale target_log_prob), so it can't just run once before the loop.
        # Combining it with one_step into a SINGLE @tf.function, traced once
        # and reused every sweep, avoids calling bootstrap_results eagerly.
        # Both psi_tf and (with sample_phi) psi_lensed's matrix-free-SHT/
        # lensing escape hatches use tf.py_function, which in eager mode
        # registers in TF's process-global eager py-function registry and is
        # never freed; direct RSS measurement (achievements.md) confirmed
        # this leaks ~1 MB per eager call at toy scale, and that a traced
        # graph -- built once, reused every call -- does not leak. This
        # combined, always-traced form replaces the previous split
        # (bootstrap_results eager every sweep; one_step only @tf.function
        # when not sample_phi), which existed because log_prob_phi_block's
        # cl_phiphi_full argument used to be a plain numpy array that would
        # have been silently baked into the trace as a constant -- see
        # cl_phiphi_full_var above, which is what makes tracing safe here
        # (and psi_lensed itself traces cleanly; verified directly, it was
        # never actually untraceable, just never re-tried after the
        # matrix-free SHT path added proper tf.py_function wrapping).
        @tf.function
        def hmc_bootstrap_and_step(state):
            pkr = hmc_kernel.bootstrap_results(state)
            return hmc_kernel.one_step(state, pkr)

        if sample_phi:
            def phi_log_prob_whitened(u):
                phi_packed = phi_whitener.unwhiten_tf(u)
                full_params = tf.concat([lncl_var[2:], state_var / mass_sqrt_var], axis=0)
                return log_prob_phi_block(model, full_params, phi_packed, cl_phiphi_full_var)

            if phi_sampler == 'mclmc':
                phi_mclmc_bootstrap_and_step = build_phi_mclmc_step(
                    phi_log_prob_whitened, phi_step_size_var, phi_mclmc_L, phi_n_lfs
                )
            elif phi_sampler == 'nuts':
                phi_nuts_kernel = tfp.mcmc.NoUTurnSampler(
                    target_log_prob_fn=phi_log_prob_whitened,
                    step_size=phi_step_size_var,
                    max_tree_depth=phi_nuts_max_tree_depth,
                    max_energy_diff=phi_nuts_max_energy_diff,
                )

                @tf.function
                def phi_nuts_bootstrap_and_step(state):
                    pkr = phi_nuts_kernel.bootstrap_results(state)
                    return phi_nuts_kernel.one_step(state, pkr)
            else:
                phi_hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=phi_log_prob_whitened,
                    step_size=phi_step_size_var,
                    num_leapfrog_steps=phi_n_lfs,
                )

                @tf.function
                def phi_hmc_bootstrap_and_step(state):
                    pkr = phi_hmc_kernel.bootstrap_results(state)
                    return phi_hmc_kernel.one_step(state, pkr)

    else:  # 'cg' or 'messenger'
        model._ensure_tf_tensors()  # idempotent, no-op if already built

        if sample_phi:  # 'cg' + Block 3 (gate-2 exactness experiment)
            cg_lncl_var = tf.Variable(
                tf.constant(np.concatenate([np.zeros(2), current_lncl]), dtype=tf.float64)
            )
            cg_alm_var = tf.Variable(tf.constant(current_alm_np, dtype=tf.float64))

            def phi_log_prob_whitened(u):
                phi_packed = phi_whitener.unwhiten_tf(u)
                full_params = tf.concat([cg_lncl_var[2:], cg_alm_var], axis=0)
                return log_prob_phi_block(model, full_params, phi_packed, cl_phiphi_full_var)

            if phi_sampler == 'mclmc':
                phi_mclmc_bootstrap_and_step = build_phi_mclmc_step(
                    phi_log_prob_whitened, phi_step_size_var, phi_mclmc_L, phi_n_lfs
                )
            elif phi_sampler == 'nuts':
                phi_nuts_kernel = tfp.mcmc.NoUTurnSampler(
                    target_log_prob_fn=phi_log_prob_whitened,
                    step_size=phi_step_size_var,
                    max_tree_depth=phi_nuts_max_tree_depth,
                    max_energy_diff=phi_nuts_max_energy_diff,
                )

                @tf.function
                def phi_nuts_bootstrap_and_step(state):
                    pkr = phi_nuts_kernel.bootstrap_results(state)
                    return phi_nuts_kernel.one_step(state, pkr)
            else:
                phi_hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
                    target_log_prob_fn=phi_log_prob_whitened,
                    step_size=phi_step_size_var,
                    num_leapfrog_steps=phi_n_lfs,
                )

                @tf.function
                def phi_hmc_bootstrap_and_step(state):
                    pkr = phi_hmc_kernel.bootstrap_results(state)
                    return phi_hmc_kernel.one_step(state, pkr)

    # Frozen curvature estimates for phi_mass_matrix='block' + sample_cl_phiphi
    # (Block 4): diag_fisher_per_L/block_hessian are expensive (gradient-eval)
    # one-time burn-in estimates of LIKELIHOOD curvature, independent of
    # cl_phiphi_full -- correct to freeze regardless of how many times Block 4
    # resamples the spectrum afterwards. Persisted here (not just local to the
    # warmup block below) so Step 3a can cheaply rebuild the Cholesky each
    # sweep from the frozen curvature + the NEW cl_phiphi_full, mirroring how
    # the 'prior'/'fisher' diagonal case already rebuilds every sweep via
    # build_phi_prior_mass_sqrt -- see the sample_cl_phiphi block-mode branch
    # below and its docstring note (ROADMAP.md, 2026-08-23).
    frozen_diag_fisher_per_L = np.zeros(lmax, dtype=np.float64)
    frozen_block_hessian = {}

    phi_rescale_accepts = []

    if sample_cl_phiphi:
        def _sync_phi_state_to_spectrum(cl_new, phi_np):
            """Push a new C_L^phiphi into the traced graph, rebuild the phi
            mass matrix from it, and re-whiten the phi state to match.

            Shared by Step 3a (Block 4's exact draw) and Step 3b (the ancillary
            rescaling move) so the two cannot drift apart -- both change the
            spectrum and both must leave the whitened state self-consistent
            with it.

            cl_phiphi_full_var is .assign()-ed rather than rebound because a
            plain Python rebind is invisible to the already-traced phi kernel,
            whereas a tf.Variable assignment is read live at call time.

            For phi_mass_matrix='block' the per-m Cholesky is rebuilt from the
            NEW spectrum but reuses the frozen diag_fisher_per_L/block_hessian
            burn-in estimates: those are LIKELIHOOD-curvature quantities,
            independent of cl_phiphi_full, so freezing them is correct however
            often the spectrum is resampled. Only the 1/cl_phiphi(L)
            prior-precision term tracks the new spectrum, and that costs a
            Cholesky per block with no new gradient evaluations.
            """
            cl_phiphi_full_var.assign(tf.constant(cl_new, dtype=tf.float64))
            if phi_whitener.mode == 'block':
                phi_whitener.update_block(build_phi_block_mass_chol(
                    lmax, cl_new, frozen_diag_fisher_per_L, frozen_block_hessian
                ))
            else:
                phi_whitener.update_diag(
                    build_phi_prior_mass_sqrt(lmax, cl_new)
                )
            phi_state_var.assign(
                tf.constant(phi_whitener.whiten_np(phi_np), dtype=tf.float64)
            )

    if phi_rescale_move:
        from .lensing import sample_phi_amplitude_rescale

        def _phi_neg_log_lik_np(phi_np):
            """psi_lensed at the CURRENT (alm, C_l) state, as a plain float.

            Deliberately excludes the phi Gaussian prior: the rescaling move's
            acceptance ratio handles that term analytically (it is exactly the
            piece the move leaves invariant), so including it here would
            double-count and bias the move.
            """
            full_params = tf.concat(
                [lncl_var[2:], state_var / mass_sqrt_var], axis=0
            )
            return float(psi_lensed(
                model, full_params, tf.constant(phi_np, dtype=tf.float64)
            ).numpy())

    recent = []
    resume_tag = "resuming, " if resuming else ""
    if alm_sampler == 'cg':
        sampler_extra = f", n_pcg_iter={n_pcg_iter}"
    elif alm_sampler == 'messenger':
        sampler_extra = f", n_messenger_iter={n_messenger_iter}"
    else:
        sampler_extra = f", n_lfs={n_lfs}"
    sampler_tag = f"alm_sampler={alm_sampler}{sampler_extra}"
    print(
        f"Starting Gibbs chain ({resume_tag}{burnin_remaining} burn-in + {n_samples_remaining} samples, "
        f"step_size={step_float:.3g}, {sampler_tag})"
    )

    for i in range(burnin_remaining + n_samples_remaining):
        is_burnin = i < burnin_remaining

        # --- Step 1: exact C_l | alm (always inverse-Gamma) ---
        if alm_sampler == 'hmc':
            alm_np = state_var.numpy() / mass_sqrt_np
        else:
            alm_np = current_alm_np

        new_lncl = model.sample_cl_given_alm(alm_np, rng)
        cl_full[2:] = np.exp(new_lncl)
        new_mass_sqrt = model.build_posterior_mass_sqrt(cl_full)
        mass_sqrt_np = new_mass_sqrt
        current_lncl = new_lncl

        # --- Step 2: alm | C_l ---
        if alm_sampler == 'hmc':
            state_var.assign(alm_np * new_mass_sqrt)
            mass_sqrt_var.assign(tf.constant(new_mass_sqrt, dtype=tf.float64))
            lncl_var.assign(
                tf.constant(np.concatenate([np.zeros(2), new_lncl]), dtype=tf.float64)
            )
            new_state, new_pkr = hmc_bootstrap_and_step(state_var)
            state_var.assign(new_state)

            accepted = bool(new_pkr.is_accepted.numpy())
            logp_val = float(-new_pkr.accepted_results.target_log_prob.numpy())

            if is_burnin:
                gamma = 1.0 / ((i + 1) ** 0.6)
                log_step = np.log(step_float) + gamma * (float(accepted) - target_accept)
                step_float = float(np.clip(np.exp(log_step), 1e-7, 2.0))
                step_size_var.assign(step_float)

        elif alm_sampler == 'cg':
            current_alm_np, _ = sample_alm_cg(model, new_lncl, rng, n_pcg_iter)
            accepted = True
            full_p = tf.constant(
                np.concatenate([new_lncl, current_alm_np]), dtype=tf.float64
            )
            logp_val = float(-model._psi_tf_raw(full_p))
            if sample_phi:
                cg_lncl_var.assign(
                    tf.constant(np.concatenate([np.zeros(2), new_lncl]), dtype=tf.float64)
                )
                cg_alm_var.assign(tf.constant(current_alm_np, dtype=tf.float64))

        else:  # 'messenger'
            current_alm_np = sample_alm_messenger(
                model, new_lncl, rng, n_messenger_iter, s0=current_alm_np,
                use_block_correction=messenger_use_block_correction,
                m_group_size=messenger_m_group_size,
            )
            accepted = True
            full_p = tf.constant(
                np.concatenate([new_lncl, current_alm_np]), dtype=tf.float64
            )
            logp_val = float(-model._psi_tf_raw(full_p))

        recent.append(accepted)

        # --- Fisher-informed / block phi mass matrix (opt-in, one-time recompute) ---
        # Waits until phi_fisher_warmup_iter so the chain state has moved
        # past the literal warm-start point before measuring curvature there.
        if sample_phi and not phi_fisher_done and is_burnin and i >= min(
            phi_fisher_warmup_iter, max(burnin_remaining - 1, 0)
        ):
            alm_now = (
                state_var.numpy() / mass_sqrt_np if alm_sampler == 'hmc' else current_alm_np
            )
            full_params_now = np.concatenate([current_lncl, alm_now])
            phi_now = phi_whitener.unwhiten_np(phi_state_var.numpy())
            full_params_tf = tf.constant(full_params_now, dtype=tf.float64)
            phi_now_tf = tf.constant(phi_now, dtype=tf.float64)
            diag_fisher_per_L = estimate_phi_diag_fisher(
                model, full_params_tf, phi_now_tf, lmax,
                n_probes=phi_fisher_n_probes, rng=rng,
            )
            if phi_mass_matrix == 'block':
                block_hessian = estimate_phi_block_hessian(
                    model, full_params_tf, phi_now_tf, lmax,
                    n_probes=phi_block_n_probes, rng=rng,
                )
                new_block_chol = build_phi_block_mass_chol(
                    lmax, cl_phiphi_full, diag_fisher_per_L, block_hessian
                )
                phi_whitener.update_block(new_block_chol)
                frozen_diag_fisher_per_L = diag_fisher_per_L
                frozen_block_hessian = block_hessian
            else:
                phi_mass_sqrt_np = build_phi_posterior_mass_sqrt(lmax, cl_phiphi_full, diag_fisher_per_L)
                phi_whitener.update_diag(phi_mass_sqrt_np)
            phi_state_var.assign(tf.constant(phi_whitener.whiten_np(phi_now), dtype=tf.float64))
            phi_fisher_done = True

        # --- Step 3a: C_L^phiphi | phi (opt-in, optional Block 4) ---
        # Same position in the sweep as Step 1 (C_l | alm): draws the new
        # spectrum from the phi state as of the end of the previous sweep,
        # then rebuilds the phi HMC mass matrix and rescales the whitened
        # phi state to match -- mirroring exactly how new_lncl/mass_sqrt_np/
        # state_var are rebuilt for the alm block in Step 1/2 above.
        # cl_phiphi_full is rebound here (not mutated in place) for the
        # numpy-side bookkeeping (cl_phiphi_out/checkpointing read it
        # directly); cl_phiphi_full_var is separately .assign()-ed to the
        # same new value so the *traced* phi_hmc_bootstrap_and_step sees the
        # update too (a plain Python rebind is invisible to an already-traced
        # graph, but a tf.Variable .assign() is read live at call time).
        if sample_cl_phiphi:
            phi_now_np = phi_whitener.unwhiten_np(phi_state_var.numpy())
            new_lncl_phiphi = sample_cl_phiphi_given_phi(
                phi_now_np, lmax, rng,
                prior_nu=cl_phiphi_prior_nu,
                cl_phiphi_fid=cl_phiphi_prior_fid,
            )
            cl_phiphi_full = cl_phiphi_full.copy()
            cl_phiphi_full[2:lmax] = np.exp(new_lncl_phiphi)
            _sync_phi_state_to_spectrum(cl_phiphi_full, phi_now_np)

            # --- Step 3b: ancillary (non-centred) joint rescaling move ---
            # Steps 3a and 3 together are a CENTRED parameterisation of an
            # amplitude-and-variance model and they funnel: job 11836793
            # measured worst lag-1 autocorrelation 0.945 with Block 4 on
            # against 0.557 with it off, same lmax=64 configuration. This is
            # the ancillary half of an ancillary-sufficient interweaving
            # strategy -- it holds xi = phi/sqrt(C) fixed and slides (phi, C)
            # along the funnel axis that 3a and 3 mix slowly across. Placed
            # immediately after 3a so the sweep reads
            # sufficient-update -> ancillary-update, the standard interweaving
            # order (see lensing.py::sample_phi_amplitude_rescale).
            if phi_rescale_move:
                move = sample_phi_amplitude_rescale(
                    phi_now_np,
                    cl_phiphi_full,
                    lmax,
                    neg_log_lik_fn=_phi_neg_log_lik_np,
                    rng=rng,
                    proposal_scale=phi_rescale_proposal_scale,
                )
                phi_rescale_accepts.append(move.accepted)
                if move.accepted:
                    cl_phiphi_full = move.cl_phiphi
                    # Recorded spectrum must be the POST-move one; leaving the
                    # pre-move Block 4 draw here would silently log a spectrum
                    # the chain never actually held. Add 2*log_alpha rather
                    # than recomputing log(cl_phiphi_full[2:lmax]): the move
                    # scales C_L by exactly alpha^2, so this is exact, whereas
                    # a log(exp(x)) round-trip perturbs the last bits.
                    new_lncl_phiphi = new_lncl_phiphi + 2.0 * move.log_alpha
                    phi_now_np = move.phi
                    _sync_phi_state_to_spectrum(cl_phiphi_full, phi_now_np)

        # --- Step 3: phi | alm, C_l, d (opt-in, Phase 2 Block 3) ---
        if sample_phi:
            if phi_sampler == 'mclmc':
                new_phi_state, new_phi_velocity, mclmc_diag = phi_mclmc_bootstrap_and_step(
                    phi_state_var, phi_velocity_var
                )
                phi_state_var.assign(new_phi_state)
                phi_velocity_var.assign(new_phi_velocity)
                # MCLMC is unadjusted (no M-H accept/reject) -- repurpose the
                # existing accept-rate bookkeeping/checkpointing plumbing to
                # track "trajectory did not diverge" instead, so downstream
                # code (mean accept-rate logging, phi_accepts_out) needs no
                # further changes. Step size is intentionally NOT adapted
                # here (fixed at phi_hmc_step_size) -- see phi_sampler
                # docstring above: this is a bounded spike, not a new
                # adaptive scheme.
                phi_accepted = not bool(mclmc_diag.diverged.numpy())
            elif phi_sampler == 'nuts':
                new_phi_state, new_phi_pkr = phi_nuts_bootstrap_and_step(phi_state_var)
                phi_state_var.assign(new_phi_state)
                phi_accepted = bool(new_phi_pkr.is_accepted.numpy())

                if is_burnin:
                    # Same external Robbins-Monro step-size adaptation as the
                    # HMC branch below -- see run_gibbs_chain's docstring for
                    # why TFP's own DualAveragingStepSizeAdaptation wrapper
                    # isn't used here (its internal adaptation state can't
                    # survive this loop's per-sweep bootstrap_results calls).
                    gamma = 1.0 / ((i + 1) ** 0.6)
                    log_step = np.log(phi_step_float) + gamma * (float(phi_accepted) - phi_target_accept)
                    phi_step_float = float(np.clip(np.exp(log_step), 1e-7, 2.0))
                    phi_step_size_var.assign(phi_step_float)
            else:
                new_phi_state, new_phi_pkr = phi_hmc_bootstrap_and_step(phi_state_var)
                phi_state_var.assign(new_phi_state)
                phi_accepted = bool(new_phi_pkr.is_accepted.numpy())

                if is_burnin:
                    gamma = 1.0 / ((i + 1) ** 0.6)
                    log_step = np.log(phi_step_float) + gamma * (float(phi_accepted) - phi_target_accept)
                    phi_step_float = float(np.clip(np.exp(log_step), 1e-7, 2.0))
                    phi_step_size_var.assign(phi_step_float)

        if i % 200 == 0:
            rate_str = f"{np.mean(recent[-100:]):.2f}" if len(recent) >= 100 else "n/a"
            phase = "burn-in" if is_burnin else "sampling"
            phi_tag = f"  phi_step={phi_step_float:.3e}" if sample_phi else ""
            print(f"  iter {i:5d} ({phase}): step={step_float:.3e}  accept={rate_str}  total_samples={len(samples_out)}{phi_tag}")

        if not is_burnin:
            alm_sample = state_var.numpy() / mass_sqrt_np if alm_sampler == 'hmc' else current_alm_np.copy()
            samples_out.append(np.concatenate([current_lncl, alm_sample]))
            logp_out.append(logp_val)
            accepts_out.append(accepted)
            if sample_phi:
                phi_sample = phi_whitener.unwhiten_np(phi_state_var.numpy())
                phi_samples_out.append(phi_sample)
                phi_accepts_out.append(phi_accepted)
                if sample_cl_phiphi:
                    cl_phiphi_out.append(new_lncl_phiphi.copy())

            if checkpoint_path and len(samples_out) % checkpoint_every == 0:
                ckpt_kwargs = {
                    "packing_version": np.int32(PACKING_VERSION),
                    "samples": np.array(samples_out, dtype=np.float64),
                    "logp": np.array(logp_out, dtype=np.float64),
                    "accepts": np.array(accepts_out, dtype=bool),
                    "alm_state": alm_sample.astype(np.float64),
                    "lncl_state": current_lncl.astype(np.float64),
                    "mass_sqrt": mass_sqrt_np.astype(np.float64),
                    "step_size": np.float64(step_float),
                }
                if sample_phi:
                    ckpt_kwargs.update(
                        phi_samples=np.array(phi_samples_out, dtype=np.float64),
                        phi_accepts=np.array(phi_accepts_out, dtype=bool),
                        phi_state=phi_sample.astype(np.float64),
                        phi_step_size=np.float64(phi_step_float),
                    )
                    if phi_sampler == 'mclmc':
                        ckpt_kwargs.update(phi_velocity=phi_velocity_var.numpy().astype(np.float64))
                if sample_cl_phiphi:
                    ckpt_kwargs.update(
                        cl_phiphi_samples=np.array(cl_phiphi_out, dtype=np.float64),
                        cl_phiphi_state=cl_phiphi_full.astype(np.float64),
                    )
                np.savez(checkpoint_path, **ckpt_kwargs)

    print(
        f"Gibbs chain done. Final step_size={step_float:.4g}, "
        f"mean accept={float(np.mean(accepts_out)):.3f}"
    )
    if sample_phi:
        print(f"  phi block: final step_size={phi_step_float:.4g}, mean accept={float(np.mean(phi_accepts_out)):.3f}")
        if sample_cl_phiphi:
            return (
                np.array(samples_out, dtype=np.float64),
                np.array(phi_samples_out, dtype=np.float64),
                np.array(logp_out, dtype=np.float64),
                np.array(accepts_out, dtype=bool),
                step_float,
                np.array(cl_phiphi_out, dtype=np.float64),
            )
        return (
            np.array(samples_out, dtype=np.float64),
            np.array(phi_samples_out, dtype=np.float64),
            np.array(logp_out, dtype=np.float64),
            np.array(accepts_out, dtype=bool),
            step_float,
        )
    return (
        np.array(samples_out, dtype=np.float64),
        np.array(logp_out, dtype=np.float64),
        np.array(accepts_out, dtype=bool),
        step_float,
    )


def run_chain_nut(
    modelparams,
    initial_state,
    _step_size,
    num_results=1000,
    num_burnin_steps=0,
    mtd=10,
    med=1000,
    u_lfs=1,
    pi=10,
):
    """Run No-U-Turn Sampler with dual-averaging step size adaptation.

    Returns the desired walks through parameter space.
    """
    if tfp is None:
        raise ImportError("tensorflow_probability is required for run_chain_nut")
    if tf is None:
        raise ImportError("tensorflow is required for run_chain_nut")

    # psi_tf is the negative log-posterior; negate to give TFP the log-posterior.
    def log_prob_fn(params):
        return -modelparams.psi_tf(params)
    nut_kernel = tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=log_prob_fn,
        step_size=_step_size,
        max_tree_depth=mtd,
        max_energy_diff=med,
        unrolled_leapfrog_steps=u_lfs,
        parallel_iterations=pi,
    )
    adaptive_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        nut_kernel,
        num_adaptation_steps=max(1, num_burnin_steps),
        target_accept_prob=0.8,
    )

    def trace_fn(current_state, kernel_results):
        inner = getattr(kernel_results, "inner_results", kernel_results)
        if hasattr(inner, "accepted_results"):
            target_log_prob = inner.accepted_results.target_log_prob
        else:
            target_log_prob = getattr(inner, "target_log_prob", tf.constant(0.0, dtype=tf.float64))
        is_accepted = getattr(inner, "is_accepted", tf.constant(False, dtype=tf.bool))
        step_size = getattr(kernel_results, "new_step_size", tf.constant(0.0, dtype=tf.float64))
        return (
            target_log_prob,
            is_accepted,
            step_size,
        )

    samples, trace_results = tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=initial_state,
        kernel=adaptive_kernel,
        trace_fn=trace_fn,
    )
    return samples, WrapperResults(*trace_results)

