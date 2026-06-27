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
    from .alm_utils import splittosingularalm_tf
    from .model import matvec_on_device
except Exception:
    splittosingularalm_tf = None
    matvec_on_device = None


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
    """Per-alm 1/C_l diagonal matching the real+imag alm parameter layout."""
    inv_cl = np.empty(n_real + n_imag, dtype=np.float64)
    idx = 0
    for L in range(2, lmax):
        cl = max(float(cl_full[L]), 1e-30)
        for _ in range(L + 1):
            inv_cl[idx] = 1.0 / cl
            idx += 1
    for L in range(2, lmax):
        cl = max(float(cl_full[L]), 1e-30)
        for m in range(L + 1):
            if m >= 2:
                inv_cl[idx] = 1.0 / cl
                idx += 1
    return inv_cl


def sample_alm_cg(model, lncl_np, rng, n_pcg_iter=50, tol=1e-6):
    """Exact Gaussian draw from p(alm | C_l, d) via preconditioned CG (Wandelt+2004).

    Solves A x = b_sample where:
        A        = diag(1/C_l per alm)  +  J^T N^{-1} J
        b_sample = J^T N^{-1} d  +  C_l^{-1/2} ω₁  +  J^T N^{-1/2} ω₂

    The matvec A p is obtained via ∇_alm ψ(p) − ∇_alm ψ(0) using TF autodiff.
    Diagonal preconditioner P = 1/C_l + Ninv_eff  (≈ mass_sqrt²) nearly diagonalises
    A for high-S/N CMB; convergence expected in O(10–50) PCG iterations.

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
    n_real = lmax * (lmax + 1) // 2 - 3
    n_imag = (lmax - 2) * (lmax - 1) // 2
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
        _part_sizes = [int(sph_p.shape[0]) for sph_p in model.sph_parts]
        _n_real_cap = n_real
        _lmax_cap = lmax

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
        p = -z + beta * p
        rz = rz_new

    if residual_norms[-1] > tol:
        print(f"    PCG: |r|={residual_norms[-1]:.3e} after {len(residual_norms)-1} iters (tol={tol:.0e})")

    return x, residual_norms


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
):
    """Gibbs sampler alternating exact C_l | alm (inverse-Gamma) + alm | C_l steps.

    Step 1 – C_l | alm: exact inverse-Gamma sample (O(lmax), no MCMC error).
    Step 2 – alm | C_l: controlled by `alm_sampler`:
        'hmc'  – one HMC accept/reject with diagonal mass M = sqrt(1/C_l + Ninv_eff).
                 Requires burn-in and step-size tuning; IAT grows with multipole.
        'cg'   – exact Gaussian draw via preconditioned CG (Wandelt+2004).
                 No accept/reject; IAT = 1 at all multipoles by construction.
                 n_burnin is still respected but step-size arguments are ignored.

    Returns (samples, logp, accepts, final_step_size) where samples shape is
    (n_samples, n_params) with the same x0 layout as the rest of the codebase.

    If checkpoint_path is provided, saves state every checkpoint_every collected samples
    and resumes from that file if it already exists (skipping burnin on resume).
    """
    if alm_sampler not in ('hmc', 'cg'):
        raise ValueError(f"alm_sampler must be 'hmc' or 'cg', got {alm_sampler!r}")
    if alm_sampler == 'hmc' and (tf is None or tfp is None):
        raise ImportError("tensorflow and tensorflow_probability are required")
    if alm_sampler == 'cg' and tf is None:
        raise ImportError("tensorflow is required for CG sampler")

    rng = np.random.default_rng(seed)
    lmax = model.lmax
    n_lncl = lmax - 2

    # --- Resume from checkpoint or initialise fresh ---
    samples_out = []
    logp_out = []
    accepts_out = []
    step_float = hmc_step_size
    resuming = False

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = np.load(checkpoint_path, allow_pickle=True)
        samples_out = list(ckpt["samples"])
        logp_out = list(ckpt["logp"])
        accepts_out = list(ckpt["accepts"].tolist())
        current_alm_np = ckpt["alm_state"].copy()
        current_lncl = ckpt["lncl_state"].copy()
        mass_sqrt_np = ckpt["mass_sqrt"].copy()
        step_float = float(ckpt["step_size"])
        resuming = True
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

    n_collected = len(samples_out)
    n_samples_remaining = n_samples - n_collected
    burnin_remaining = 0 if resuming else n_burnin

    if n_samples_remaining <= 0:
        print("All samples already collected from checkpoint.")
        return (
            np.array(samples_out, dtype=np.float64),
            np.array(logp_out, dtype=np.float64),
            np.array(accepts_out, dtype=bool),
            step_float,
        )

    cl_full = np.zeros(lmax)
    cl_full[2:] = np.exp(current_lncl)

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
            return -model.psi_tf(full_params)

        hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=log_prob_whitened,
            step_size=step_size_var,
            num_leapfrog_steps=n_lfs,
        )
        pkr = hmc_kernel.bootstrap_results(state_var)

        @tf.function
        def hmc_one_step(state, pkr):
            return hmc_kernel.one_step(state, pkr)

    else:  # 'cg'
        if model.sph1 is None:
            model._ensure_tf_tensors()

    recent = []
    resume_tag = "resuming, " if resuming else ""
    sampler_tag = f"alm_sampler={alm_sampler}" + (f", n_pcg_iter={n_pcg_iter}" if alm_sampler == 'cg' else f", n_lfs={n_lfs}")
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
            pkr = hmc_kernel.bootstrap_results(state_var)

            new_state, new_pkr = hmc_one_step(state_var, pkr)
            state_var.assign(new_state)
            pkr = new_pkr

            accepted = bool(new_pkr.is_accepted.numpy())
            logp_val = float(-new_pkr.accepted_results.target_log_prob.numpy())

            if is_burnin:
                gamma = 1.0 / ((i + 1) ** 0.6)
                log_step = np.log(step_float) + gamma * (float(accepted) - target_accept)
                step_float = float(np.clip(np.exp(log_step), 1e-7, 2.0))
                step_size_var.assign(step_float)

        else:  # 'cg'
            current_alm_np, _ = sample_alm_cg(model, new_lncl, rng, n_pcg_iter)
            accepted = True
            full_p = tf.constant(
                np.concatenate([new_lncl, current_alm_np]), dtype=tf.float64
            )
            logp_val = float(-model._psi_tf_raw(full_p))

        recent.append(accepted)

        if i % 200 == 0:
            rate_str = f"{np.mean(recent[-100:]):.2f}" if len(recent) >= 100 else "n/a"
            phase = "burn-in" if is_burnin else "sampling"
            print(f"  iter {i:5d} ({phase}): step={step_float:.3e}  accept={rate_str}  total_samples={len(samples_out)}")

        if not is_burnin:
            alm_sample = state_var.numpy() / mass_sqrt_np if alm_sampler == 'hmc' else current_alm_np.copy()
            samples_out.append(np.concatenate([current_lncl, alm_sample]))
            logp_out.append(logp_val)
            accepts_out.append(accepted)

            if checkpoint_path and len(samples_out) % checkpoint_every == 0:
                np.savez(
                    checkpoint_path,
                    samples=np.array(samples_out, dtype=np.float64),
                    logp=np.array(logp_out, dtype=np.float64),
                    accepts=np.array(accepts_out, dtype=bool),
                    alm_state=alm_sample.astype(np.float64),
                    lncl_state=current_lncl.astype(np.float64),
                    mass_sqrt=mass_sqrt_np.astype(np.float64),
                    step_size=np.float64(step_float),
                )

    print(
        f"Gibbs chain done. Final step_size={step_float:.4g}, "
        f"mean accept={float(np.mean(accepts_out)):.3f}"
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

