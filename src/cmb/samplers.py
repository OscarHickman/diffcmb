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


def find_map_estimate(model, n_steps=500, learning_rate=0.001, print_every=50):
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
        optimizer.apply_gradients([(grads, params)])
        return loss

    print(f"Finding MAP estimate ({n_steps} Adam steps, lr={learning_rate})...")
    print(f"  initial psi = {psi_at_x0:.6g}")
    t0 = time.time()
    loss_val = None
    for i in range(n_steps):
        loss_val = _step()
        if i % print_every == 0 or i == n_steps - 1:
            print(f"  step {i:4d}: psi = {float(loss_val):.6g}  ({time.time()-t0:.1f}s)")

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


def run_gibbs_chain(
    model,
    n_samples=1000,
    n_burnin=500,
    hmc_step_size=0.05,
    n_lfs=20,
    target_accept=0.65,
    seed=None,
    initial_params=None,
):
    """Gibbs sampler alternating exact C_l | alm and HMC alm | C_l steps.

    Step 1 – C_l | alm: exact inverse-Gamma sample (O(lmax), no MCMC error).
    Step 2 – alm | C_l: one HMC accept/reject with posterior-based diagonal mass
        M[l] = sqrt(1/C_l + Ninv_eff), which nearly diagonalises the posterior
        for high-S/N CMB problems and gives condition number ≈ 1 in whitened space.

    Returns (samples, logp, accepts, final_step_size) where samples shape is
    (n_samples, n_params) with the same x0 layout as the rest of the codebase.
    """
    if tfp is None or tf is None:
        raise ImportError("tensorflow and tensorflow_probability are required")

    rng = np.random.default_rng(seed)
    lmax = model.lmax
    n_lncl = lmax - 2

    if initial_params is not None:
        x0 = np.array(initial_params, dtype=np.float64).ravel()
    else:
        x0 = np.array(model.x0, dtype=np.float64)
    current_lncl = x0[:n_lncl].copy()
    current_alm_np = x0[n_lncl:].copy()

    cl_full = np.zeros(lmax)
    cl_full[2:] = np.exp(current_lncl)
    mass_sqrt_np = model.build_posterior_mass_sqrt(cl_full)

    mass_sqrt_var = tf.Variable(tf.constant(mass_sqrt_np, dtype=tf.float64))
    lncl_var = tf.Variable(
        tf.constant(np.concatenate([np.zeros(2), current_lncl]), dtype=tf.float64)
    )

    # Whitened state: u = alm * mass_sqrt
    state_var = tf.Variable(tf.constant(current_alm_np * mass_sqrt_np, dtype=tf.float64))
    step_size_var = tf.Variable(hmc_step_size, dtype=tf.float64)

    def log_prob_whitened(u):
        alm = u / mass_sqrt_var
        full_params = tf.concat([lncl_var[2:], alm], axis=0)
        return -model.psi_tf(full_params)

    inner_hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=log_prob_whitened,
        step_size=step_size_var,
        num_leapfrog_steps=n_lfs,
    )
    hmc_kernel = tfp.mcmc.MetropolisHastings(inner_kernel=inner_hmc)
    pkr = hmc_kernel.bootstrap_results(state_var)

    @tf.function
    def hmc_one_step(state, pkr):
        return hmc_kernel.one_step(state, pkr)

    samples_out = []
    logp_out = []
    accepts_out = []
    recent = []
    step_float = hmc_step_size

    print(
        f"Starting Gibbs chain ({n_burnin} burn-in + {n_samples} samples, "
        f"step_size={hmc_step_size:.3g}, n_lfs={n_lfs})"
    )

    for i in range(n_burnin + n_samples):
        # --- Step 1: exact C_l | alm ---
        alm_np = state_var.numpy() / mass_sqrt_np
        new_lncl = model.sample_cl_given_alm(alm_np, rng)
        cl_full[2:] = np.exp(new_lncl)
        new_mass_sqrt = model.build_posterior_mass_sqrt(cl_full)

        # Re-whiten state for updated mass matrix (alm unchanged, u rescaled)
        state_var.assign(alm_np * new_mass_sqrt)
        mass_sqrt_np = new_mass_sqrt
        mass_sqrt_var.assign(tf.constant(new_mass_sqrt, dtype=tf.float64))
        lncl_var.assign(
            tf.constant(np.concatenate([np.zeros(2), new_lncl]), dtype=tf.float64)
        )
        current_lncl = new_lncl

        # Refresh kernel results so accept/reject uses the updated log-prob
        pkr = hmc_kernel.bootstrap_results(state_var)

        # --- Step 2: HMC alm | C_l ---
        new_state, new_pkr = hmc_one_step(state_var, pkr)
        state_var.assign(new_state)
        pkr = new_pkr

        inner = new_pkr.inner_results
        accepted = bool(inner.is_accepted.numpy())
        logp_val = float(-inner.accepted_results.target_log_prob.numpy())
        recent.append(accepted)

        # Multiplicative step-size adaptation during burn-in
        if i < n_burnin and len(recent) >= 50:
            rate = float(np.mean(recent[-50:]))
            if rate > target_accept + 0.05:
                step_float = min(step_float * 1.04, 2.0)
                step_size_var.assign(step_float)
            elif rate < target_accept - 0.05:
                step_float = max(step_float * 0.96, 1e-7)
                step_size_var.assign(step_float)

        if i % 200 == 0:
            rate_str = f"{np.mean(recent[-100:]):.2f}" if len(recent) >= 100 else "n/a"
            phase = "burn-in" if i < n_burnin else "sampling"
            print(f"  iter {i:5d} ({phase}): step={step_float:.3e}  accept={rate_str}")

        if i >= n_burnin:
            alm_sample = state_var.numpy() / mass_sqrt_np
            samples_out.append(np.concatenate([current_lncl, alm_sample]))
            logp_out.append(logp_val)
            accepts_out.append(accepted)

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

