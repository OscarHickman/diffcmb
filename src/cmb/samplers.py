try:
    import tensorflow_probability as tfp
except Exception:
    tfp = None


def run_chain_hmc(
    modelparams,
    initial_state,
    _step_size=0.01,
    num_results=1000,
    num_burnin_steps=0,
    _n_lfs=2,
):
    """Run Hamiltonian Monte Carlo sampler with dual-averaging step size adaptation.

    Returns the desired walks through parameter space.
    """
    if tfp is None:
        raise ImportError("tensorflow_probability is required for run_chain_hmc")
    # psi_tf is the negative log-posterior; negate to give TFP the log-posterior.
    log_prob_fn = lambda params: -modelparams.psi_tf(params)
    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=log_prob_fn,
        step_size=_step_size,
        num_leapfrog_steps=_n_lfs,
    )
    adaptive_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        hmc_kernel,
        num_adaptation_steps=max(1, num_burnin_steps),
        target_accept_prob=0.75,
    )
    return tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=initial_state,
        kernel=adaptive_kernel,
        trace_fn=lambda current_state, kernel_results: kernel_results,
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
    # psi_tf is the negative log-posterior; negate to give TFP the log-posterior.
    log_prob_fn = lambda params: -modelparams.psi_tf(params)
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
    return tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=initial_state,
        kernel=adaptive_kernel,
        trace_fn=lambda current_state, kernel_results: kernel_results,
    )
