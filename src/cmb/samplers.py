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
    """Run Hamiltonian Monte Carlo sampler.

    Returns the desired walks through parameter space for a fixed step size.
    """
    if tfp is None:
        raise ImportError("tensorflow_probability is required for run_chain_hmc")
    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=modelparams.psi_tf,
        step_size=_step_size,
        num_leapfrog_steps=_n_lfs,
    )
    return tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=initial_state,
        kernel=hmc_kernel,
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
    """Run No-U-Turn Sampler chain.

    Returns the desired walks through parameter space for a fixed step size.
    """
    if tfp is None:
        raise ImportError("tensorflow_probability is required for run_chain_nut")
    nut_kernel = tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=modelparams.psi_tf,
        step_size=_step_size,
        max_tree_depth=mtd,
        max_energy_diff=med,
        unrolled_leapfrog_steps=u_lfs,
        parallel_iterations=pi,
    )
    return tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=initial_state,
        kernel=nut_kernel,
        trace_fn=lambda current_state, kernel_results: kernel_results,
    )
