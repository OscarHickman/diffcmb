import argparse
import os
import sys
import time

import numpy as np

# Ensure diffcmb/ source dir is in path so 'from diffcmb import ...' resolves
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "diffcmb")))

try:
    import tensorflow as tf
    # Configure GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU memory growth configuration error: {e}")

    from diffcmb import (
        CosmologyAdvancedSampling,
        find_map_estimate,
        run_chain_hmc,
        run_chain_nut,
        run_gibbs_chain,
    )
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run MCMC chain for CMB sampling.")
    parser.add_argument("--sampler", type=str, choices=["nuts", "hmc", "gibbs"], default="nuts")
    parser.add_argument("--lmax", type=int, default=200)
    parser.add_argument("--nside", type=int, default=128)
    parser.add_argument("--noise_sig", type=float, default=1.0)
    parser.add_argument("--data_mode", type=str, choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--n_burnin", type=int, default=500)
    parser.add_argument("--step_size", type=float, default=0.01)
    parser.add_argument("--n_lfs", type=int, default=10, help="Number of leapfrog steps (HMC only)")
    parser.add_argument("--chain_id", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--parameterization", type=str, choices=["centered", "non-centered"], default="centered",
                        help="Sampling strategy: 'centered' (standard) or 'non-centered' (reparameterized for speed)")
    parser.add_argument("--no_mass_matrix", action="store_true",
                        help="Disable diagonal mass matrix preconditioning (use identity mass)")
    parser.add_argument("--map_steps", type=int, default=0,
                        help="Adam steps for MAP initialisation before MCMC (0 = disabled)")
    parser.add_argument("--map_lr", type=float, default=0.005,
                        help="Learning rate for MAP optimization")
    parser.add_argument("--data_seed", type=int, default=42,
                        help="RNG seed for synthetic data generation (fixed so all chains share the same dataset)")
    parser.add_argument("--double_precision", action="store_true",
                        help="Use double precision (complex128/float64) for matrix operations to prevent gradient noise")
    parser.add_argument("--alm_sampler", type=str, choices=["hmc", "cg", "messenger"], default="hmc",
                        help="alm | C_l sampler: 'hmc' (default), 'cg' (exact Gaussian via PCG), "
                             "or 'messenger' (exact-in-the-limit Gaussian via messenger fields, "
                             "ROADMAP.md Phase 0c -- converges on a masked sky where 'cg' does not)")
    parser.add_argument("--n_pcg_iter", type=int, default=50,
                        help="Maximum PCG iterations per CG step (ignored for HMC/messenger)")
    parser.add_argument("--n_messenger_iter", type=int, default=100,
                        help="Messenger-field inner Gibbs iterations per alm|C_l draw (messenger only)")
    parser.add_argument("--messenger_use_block_correction", action="store_true",
                        help="Use the block-diagonal-by-m A^T A correction (Phase 0c Step 5/6) "
                             "instead of the plain diagonal approximation, which is biased on a "
                             "masked sky (messenger only)")
    parser.add_argument("--messenger_m_group_size", type=int, default=1,
                        help="Number of consecutive m values per block for the block correction "
                             "(messenger only, ignored unless --messenger_use_block_correction)")
    parser.add_argument("--use_matrixfree_sht", action="store_true",
                        help="Use the matrix-free ducc0 SHT instead of the dense sph matrix "
                             "(required for --alm_sampler messenger; also usable with cg/hmc)")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    checkpoint_path = os.path.join(args.output_dir, f"checkpoint_chain_{args.chain_id}.npz")
    map_cache_path = os.path.join(args.output_dir, f"map_init_chain_{args.chain_id}.npy")

    print(f"=== Chain {args.chain_id} Starting ===")
    print(f"Sampler: {args.sampler.upper()}")
    print(f"Data: {args.data_mode}  (data_seed={args.data_seed})")
    print(f"Parameterization: {args.parameterization}")
    print(f"LMAX: {args.lmax}, NSIDE: {args.nside}, Noise: {args.noise_sig}")
    print(f"Samples: {args.n_samples}, Burn-in: {args.n_burnin}, Step Size: {args.step_size}")
    print(f"Precision: {'double' if args.double_precision else 'single'}")
    if args.alm_sampler == 'cg':
        alm_sampler_desc = f" (n_pcg_iter={args.n_pcg_iter})"
    elif args.alm_sampler == 'messenger':
        alm_sampler_desc = (
            f" (n_messenger_iter={args.n_messenger_iter}, "
            f"use_block_correction={args.messenger_use_block_correction}, "
            f"m_group_size={args.messenger_m_group_size})"
        )
    else:
        alm_sampler_desc = ""
    print(f"alm sampler: {args.alm_sampler}{alm_sampler_desc}")

    # Fix the data-generation RNG so all chains sample the same posterior.
    if args.data_mode == "synthetic":
        np.random.seed(args.data_seed)

    if args.alm_sampler == 'messenger' and not args.use_matrixfree_sht:
        print("--alm_sampler messenger requires --use_matrixfree_sht; enabling it.")
        args.use_matrixfree_sht = True

    print("Constructing model...")
    t0 = time.time()
    dtype = tf.complex128 if args.double_precision else tf.complex64
    model = CosmologyAdvancedSampling(
        _lmax=args.lmax,
        _NSIDE=args.nside,
        _noisesig=args.noise_sig,
        data_mode=args.data_mode,
        data_dir=args.data_dir,
        parameterization=args.parameterization,
        dtype=dtype,
        use_matrixfree_sht=args.use_matrixfree_sht,
    )
    print(f"Model init took {time.time()-t0:.1f}s")

    print("Pre-loading spherical harmonic matrix...")
    t1 = time.time()
    model._ensure_tf_tensors()
    print(f"Tensor loading took {time.time()-t1:.1f}s")

    initial_state = model.prior_parameters_tf()

    if os.path.exists(checkpoint_path):
        print("Gibbs checkpoint found — skipping MAP init.")
        initial_state = None
    elif os.path.exists(map_cache_path):
        print("MAP cache found — loading and skipping MAP init.")
        initial_state = np.load(map_cache_path)
    elif args.map_steps > 0:
        t_map = time.time()
        initial_state = find_map_estimate(model, n_steps=args.map_steps, learning_rate=args.map_lr)
        print(f"MAP initialisation took {time.time()-t_map:.1f}s")
        np.save(map_cache_path, initial_state)

    mass_sqrt_diag = None
    if args.sampler == "hmc" and not args.no_mass_matrix:
        print("Building diagonal mass matrix...")
        mass_np = model.build_mass_sqrt_diag()
        mass_sqrt_diag = tf.constant(mass_np, dtype=tf.float64)
        print(f"  mass_sqrt range: [{mass_np.min():.4f}, {mass_np.max():.4f}]  "
              f"median={float(np.median(mass_np)):.4f}")

    print(f"Starting {args.sampler.upper()} sampling for chain {args.chain_id}...")
    t_chain = time.time()

    if args.sampler == "gibbs":
        samps_np, logp_np, accepts_np, final_step = run_gibbs_chain(
            model,
            n_samples=args.n_samples,
            n_burnin=args.n_burnin,
            hmc_step_size=args.step_size,
            n_lfs=args.n_lfs,
            seed=args.chain_id,
            initial_params=initial_state,
            checkpoint_path=checkpoint_path,
            checkpoint_every=100,
            alm_sampler=args.alm_sampler,
            n_pcg_iter=args.n_pcg_iter,
            n_messenger_iter=args.n_messenger_iter,
            messenger_use_block_correction=args.messenger_use_block_correction,
            messenger_m_group_size=args.messenger_m_group_size,
        )
        accept_rate = float(accepts_np.mean())
        print(f"Adapted step size: {final_step:.6g}")
    else:
        if args.sampler == "nuts":
            samples, results = run_chain_nut(
                model,
                initial_state,
                args.step_size,
                num_results=args.n_samples,
                num_burnin_steps=args.n_burnin,
            )
        else:
            samples, results = run_chain_hmc(
                model,
                initial_state,
                _step_size=args.step_size,
                num_results=args.n_samples,
                num_burnin_steps=args.n_burnin,
                _n_lfs=args.n_lfs,
                mass_sqrt_diag=mass_sqrt_diag,
            )
        samps_np = samples.numpy()
        inner = getattr(results, "inner_results", results)
        try:
            logp_np = inner.target_log_prob.numpy()
        except AttributeError:
            logp_np = np.full(args.n_samples, np.nan)
        try:
            accept_rate = float(inner.is_accepted.numpy().mean())
        except AttributeError:
            accept_rate = np.nan
        try:
            final_step_size = float(results.new_step_size.numpy()[-1])
            print(f"Adapted step size: {final_step_size:.6f}")
        except Exception:
            pass

    elapsed = time.time() - t_chain
    print(f"Chain {args.chain_id} complete in {elapsed/3600:.2f}h")

    filename = os.path.join(args.output_dir, f"chain_{args.chain_id}.npz")
    np.savez(filename, samples=samps_np, logp=logp_np, accept_rate=accept_rate, sampler=args.sampler)
    print(f"Saved results to {filename}")

if __name__ == "__main__":
    main()
