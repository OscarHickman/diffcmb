import argparse
import os
import sys
import time

import numpy as np

# Ensure repo root is in path so 'from src.cmb import ...' resolves
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import tensorflow as tf

    from src.cmb import CosmologyAdvancedSampling, run_chain_hmc, run_chain_nut
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run MCMC chain for CMB sampling.")
    parser.add_argument("--sampler", type=str, choices=["nuts", "hmc"], default="nuts")
    parser.add_argument("--lmax", type=int, default=200)
    parser.add_argument("--nside", type=int, default=128)
    parser.add_argument("--noise_sig", type=float, default=1.0)
    parser.add_argument("--data_mode", type=str, choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--n_burnin", type=int, default=500)
    parser.add_argument("--step_size", type=float, default=0.01)
    parser.add_argument("--n_lfs", type=int, default=2, help="Number of leapfrog steps (HMC only)")
    parser.add_argument("--chain_id", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--parameterization", type=str, choices=["centered", "non-centered"], default="centered",
                        help="Sampling strategy: 'centered' (standard) or 'non-centered' (reparameterized for speed)")
    parser.add_argument("--data_seed", type=int, default=42,
                        help="RNG seed for synthetic data generation (fixed so all chains share the same dataset)")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"=== Chain {args.chain_id} Starting ===")
    print(f"Sampler: {args.sampler.upper()}")
    print(f"Data: {args.data_mode}  (data_seed={args.data_seed})")
    print(f"Parameterization: {args.parameterization}")
    print(f"LMAX: {args.lmax}, NSIDE: {args.nside}, Noise: {args.noise_sig}")
    print(f"Samples: {args.n_samples}, Burn-in: {args.n_burnin}, Step Size: {args.step_size}")

    # Fix the data-generation RNG so all chains sample the same posterior.
    if args.data_mode == "synthetic":
        np.random.seed(args.data_seed)

    print("Constructing model...")
    t0 = time.time()
    model = CosmologyAdvancedSampling(
        _lmax=args.lmax,
        _NSIDE=args.nside,
        _noisesig=args.noise_sig,
        data_mode=args.data_mode,
        data_dir=args.data_dir,
        parameterization=args.parameterization
    )
    print(f"Model init took {time.time()-t0:.1f}s")

    print("Pre-loading spherical harmonic matrix...")
    t1 = time.time()
    model._ensure_tf_tensors()
    print(f"Tensor loading took {time.time()-t1:.1f}s")

    initial_state = model.prior_parameters_tf()

    print(f"Starting {args.sampler.upper()} sampling for chain {args.chain_id}...")
    t_chain = time.time()

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
            _n_lfs=args.n_lfs
        )

    elapsed = time.time() - t_chain
    print(f"Chain {args.chain_id} complete in {elapsed/3600:.2f}h")

    # Convert to numpy for saving.
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

    filename = os.path.join(args.output_dir, f"chain_{args.chain_id}.npz")
    np.savez(filename, samples=samps_np, logp=logp_np, accept_rate=accept_rate, sampler=args.sampler)
    print(f"Saved results to {filename}")

if __name__ == "__main__":
    main()
