import sys
import os
import time
import argparse
import numpy as np

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    import tensorflow as tf
    from src.cmb import CosmologyAdvancedSampling, run_chain_nut, run_chain_hmc
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
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"=== Chain {args.chain_id} Starting ===")
    print(f"Sampler: {args.sampler.upper()}")
    print(f"Data: {args.data_mode}")
    print(f"LMAX: {args.lmax}, NSIDE: {args.nside}, Noise: {args.noise_sig}")
    print(f"Samples: {args.n_samples}, Burn-in: {args.n_burnin}, Step Size: {args.step_size}")

    print("Constructing model...")
    t0 = time.time()
    model = CosmologyAdvancedSampling(
        _lmax=args.lmax, 
        _NSIDE=args.nside, 
        _noisesig=args.noise_sig,
        data_mode=args.data_mode,
        data_dir=args.data_dir
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

    # Convert to numpy for saving
    samps_np = samples.numpy()
    logp_np = results.target_log_prob.numpy()
    
    try:
        accept_rate = float(results.is_accepted.numpy().mean())
    except AttributeError:
        accept_rate = np.nan

    filename = os.path.join(args.output_dir, f"chain_{args.chain_id}.npz")
    np.savez(filename, samples=samps_np, logp=logp_np, accept_rate=accept_rate, sampler=args.sampler)
    print(f"Saved results to {filename}")

if __name__ == "__main__":
    main()
