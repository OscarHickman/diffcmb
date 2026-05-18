import os
import numpy as np

def load_cmb_chains(results_dir):
    """
    Loads MCMC chains from .npz files in results_dir.
    Files are expected to be named chain_1.npz, chain_2.npz, etc.
    """
    chains_samples = []
    chains_logprob = []
    chains_accepted = []
    
    # Find all chain_*.npz files
    files = sorted([f for f in os.listdir(results_dir) if f.startswith('chain_') and f.endswith('.npz')])
    
    for f in files:
        path = os.path.join(results_dir, f)
        data = np.load(path)
        chains_samples.append(data['samples'])
        chains_logprob.append(data['logp'])
        chains_accepted.append(data['accept_rate'])
        
    print(f"Loaded {len(chains_samples)} chains from {results_dir}")
    return chains_samples, chains_logprob, chains_accepted
