# Cosmology-from-the-CMB-with-advanced-sampling-techniques

[![Python Tests](https://github.com/OscarHickman/CMB_Advanced_Sampling/actions/workflows/test.yml/badge.svg)](https://github.com/OscarHickman/CMB_Advanced_Sampling/actions/workflows/test.yml)

The aim of this repository is to accurately sample from the CMB using Tensorflow probability and advanced sampling techniques. 
 

Code is written for python and uses the package healpy - which is only supported by linux and macos. Windows users must therefore either use Google Colab or a virtualbox to use the healpy functions in this repository. The other packages which must be installed prior to use are CAMB, Tensorflow and Tensorflow Probability.

## Project Structure

The code has been fully refactored into a Python package structure:

### Source Code (`src/cmb/`)
All functionality has been extracted into organized modules:
- **`power.py`** - CAMB power spectrum generation (`call_CAMB_map`)
- **`alm.py`** - Basic alm/map utilities (`noisemapfunc`, `sphharm`)
- **`alm_utils.py`** - Comprehensive alm transformation utilities (cltoalm, hpcltomap, almtomap, almtocl, etc.)
- **`tf_helpers.py`** - TensorFlow tensor construction helpers (`multtensor`)
- **`model.py`** - Main `CosmologyAdvancedSampling` class with `psi` and `psi_tf` methods
- **`samplers.py`** - MCMC samplers (`run_chain_hmc`, `run_chain_nut`)

### Examples (`examples/`)
Example Jupyter notebooks demonstrating usage:
- **`basic_usage.ipynb`** - Getting started guide with the refactored package
- **`hmc_sampling.ipynb`** - HMC sampling examples
- **`nut_sampling.ipynb`** - NUTS sampling examples

### Tests (`tests/`)
Unit tests using pytest:
- **`test_alm.py`** - Tests for alm utilities
- **`test_power.py`** - Tests for power spectrum functions

### Legacy Files
- **`CMB_with_advanced_sampling_techniques.ipynb`** - Original usage examples and demonstrations

## Usage

```python
from src.cmb import CosmologyAdvancedSampling, run_chain_hmc

# Create model
model = CosmologyAdvancedSampling(_lmax=8, _NSIDE=2, _noisesig=1.0)

# Run HMC sampling
samples, results = run_chain_hmc(model, initial_state, num_results=1000)
```

See `examples/basic_usage.ipynb` for a complete walkthrough.

## Installation

To run the tests and examples, install dependencies:
```bash
pip install -r requirements.txt
pytest  # Run tests
```