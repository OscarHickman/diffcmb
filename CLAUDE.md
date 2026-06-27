# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests
make test
# or
PYTHONPATH=diffcmb pytest

# Run a single test file
PYTHONPATH=diffcmb pytest tests/test_alm.py

# Lint (ruff via pre-commit)
make precommit
# or directly:
ruff check diffcmb/ tests/ --fix

# Set up virtualenv with all dependencies
make setup

# Build the Rust spherical-harmonic extension (optional but recommended)
make build-rust

# Run the minimal entry point
PYTHONPATH=diffcmb python Main.py
```

## Architecture

The package lives in `diffcmb/diffcmb/` and is structured as a pipeline from raw cosmological parameters to MCMC samples:

```
CAMB params → power.py → alm_utils.py → model.py → samplers.py
```

The `diffcmb/rust_sph/` directory contains an optional Rust extension that parallelises spherical harmonic matrix construction using Rayon, providing significant speedups for large lmax.

### Module responsibilities

- **`power.py`** — Calls CAMB to produce a CMB angular power spectrum (`C_l` array). Only works for `lmax ≤ 2551`. Default ΛCDM parameters are `[H0=67.74, ombh2=0.0486, omch2=0.2589, mnu=0.06, omk=0.0, tau=0.066]`.

- **`alm.py`** — Minimal utilities: adding Gaussian noise to a pixel map (`noisemapfunc`) and a single-pixel spherical harmonic evaluation (`sphharm`).

- **`alm_utils.py`** — All alm/map transforms. There are **two alm index orderings** in use:
  - *Author ordering* (`mo`): row-major by `(L, m)` — used internally in `psi`
  - *Healpy ordering* (`ho`): column-major by `m` — used by all `hp.*` functions
  - `almmotho` converts author→healpy; `almhotmo` converts healpy→author. Functions prefixed with `hp` use healpy ordering; bare names use author ordering.
  - TF variants of core transforms (`splittosingularalm_tf`, `almtomap_tf`) accept and return TensorFlow tensors.

- **`tf_helpers.py`** — Builds `shape`, the `(lmax × len_alm)` tensor of `1.0`/`2.0` weights used in the `psi3` term.

- **`model.py`** — `CosmologyAdvancedSampling` is the central class. Its `__init__` runs the full setup pipeline (CAMB → alms → prior map → initial parameter vector `x0`). TensorFlow-dependent tensors (`self.sph`, `self.shape`) are created **lazily** on the first call to `psi_tf` via `_ensure_tf_tensors()`, to allow importing without TF. `psi_tf` is the negative log-posterior used as the target for MCMC.

- **`samplers.py`** — Thin wrappers around `tfp.mcmc.HamiltonianMonteCarlo` (`run_chain_hmc`) and `tfp.mcmc.NoUTurnSampler` (`run_chain_nut`). Both take a `CosmologyAdvancedSampling` instance and an initial state tensor.

### Dependency guards

All heavy dependencies (`healpy`, `scipy`, `tensorflow`, `tensorflow_probability`, `camb`) are imported with `try/except` at module level and set to `None` on failure. Functions that need them raise `ImportError` at call time. This keeps the package importable in restricted environments (e.g. for lightweight testing).

### Parameter vector layout (`x0` / `_params`)

The sampled parameter vector encodes:
1. `_lncl[2 : lmax]` — log power spectrum coefficients (length `lmax - 2`)
2. `_realalm` — real parts of alm coefficients for `L ≥ 2, m ≥ 0` (excluding monopole/dipole and low-`m` imaginary parts)
3. `_imagalm` — imaginary parts for `m ≥ 2`
