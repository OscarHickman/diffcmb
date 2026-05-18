# Project Instructions: cosmology-from-the-cmb-with-advanced-sampling-techniques

## Project Overview
This repository contains tools for performing CMB analysis and sampling using TensorFlow Probability. It implements advanced sampling techniques (HMC, NUTS) for cosmological parameter estimation. The codebase is structured as a Python package (`src/cmb`).

## Dependencies
- **Core**: `tensorflow`, `tensorflow-probability`, `healpy`, `camb`, `astropy`, `emcee`, `numpy`, `scipy`, `pandas`.
- **Infrastructure**: `mpi4py`, `pyyaml`.
- **Development**: `pytest`, `ruff`, `pre-commit`.

## Workflows

### Testing
Run the project's unit tests using `pytest`:
```bash
pytest
```

### HPC Execution (COSMA)
This project is designed for the COSMA HPC environment.
- **Job Submission**: Use the script in `scripts/submit_chains.slurm`.
- **Command**: `submit scripts/submit_chains.slurm` (via custom `submit` alias).
- **Data Partitioning**: If jobs read/write to `/cosma5`, ensure they run on `cosma5` or compatible `shm`/`ska` partitions as per `~/.gemini/GEMINI.md`.

### Development
- **Linting/Formatting**: The project uses `ruff`.
- **Code Organization**: Core logic resides in `src/cmb/` and `src/rust_sph/`.
  - `src/cmb/`: Python package (`model.py`, `samplers.py`, `power.py`, `alm_utils.py`).
  - `src/rust_sph/`: Rust extension for parallelised spherical harmonic matrix construction.
- **Commands**:
  - `make setup`: Set up virtualenv and dependencies.
  - `make build-rust`: Build the Rust extension.
  - `make test`: Run tests.

## Conventions
- **Naming**: Consistent with Python `snake_case`.
- **Organization**: Package-based structure in `src/`.
- **Notebooks**: `examples/` contains pedagogical notebooks; `CMB_with_advanced_sampling_techniques.ipynb` is a legacy reference.
