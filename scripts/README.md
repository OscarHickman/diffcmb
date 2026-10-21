# scripts/

Production, validation and figure scripts for the diffcmb analysis. The
package itself is in `diffcmb/diffcmb/`; unit tests are in `tests/`.

Run every command from the repository root, with the package on the path:

```bash
PYTHONPATH=diffcmb .venv/bin/python scripts/<script>.py --help
```

| Location | Contents |
|---|---|
| `scripts/*.py` | The current analysis pipeline and the modules other scripts import: the chain driver `coverage_ensemble_chain.py`, real-sky setup (`real_sky_setup.py`, `prepare_real_data.py`), harvesting and calibration (`aggregate_coverage_ranks.py`, `sbc_joint_likelihood.py`, `strict_sbc_ranks.py`, `validate_*.py`, `null_alm_power_rank_flat_prior.py`, `diagnose_calibration_stationarity.py`), the real-data summaries (`realdata_*.py`) and the neural posterior estimation route (`npe_*.py`). These scripts import one another by bare module name, so they are kept together in one directory. |
| `paper/` | One script per paper figure, plus `paper_style.py`. `make figures` runs them. |
| `diagnostics/` | One-off checks from the exact-operator analysis (band-edge, dense reference, real-sky versus full-sky comparison, mask edge and similar). Their results are recorded in `achievements.md`. |
| `slurm/` | SLURM wrappers for the current scripts. Submit them from the repository root, for example `sbatch scripts/slurm/submit_ensemble_exact_lmax64.slurm`. |
| `archive/` | Superseded scripts from before the exact lensing operator (2026-09-23): legacy samplers and pilots, `debug_*`, `gate_*`, `smoke_*`, benchmarks and messenger/MCLMC studies. `archive/slurm/` holds their wrappers. They are kept for the record and are not maintained. |

## Tests

```bash
make test        # or: PYTHONPATH=diffcmb .venv/bin/python -m pytest -q
```

`tests/conftest.py` puts `diffcmb/`, `scripts/`, `scripts/paper/` and
`scripts/diagnostics/` on `sys.path`, so a test can import any of these scripts
by module name. Tests marked `slow` are deselected by default (run them with
`-m slow`). Tests marked `needs_cosma_data` are skipped when
`results/analysis/` is not available.
