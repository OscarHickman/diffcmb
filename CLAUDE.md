# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project documents — read these before planning any nontrivial work

**State as of 2026-09-23:** everything before that date used the legacy forward model. Three exact-operator ensembles (amplified lensing A_φ = 3000, σ = 30 μK/pixel, lmax = 64) are complete and were first harvested 2026-09-23: physics clean, **calibration fails** (joint-likelihood SBC rejected on Block-4-ON; a_ℓm `[30,60)` rank offset in both ensembles; a_ℓm ESS ~35/1200). `ROADMAP.md` → T0.1 (diagnose it) is blocking — its thinning/stationarity steps (a, b) are job 12040935, not yet harvested; numbers in `results/analysis/dashboard.md`. Work is on branch `exact-operator-rerun` in both `diffcmb` and `papers/7_DiffCMB`.

This is an active research project (differentiable, curved-sky, joint Bayesian CMB lensing analysis) with three living docs at the repo root, each with a distinct job — don't duplicate content across them:

- **`ROADMAP.md`** — forward-looking only. The todo list of all outstanding work, in priority order, with a "Standing discipline" section of hard-won rules (precision, validation-before-production, claims hygiene). Check here first for "what's next."
- **`achievements.md`** — condensed record of everything validated, closed-out, or fixed so far, including sampler routes that were tried and abandoned (with why, so they aren't retried without new evidence) and real bugs caught by validation. Check here before proposing an approach that might already be tried.
- **`literature.md`** — annotated bibliography and the novelty/positioning argument (what's been done in the field, what hasn't, why the claim still holds). Check here before any framing/motivation writing. Its standing claims-hygiene rule requires re-scanning arXiv **both** for curved-sky samplers/MUSE *and* for the generative/diffusion route before every submission milestone.
- **`docs/notes/`** — longer-form research/design notes that would bloat the three docs above: `cmblensing_benchmark_notes.md` (the CMBLensing.jl comparison design, now written up in `docs/paper/main.tex::sec:cmblensing`) and `restore_missing_alm_dof_scoping.md` (the staged plan for the missing `Im(a_{L,1})` dof — restored 2026-09-06, kept for its record of every packing-dependent site; read it before touching the packing). Referenced from `ROADMAP.md` rather than duplicated into it.
- **`results/analysis/dashboard.md`** — live status of the production chains: current headline SBC numbers, what is in flight, which output directories are invalid, and how to read the spectrum rows. Check it before harvesting or citing any chain.

When asked to "continue the roadmap," read `ROADMAP.md`'s next unchecked item and `achievements.md`'s most recent entries for context on what's already been validated.

## Commands

```bash
# Run all tests (use make, or the venv python explicitly -- the *system* python
# lacks TF/healpy and silently degrades to ~69 skips plus spurious ImportError
# failures that look like real breakage)
make test
# or
PYTHONPATH=diffcmb .venv/bin/python -m pytest -q

# Run a single test file
PYTHONPATH=diffcmb .venv/bin/python -m pytest tests/test_alm.py

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

The package lives in `diffcmb/diffcmb/` and is structured as a pipeline from raw cosmological parameters to MCMC samples, culminating in a Gibbs sampler over (alm, C_ℓ, φ, and optionally C_L^φφ):

```
CAMB params → power.py → alm_utils.py → model.py ─┬─ samplers.py (C_ℓ|alm exact, alm|C_ℓ,φ HMC)
                                                    └─ lensing.py + sht_ducc.py (φ|alm,C_ℓ HMC,
                                                                                  C_L^φφ|φ exact)
```

The `diffcmb/rust_sph/` directory contains an optional Rust extension that parallelises spherical harmonic matrix construction using Rayon, providing significant speedups for large lmax. It is superseded for production-scale (lmax≳300) runs by the matrix-free ducc0 SHT path (`use_matrixfree_sht=True`, see below) — the dense matrix (Rust-accelerated or not) doesn't scale.

### Module responsibilities

- **`power.py`** — CAMB spectra. **Production uses `fiducial_spectra(lmax)`** (since 2026-09-23): raw unlensed C_ℓ^TT and C_L^φφ at `LCDM_PARAMS` (physical densities ω_b = 0.0223, ω_c = 0.1188). The legacy `call_CAMB_map` + `LCDM_PARAMS_LEGACY = [67.74, 0.0486, 0.2589, ...]` had three defects: Ω passed as ω, C_ℓ low by 2π, and lensed TT used as unlensed. They are kept only so legacy chains can be reproduced; replays read spectra from the chain file, never from CAMB.

- **`alm.py`** — Minimal utilities: adding Gaussian noise to a pixel map (`noisemapfunc`) and a single-pixel spherical harmonic evaluation (`sphharm`).

- **`alm_utils.py`** — All alm/map transforms. There are **two alm index orderings** in use:
  - *Author ordering* (`mo`): row-major by `(L, m)` — used internally in `psi`
  - *Healpy ordering* (`ho`): column-major by `m` — used by all `hp.*` functions
  - `almmotho` converts author→healpy; `almhotmo` converts healpy→author. Functions prefixed with `hp` use healpy ordering; bare names use author ordering.
  - TF variants of core transforms (`splittosingularalm_tf`, `almtomap_tf`) accept and return TensorFlow tensors.
  - **`packed_sizes(lmax)` / `packed_length(lmax)`** — the single definition of the packed vector's `(n_real, n_imag)` split. Every call site derives its lengths from these; never re-write the formula inline (that literal appeared 75 times before the 2026-09-06 restoration and was the main cost of changing the layout).
  - **`packed_dof_per_multipole(lmax)` / `invgamma_shape_for_spectrum(lmax, a0)`** — the number of *real* degrees of freedom the packed vector carries at each multipole, and the resulting inverse-Gamma shape for Blocks 1 and 4. **It is 2L+1 since the 2026-09-06 `Im(a_{L,1})` restoration** (it was 2L before; see the parameter-vector section below). Both spectrum blocks derive their shape from these rather than hardcoding a formula, so the conditionals stay correct if the packing ever changes.

- **`tf_helpers.py`** — Builds `shape`, the `(lmax × len_alm)` tensor of `1.0`/`2.0` weights used in the `psi3` term.

- **`sht_ducc.py`** — Matrix-free spherical harmonic transforms via ducc0, wrapped in `tf.custom_gradient` (a `tf.py_function` escape hatch, since ducc0 isn't TF-native). `HealpixSHT` (masked-sky synthesis, `nthreads` param) and `full_synthesis_tf` (full-sky synthesis — lensing needs this because deflected positions can fall outside the eventual mask). This is what makes lmax=300+ tractable at all: the dense SHT matrix doesn't fit in GPU memory and is ~500x slower per call.

- **`lensing.py`** — The differentiable weak-lensing forward operator. **Two operators, selected by `model.lensing_operator`:** `'exact'` (production since 2026-09-23; `lens_alm_exact_tf`) evaluates the band-limited alm at *geodesically* deflected positions with ducc0 `synthesis_general`, using the physical sign T(n̂+∇φ); `'bilinear'` (legacy, the default for backward compatibility) interpolates the pixel map with `hp.get_interp_weights` and lenses by −∇φ (`DEFLECTION_SIGN_LEGACY`). At nside = lmax the bilinear smoothing dominated physical lensing, which is what retracted the old bias-reduction result (`achievements.md`). `apply_lensing_tf`/`lens_map_tf` deflect an unlensed alm through φ to a lensed map; `psi_lensed` is the corresponding negative-log-posterior for the φ|alm,C_ℓ HMC block. Both branch on `model.use_matrixfree_sht`: `True` routes through `sht_ducc.py::full_synthesis_tf`; `False` uses the dense `model.sph_parts` Y-matrix. Both full-sky **and** masked-sky matrix-free paths are validated (see `achievements.md`; at nonzero φ the dense path is *not* a valid masked-sky reference, so that case is validated by finite differences on the matrix-free path itself). Also holds `estimate_phi_diag_fisher` (opt-in Fisher curvature for the φ mass matrix — implemented but closed as unfavorable, see `achievements.md`) and `compute_sl_phi_np`/`sample_cl_phiphi_given_phi` (the Block 4 exact C_L^φφ|φ inverse-Gamma draw, mirroring `model.py`'s Block 1 pair). `sample_cl_phiphi_given_phi` takes optional `prior_nu`/`cl_phiphi_fid`: the default flat prior on C_L^φφ is **improper** (integrating it out leaves a φ marginal flat in S_L, rising with amplitude), so `prior_nu=ν>0` puts a proper conjugate `InvGamma(ν/2, ν·C_L^fid/2)` prior on it instead — required for any strict SBC statement about C_L^φφ.

- **`model.py`** — `CosmologyAdvancedSampling` is the central class. Its `__init__` runs the full setup pipeline (CAMB → alms → prior map → initial parameter vector `x0`), and accepts `use_matrixfree_sht=False, sht_nthreads=0` to select the SHT backend and `lensing_operator='bilinear'|'exact'` (`'exact'` requires the matrix-free path; production scripts default to it). Two opt-in realism kwargs follow the same "None = old behaviour, zero effect on existing call sites" pattern and are both validated against independent healpy/analytic ground truths: `beam_fwhm_arcmin=None` (Gaussian beam × HEALPix pixel window, a diagonal multiply on the unlensed alm, built by `power.py::beam_pixwin_transfer`) and `noise_map=None` (a length-NPIX per-pixel noise **sigma** array giving a spatially varying `self.Ninv = 1/noise_map**2` in place of the uniform `1/_noisesig**2`). TensorFlow-dependent tensors (`self.sph`, `self.shape`) are created **lazily** on the first call via `_ensure_tf_tensors()`, to allow importing without TF. `psi_tf` is the negative log-posterior for the unlensed alm|C_ℓ block. `compute_sl_np` computes the exact S_l = Σ_m |a_lm|² (with correct packed real/imag m=0-vs-m>0 weighting) that `sample_cl_given_alm`'s exact inverse-Gamma C_ℓ|alm conditional is built on. That conditional is **`C_l|alm ~ InvGamma(k_l/2 - 1, S_l/2)` with `k_l` the packed dof**, read from `invgamma_shape_for_spectrum` rather than hardcoded. `k_l` was `2l+1` (hardcoded, wrong for the packing then in force) until 2026-08-31, `2l` (derived) until the 2026-09-06 `Im(a_{L,1})` restoration, and is `2l+1` again now — this time because the packing genuinely carries it. See `achievements.md`.

- **`samplers.py`** — `run_chain_hmc`/`run_chain_nut` are thin wrappers around `tfp.mcmc.HamiltonianMonteCarlo`/`NoUTurnSampler` for single-block sampling. `run_gibbs_chain` is the production Gibbs driver, up to 4 blocks: Block 1 (C_ℓ|alm) is an exact inverse-Gamma draw; Block 2 (alm|C_ℓ,φ) is HMC (or `alm_sampler='cg'`, closed as biased when a φ block is active — see `achievements.md`); Block 3 (φ|alm,C_ℓ), enabled by passing `cl_phiphi_full`, is HMC and requires `alm_sampler` in `('hmc','cg')`; Block 4 (C_L^φφ|φ), enabled by `sample_cl_phiphi=True` (requires Block 3), is another exact inverse-Gamma draw that resamples the φ power spectrum every sweep and rebuilds the φ mass matrix to match — it adds a `cl_phiphi_samples` array to the return tuple and is mutually exclusive with `phi_mass_matrix='fisher'`. `cl_phiphi_prior_nu` switches Block 4 to the proper conjugate prior (above); its fiducial is snapshotted **before** the sweep loop, because `run_gibbs_chain` rebinds `cl_phiphi_full` every sweep and a prior centred on that moving value would chase the chain and exert no restoring force. `seed` seeds **both** the numpy stream and TensorFlow's global stream — until 2026-08-31 it seeded only numpy, so HMC momenta came from process-global TF state and "seeded" chains were not reproducible. Note the φ-block trajectory length (`phi_n_lfs`) is the parameter that actually controls φ mixing — see `achievements.md`'s leapfrog-schedule entry before tuning anything else. Supports checkpointing (`checkpoint_path`/`checkpoint_every`) that resumes alm-, phi-, and (when enabled) C_L^φφ state, needed because production lmax=300 chains run tens of seconds to minutes per sweep under a SLURM walltime.

### Dependency guards

All heavy dependencies (`healpy`, `scipy`, `tensorflow`, `tensorflow_probability`, `camb`) are imported with `try/except` at module level and set to `None` on failure. Functions that need them raise `ImportError` at call time. This keeps the package importable in restricted environments (e.g. for lightweight testing).

### Parameter vector layout (`x0` / `_params`)

The sampled parameter vector encodes:
1. `_lncl[2 : lmax]` — log power spectrum coefficients (length `lmax - 2`)
2. `_realalm` — real parts of alm coefficients for `L ≥ 2, m ≥ 0` (excluding monopole/dipole)
3. `_imagalm` — imaginary parts for `m ≥ 1`

**⚠ This layout carries 2L+1 real dof per multipole since the `Im(a_{L,1})` restoration (2026-09-06); every saved chain and checkpoint written before that carries 2L and is a different model.** `alm_utils.py::splittosingularalm` writes `complex(real, 0)` only when `m == 0`, which is correct and required (a real field's m=0 coefficient is real). It used to write it for `m == 0 or m == 1` as well, forcing `Im(a_{L,1}) = 0` and making the model unable to represent a general sky — ~20% of the modes missing at ℓ=2, 0.8% at ℓ=63.

- **Statistical:** the Gaussian prior normalisation is `C^{-k_L/2}` with `k_L = 2L+1`, so the exact conditional against a prior `InvGamma(a0,b0)` is `InvGamma(k_L/2 + a0, b0 + S_L/2)` = `InvGamma(L - 0.5, S_L/2)` under a flat prior. Use `invgamma_shape_for_spectrum`, never a hardcoded formula — the shape is `L - 0.5` again, which is what the code had *before* 2026-08-31, so don't read the git history as a fix that was reverted: the 2L era was a real and separately-correct intermediate state.
- **Every φ number measured before 2026-09-06 was measured through the 2L packing** and is a statement about a different model. That includes the pre-restoration headline SBC pair (φ 0.4688 / alm 0.5312, job 11903181). **Re-confirmed 2026-09-08 under the restored 2L+1 packing** (job 11955622, 12 realizations, same configuration): φ 0.4792 (KS_p 0.4078) / alm 0.5130 (KS_p 0.6369), both uniform. ⚠ **That and every other pre-2026-09-23 result is now legacy** (bilinear operator, −∇φ, defective fiducial). The current certification comes from the exact-operator ensembles `results/analysis/ens_exact_l64_A3000_n30{,_nocl4}` (see `achievements.md` and `results/analysis/dashboard.md`).
- **Checkpoints are versioned.** `samplers.PACKING_VERSION` is written into every checkpoint and `run_gibbs_chain` refuses to resume one whose version or vector length does not match, instead of silently resuming into a wrong-length vector. Bump it if the layout changes again.
- Background and the staged plan this followed: `docs/notes/restore_missing_alm_dof_scoping.md`.

### `tests/` vs `scripts/`

`tests/` holds fast, deterministic pytest coverage at small lmax (`test_lensing.py`, `test_lensing_exact.py`, `test_samplers.py`, `test_sht_ducc.py`, etc.) — run these for any code change. Since 2026-09-23 the suite also covers `scripts/`: `test_scripts_pipeline.py` runs `coverage_ensemble_chain.py` end to end (aware and blind), replays chains, and runs the validation scripts; `test_paper_figures.py` runs every figure script on synthetic ensembles. `scripts/` is production/validation infrastructure, not unit tests: `gate_*.py` are one-off go/no-go checks before scaling a sampler configuration to lmax=300 (results recorded in `achievements.md`, not re-run routinely); `debug_*.py` are investigation scripts from specific past bugs (see `achievements.md`'s "Real bugs" list for which); `plot_joint_cl_clpp_posterior.py` builds the paper's differentiator figure (within-posterior `C_ℓ^TT`-`C_L^φφ` correlation) from a Block-4-ON, proper-prior ensemble — it standardises each chain before pooling, bootstraps over *chains* rather than sweeps, and scores against a within-chain permutation null, because at realistic effective sample sizes the estimator's own noise floor (|r| ~ 0.15) is the same size as the effect; `validate_coverage_rank_nulls.py` calibrates the coverage-ensemble rank statistics against what a *correct* sampler produces — **run it before reading any spectrum-row FLAG from `aggregate_coverage_ranks.py` as bias**, because that statistic ranks the truth against its own conditional's mode and is non-uniform by construction. It also carries the two *strict* checks: `block4_exactness` (a PIT of Block 4 against its own conditional — it runs the misalignment control at several lags via `--control_lags`, default `1,10,50`, prints the measured φ autocorrelation beside each, and refuses to call the aligned result a pass unless one control is rejected; **lag-1 alone is not enough** — under the Block-4-ON funnel φ barely moves sweep-to-sweep, so the lag-1 control passes and makes the statistic vacuous) and `strict_clpp_sbc` (rank of `cl_phiphi_true` among the Block 4 samples; uniform under a correct sampler with no null needed, but defined only for runs with `--cl_phiphi_prior_nu`, since only then is the truth drawn from the sampler's own prior). **Both this script and `aggregate_coverage_ranks.py` re-implement the spectrum conditionals, so they are inside the blast radius of any change to the dof or prior — they read the shape and prior off the saved chain rather than hardcoding, and must stay that way** (they hardcoded `2L+1` for a window after the samplers were fixed, and silently compared corrected chains against a stale reference); `diagnose_calibration_stationarity.py` (T0.1) separates thinning, burn-in and sampler defects in a failed rank test — field ranks by thin and chain half, rank mean per chain quarter, across-chain drift z — tested in `tests/test_diagnose_stationarity.py`; `submit_*.slurm`/`*.slurm` are the SLURM wrappers for the matching `.py` script, following a consistent pattern (`-A durham -p dine2`, per-job `$TMPDIR`, `PYTHONPATH` set to `diffcmb/`). When adding a new production-scale validation, mirror an existing gate/smoke script's structure rather than inventing a new one.
