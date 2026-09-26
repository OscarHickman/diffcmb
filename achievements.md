# Achievements — DiffCMB

*Condensed record of what is validated, closed out, retracted or fixed — current state, not a change log. Full history is in git. Forward plan: `ROADMAP.md`.*

**Read first.** Everything produced before 2026-09-23 was made with the **legacy forward model**: bilinear-interpolation lensing on an nside = lmax grid, deflection by −∇φ, and a defective fiducial cosmology. Those results remain valid as statements about *the sampler inverting the model it was given*, and they are reproducible (`lensing_operator='bilinear'`, chains read their own spectra). They are **not** statements about lensing on the sky. Every paper number is being re-derived from the exact-operator ensembles (first section below).

---

## 2026-09-22/23 — operator retraction, physics fixes, and the exact-operator re-run

### The retraction: the "lensing bias" was the interpolation operator

`scripts/validate_lensing_power_transfer.py` (job 12037446, `results/analysis/lensing_power_transfer.npz`) lensed the same Gaussian skies five ways. The production operator alone reproduced figure 2's lensing-blind deficit: **−0.24 / −0.93 / −2.79 / −5.30 %** in `[10,30)/[30,60)/[60,100)/[100,128)` at lmax = 128 (figure 2 had −0.2/−0.9/−2.6/−5.4), and −3.54 / −5.64 / −8.24 % in `[100,128)/[128,160)/[160,192)` at lmax = 192. That is also why the deficit at a fixed ℓ shrank as lmax = nside grew. Exact evaluation at the same deflected angles changes in-band power by ≤ 0.5 %. The physical effect (unbandlimited input, and CAMB lensed/unlensed) is **+0.05 to +0.2 %**, the opposite sign and 30–50× smaller.

**Retracted:** the 93.7 % ± 1.8 % (N=4, lmax=128) and 98.4 % (lmax=192) "lensing-bias reduction" as a statement about lensing, and its explanation ("lensing smooths the acoustic peaks"). **Still standing:** the aware sampler was unbiased with respect to its own nonlinear, φ-dependent forward model.

### Physical lensing information at these scales is essentially nil

With the corrected fiducial, the TT quadratic estimator's total S/N on C_L^φφ is **0.002 / 0.013 / 0.038 / 0.13** at lmax = 64 / 128 / 192 / 300 (median N_L/C_L ≈ 24,000 / 6,300 / 3,200 / 1,900). This is independent of noise, because the limit is cosmic variance of the unlensed field; temperature lensing information lives at ℓ ≳ 1000. So the φ information the legacy chains had came from the operator: the φ map recovery (r = 0.67), figure 3's "information beyond the QE", and the non-trivial φ posteriors in the rank tests.

### Fixed at the source before the re-run

| defect | effect | fix |
|---|---|---|
| bilinear interpolation lensing | smoothing ∝ (ℓ/nside)², ~30–50× physical lensing | `lensing.lens_alm_exact_tf`: alm evaluated at the deflected positions (ducc0 `synthesis_general`, ε = 1e-11), adjoint for d/dalm, spin-1 gradient + geodesic Jacobian for d/dφ |
| coordinate-offset displacement | wrong at O(\|d\|² cot θ), clipped at the poles | geodesic remap n′ = exp_n(∇φ) (`_exact_lensing_geometry`), the standard curved-sky convention |
| deflection sign | `deflection_field` used glm = −√(ℓ(ℓ+1)) φ, i.e. **−∇φ** | `DEFLECTION_SIGN_PHYSICAL/LEGACY`; bilinear keeps the legacy sign so old chains replay exactly, exact uses +∇φ |
| QE divergence sign | `qe_tt_reconstruct` returned +div[A∇B], not the documented −div | fixed; it had cancelled against the legacy −∇φ and so passed the old response test |
| fiducial cosmology | Ω_b, Ω_c passed as ω_b, ω_c; C_ℓ low by 2π (D_ℓ/ℓ(ℓ+1)); lensed TT used as unlensed truth | `power.LCDM_PARAMS` (physical densities) + `power.fiducial_spectra` (raw, unlensed TT); legacy list kept as `LCDM_PARAMS_LEGACY` |
| prior precision | \|a\|² prior terms computed in float32 (`model.py`, `psi_lensed`, φ prior) | float64, as Re(a·ā) |
| beam/prior ordering | `psi_lensed` applied the Gaussian prior to the *beamed* alm | prior on the sky alm (harmless while production had no beam) |

Every chain now records `lensing_operator`, `fiducial`, `phi_amplitude` and `noisesig`, and every replay (`sbc_joint_likelihood.chain_lensing_operator`, `fig_maps`, `fig3`) reads them, so a chain is never re-analysed through a different model.

### Amplified-lensing validation (decision D4) and its pilots

Physical lensing is unmeasurable at lmax = 64, so the validation simulates C_L^φφ × A_φ, with the prior centred on the same amplified spectrum. The SBC therefore still tests the sampler's own target. It is stated openly as a stress test.

| pilot (lmax=64, 100 burn-in) | result |
|---|---|
| A_φ=3000 / 10000, σ = 1 μK/pix (jobs 12037774/5) | **Gibbs crawls**: φ power 6–8× truth, ESS ≈ 4/100. At 1 μK the φ\|alm conditional is razor-sharp. |
| A_φ=10000, σ = 30 (job 12037979) | stalls: φ ratio 0.33, ESS 6–16/300 |
| A_φ=10000, σ = 10 (job 12037981) | φ inflated 3–4×, alm accept 0.33 |
| **A_φ=3000, σ = 30 (job 12037980)** | **φ power → 0.94–0.98 of truth, corr(⟨φ⟩, φ_true) 0.77–0.82 in every L bin, ESS 8–43/300, ~12 s/sweep** |

The lever: the QE S/N is cosmic-variance limited (A_φ = 3000: 5.0 at σ = 1, 4.6 at σ = 30), while the alm–φ lock-in loosens as σ². **Production configuration: A_φ = 3000, σ = 30 μK/pixel, lmax = nside = 64, 400 burn-in + 1200 samples, exact operator, corrected fiducial.**

### Production ensembles (identical skies across modes)

`scripts/submit_ensemble_exact_lmax64.slurm` (tasks packed per node, one GPU each, TF memory growth):

| mode | job | output | state 2026-09-23 |
|---|---|---|---|
| Block-4-ON, ν = 6 | 12038494 | `results/analysis/ens_exact_l64_A3000_n30/chain_rNNN.npz` | 24/24 complete |
| lensing-blind (same data) | 12038496 | same dir, `blind_rNNN.npz` | 24/24 complete |
| Block-4-OFF | 12038495 | `..._nocl4/chain_rNNN.npz` | 24/24 complete |

**QE validated at the production configuration** (job 12038409, `results/analysis/qe_noise_validation_exact_A3000_n30.npz`): noise ratio **0.995**, response **0.971** (1.09 / 1.00 / 0.94 / 0.93 by bin). The QE weights and filter now use the lensed spectrum *measured* from exact-operator simulations. That, together with the sign fix, resolves the old 0.883 response.

### First harvest of the exact-operator ensembles (2026-09-23) — physics validated, calibration NOT passed

All 72 tasks completed; no tracebacks; `phi_calibration_ok` 24/24 in both sampled ensembles. Sources:
- `make figures` (log `logs/make_figures_exact.log`);
- harvest job 12040798 (`scripts/submit_harvest_exact_lmax64.slurm`, log `logs/harvest_exact_l64_12040798.out`);
- figure 1's test run on the Block-4-OFF ensemble.

**The physics validation is clean (figure 2).** On 24 same-sky pairs, the blind fit lands on the lensing each sky actually received, bin by bin:

| bin | blind bias | lensing received | aware bias (± SEM) |
|---|---|---|---|
| `[2,10)` | +0.90 % | +1.70 % | **−2.96 ± 1.05 %** |
| `[10,20)` | +6.65 % | +6.42 % | +1.50 ± 0.69 % |
| `[20,30)` | +0.55 % | +0.03 % | +0.76 ± 0.82 % |
| `[30,45)` | −15.72 % | −15.96 % | +0.85 ± 0.56 % |
| `[45,64)` | −45.64 % | −46.02 % | +0.75 ± 0.50 % |

Mean |bias| over bins: blind 13.9 %, aware 1.37 % (90.2 % reduction). The aware residuals are within ~1.5σ except `[2,10)` (−2.8σ) and `[10,20)` (+2.2σ). Given the calibration results below, they are not yet citable as "consistent with zero".

**Calibration fails on the Block-4-ON ensemble (the title's object):**
- joint-likelihood SBC **rejected**: mean_u 0.716, KS_p 0.0011;
- figure 1 field ranks: φ p_bin **0.004** (mean_u 0.625, i.e. posterior φ power sits low; `[2,10)`–`[30,60)` means all p ≤ 0.011); a_ℓm p_bin **< 0.001** (`[30,60)` mean);
- strict C_L^φφ SBC borderline: pooled 0.552, KS_p 0.067; `[30,60)` 0.666, KS_p 0.020; figure 1 p_bin 0.038;
- Block 4's own conditional exact: PIT aligned KS_p 0.937, with lag-10/50 controls rejected at KS_p = 0 (lag-1 passes vacuously, since φ lag-1 autocorrelation is +0.877);
- φ power bias per bin 0.85 / 0.95 / 0.98 / 1.02.

**Block-4-OFF ensemble:**
- joint-likelihood SBC **passes** (0.480, KS_p 0.67);
- φ ranks pass (mean_u 0.497, p_bin 0.039, from one spread test in `[2,10)`);
- **a_ℓm ranks fail**: p_bin < 0.001, `[30,60)` mean (mean_u 0.426 overall, i.e. posterior power sits high);
- φ power bias 1.07 / 0.99 / 1.00 / 1.00.

**The a_ℓm `[30,60)` mean offset appears in both ensembles.** That rules out a single-ensemble fluke, but per standing discipline it is not yet a diagnosed defect.

**Convergence (figure_convergence, Block-4-ON; rank draws 120/chain at thin 10):**
- R̂ > 1.01 for 48 % (Block 1) and 55 % (Block 4) of (chain, multipole) pairs; max 1.54 / 1.44;
- median bulk ESS per 1200 sweeps: Block 1 123, **Block 2 (a_ℓm) 35**, Block 3 (φ) 65, Block 4 80.

All alm and most φ coordinates have fewer effective draws than the rank test uses. The per-mode normalised φ residual has sd 1.108 pooled over 24 skies, against ≈1.03–1.05 for a calibrated posterior, i.e. posteriors too narrow. Map recovery r = 0.82 on realization 0.

**Figure 3 is not interpretable yet.** It reports joint-posterior width / (QE⊕prior) = 0.61 / 0.63 / 0.68 / 0.74 at L ~ 15 / 25 / 38 / 55, but those widths come from the same under-mixed chains, and too-narrow posteriors would produce exactly this.

**Figure 4:** 0/16 cells above null.

**Reading (hypothesis, not a finding):** the amplified-lensing configuration mixes too slowly for 400 + 1200 sweeps. The alms mix worst (ESS ~35), and non-stationarity from the MAP start would bias rank means in exactly this way. Candidates to test, in order:
1. the rank tests at thin ≥ τ_int (≈ 35–40) — does the mean offset survive?
2. first half vs second half of each chain (drift);
3. longer chains (the ensemble costs ~5 h per task at 12 s/sweep);
4. a lower A_φ.

Rank *means* are unbiased under autocorrelation if the chain is stationary, so a surviving mean offset would indict stationarity or the sampler, not the thinning.

### T0.1 calibration diagnosis (2026-09-24/25): a_ℓm is the statistic, φ is funnel mixing, and ν = 30 passes

Sources:
- job 12040935 (stationarity; `logs/diag_calib_exact_12040935.out`);
- job 12042747 (a_ℓm null);
- job 12062700 (long-chain harvest; `logs/harvest_t01c_12062700.out`).

Full tables: `results/analysis/dashboard.md`.

- **a_ℓm `[30,60)` offset = the statistic, not the sampler.** The truth is drawn at a fixed C_ℓ^fid while Block 1's prior is flat, so the power rank is non-uniform by construction. Against the exact-sampler null (`null_alm_power_rank_flat_prior.py`, effective noise), every a_ℓm bin in all four exact-operator ensembles is within |z| < 3. **The three-block core (Block-4-OFF) is certified on the exact operator.**
- **Block-4-ON low-L φ burn-in resolved by length** (ν = 6, 1000 + 3600 sweeps, job 12047785): φ `[2,10)` drift z 4.05 → 1.27, logp flat, and the joint-likelihood SBC no longer rejects (0.648, KS_p 0.058). But low-L φ τ_int is **~430 sweeps**. The 1200-sweep estimate of 173 was truncated.
- **The φ `[30,60)` offset depends on ν**: 0.685 (p 0.002) at ν = 6 vs 0.574 (p 0.20) at ν = 30 (job 12047786). The truth is drawn from each ν's own prior, so an exact sampler is uniform at any ν. This is therefore a **mixing defect of the centred (φ, C_L^φφ) hierarchy under a weak hyperprior**, not a prior mismatch. Block 4's own conditional is exact at both ν: PIT KS_p 0.26 / 0.47, with the lag-10/50 controls rejected.
- **ν = 30, 3600 sweeps passes every calibrated test:**
  - joint-likelihood SBC 0.474, KS_p 0.49;
  - all φ field-rank bins;
  - strict C_L^φφ SBC pooled 0.487, KS_p 0.69;
  - Block 4 PIT;
  - a_ℓm within the null.
  
  Which ν the paper adopts awaits sign-off (ROADMAP T0.1d).
- Open, not yet evidence: a_ℓm `[60,64)` sits at +2.3 to +2.9σ against the null in all four ensembles, but they share the same 24 unlensed skies.

### Figures and statistics reworked (2026-09-22)

- **`figure_maps` was wrong.** Author-ordered alms were passed to healpy, producing zonal stripes and a meaningless r = 0.815. It now routes through `lensing._alm_packed_to_hp` and has six panels, including the noise-free lensing signal and a residual normalised per mode. Its stat box is the per-mode sd, because the pixel sd of one sky is a statistic of ~10 low-L modes.
- **`figure_convergence` had a fabricated burn-in line** (chains save post-burn-in). Rebuilt: rank-normalised folded split-R̂ vs 1.01 (per chain, since each chain is a different sky), and bulk ESS for all four blocks against the rank-test draws.
- **`figure_schematic`** had a stray −1 in Block 4's shape, called Block 2 MCLMC, and used d for both deflection and data. The formulas are now pinned to the code by a test.
- **Figure 1's statistics were invalid.**
  - The pooled N=96 ranks are 24 chains × 4 correlated bins, so pooled tests are over-confident (at thin=60 they "rejected" a_ℓm spread at p = 0.003).
  - The pooled χ² `cal_p` has ~1.6 counts per cell and no power.
  - The C_L^φφ panel still printed the retired KS test.

  Figure 1 now reports `p_bin`: the Bonferroni-corrected minimum of per-bin rank-mean and rank-spread p-values, each with a simulated null over 24 independent chains. On the legacy ensemble: φ 0.41, a_ℓm 1.00, C_L^φφ 0.16.
- **Figure 2 redesigned** as same-sky blind-vs-aware at lmax = 64, with *the lensing each sky actually received* (exact operator, noise-free) as the expected blind bias. **Figure 3** reads the noise level from the chains and refuses a QE validation whose (lmax, nside, σ, A_φ, fiducial, operator) differ from the chains'.
- **`paper_style.py`** has journal presets (PRD default: 3.375/7.0 in, Computer Modern) and constrained layout with standard bbox, so every panel is saved at exactly its printed size. **`make figures`** rebuilds all seven from one command, pointed at the exact-operator ensemble.

### Test suite built out: 163 → 200+ tests

- `tests/test_lensing_exact.py` (13 tests):
  - φ = 0 identity; brute-force Y_ℓm sum at independently (Rodrigues) remapped points;
  - geodesic distance = |d|; deflection derivatives vs FD; sign conventions vs `hp.alm2map_der1`;
  - FD gradients of `psi_lensed` in alm and φ and of `log_prob_phi_block`;
  - `tf.function` tracing; bilinear → exact convergence; wiring.
- `tests/test_paper_figures.py`:
  - page sizes, single-mode map placement, `mode_z`;
  - R̂ and ESS against AR(1) theory;
  - rank-test calibration and power;
  - figure 1 end to end passing a correct and rejecting a biased synthetic sampler;
  - figure 4 and convergence end to end, schematic formulas, the power-transfer operator check.
- `tests/test_scripts_pipeline.py`:
  - `coverage_ensemble_chain` aware and blind runs end to end (same sky, metadata, amplitude applied to the prior);
  - replay through the chain's own operator;
  - rank helpers, fiducial sanity, fp64 prior, beam/prior ordering, the Block 4 PIT with a failing misaligned control;
  - `validate_qe_noise` run; figures 2, 3 and maps end to end.
- `tests/test_qe.py`: QE sign test on exact-lensed skies (response 0.85–1.15).
- Mutation checks were run by reintroducing the alm-ordering bug, the Block 4 −1, the tight bbox, the beamed prior and the old QE sign; each is caught.

**Lessons (these are now standing discipline in `ROADMAP.md`):**
- validate a forward operator on the observable the paper reports, not only its gradients;
- estimate the physical information content before interpreting a posterior;
- pin sign conventions against an independent reference, because two opposite sign errors pass every internal consistency test;
- test the scripts, not only the package.

---

## Legacy results (bilinear operator) — superseded, kept for method and lessons

### Sampler exactness bounds (legacy model)

Stated as bounds. At N=24 and 60 draws per chain, detection power is **50 % at a 0.42 σ** posterior-mean shift.

| ensemble | jobs | Block 4 | certifies | result (thin=10) |
|---|---|---|---|---|
| Block-4-OFF | 11955622 + 11965828 | OFF (C_L^φφ pinned) | (a_ℓm, C_ℓ, φ) | φ 0.4585, alm 0.4956, no bin flagged; joint-likelihood SBC 0.4587 |
| Block-4-ON, ν = 6 | 11965813 + 11980637 | ON, proper prior | full (a_ℓm, C_ℓ, φ, C_L^φφ) | strict C_L^φφ rank 0.4518, per-bin p 0.58/0.39/0.71/0.14; Block 4 PIT pass with lag-10/50 controls rejected |

**Name the configuration:** "headline exactness" meant the Block-4-OFF ensemble for two weeks while the title claimed four blocks. The exact-operator re-run keeps both ensembles and cites Block-4-ON for the title's object.

### The missing `Im(a_{L,1})` degree of freedom — found, restored, re-validated (2026-08-31 → 09-08)

`splittosingularalm` forced m = 1 real as well as m = 0, so the packing carried 2L dof, not 2L+1. The fix made `alm_utils.packed_sizes`/`packed_length` the single definition, with `packed_dof_per_multipole`/`invgamma_shape_for_spectrum` deriving both spectrum blocks' shapes. Checkpoints are versioned (`PACKING_VERSION=2`). Two missed sites were caught by tests (`hpalminit`; the amplitude-rescale Jacobian). `test_general_synalm_draw_survives_pack_unpack_with_no_power_loss` is the test that catches the original defect (a round trip cannot, since the restriction is idempotent).

### Rank-test recalibration (2026-09-12/13)

- **The continuous KS test over-rejects discrete ranks** (30.3 % / 12.5 % at nominal 5 % / 1 % for pooled N=96 with 8 draws). It is cured by more draws per chain, not more chains.
- **A sub-0.5 mean means a posterior sitting high; under-dispersion shows only in `sd_u`.**
- **A pass is a bound with stated power.** At 60 draws per chain the strict C_L^φφ rank went 0.0009 → 0.0567 with the mean unchanged, i.e. a measurement artefact.
- **Three "ℓ-localised defects" dissolved under more realizations or a calibrated test.** The Block-4-ON `[10,30)` "residual" (two weeks of Hessian-coupling and Nyström work) was small-N plus the miscalibrated test. **Do not reopen without new evidence.**

### Spectrum comparisons: reference the correct posterior mean

Under the flat C_ℓ prior, Block 1's posterior mean is S_ℓ/(k_ℓ−4), not S_ℓ/k_ℓ (+15 % at ℓ ~ 20). Against the realized power (or the fiducial) a correct sampler looks biased, and the comparison once named the lensing-aware chain the more biased one. The new figure 2 uses S_ℓ/(k_ℓ−4).

### Legacy figure 3 and the QE machinery (2026-09-18)

- **`diffcmb/qe.py`:** curved-sky TT N_L from the Hu–Okamoto coupling, with 3j symbols checked against sympy.
- **`validate_qe_noise.py`:** validates N_L against the estimator's own definition (response on lensed skies plus noise on unlensed skies, which must pass together).
- **Two traps recorded:**
  - N_L is *not* the fixed-alm Fisher (`estimate_phi_diag_fisher`); that comparison was off by 132× because it conditions on a known CMB.
  - Compare against QE⊕prior, never raw N_L.
- The legacy result (posterior width 0.953 ± 0.010 of QE⊕prior at 10 ≤ L < 20) is retracted with the operator.

### lmax scale-up (legacy)

The low-L φ mode fails equilibration at lmax = 128 (lag-1 0.967, job 11966631) and 192 (0.976, job 11987444). Exactness was never available above lmax = 64. Do not spend more φ-equilibration effort there without sign-off.

---

## Real bugs found and fixed

- **2026-09-22/23:** the operator, deflection sign, QE sign, fiducial, fp32 prior and beam/prior bugs (table above); the `figure_maps` alm ordering; the fabricated convergence burn-in; the schematic's Block 4 −1; pooled rank tests assuming independence. Also, an edit placed chain metadata into the `run_gibbs_chain(...)` call instead of `np.savez(...)` in `run_lensing_blind_baseline.py`; it was caught before any run, and the script smoke tests now cover this class.
- **Stale lensing-blind baseline was a different model (2026-09-11):** `alm_true_packed` was 16254 long against `packed_length(128)` = 16380. `PACKING_VERSION` guards checkpoints, not analysis products; check widths before combining.
- **alm ordering never converted (2026-08-24):** `_alm_packed_to_hp`/`_alm_hp_to_packed` skipped `almmotho`/`almhotmo`; every φ coefficient sat at the wrong multipole. A round-trip test cannot catch a consistent permutation; pin an absolute (L, m).
- **Blocks 1 and 4 assumed 2L+1 dof when the packing carried 2L (2026-08-31):** a PIT against the sampler's own conditional cannot catch a derivation error; only an independent generative SBC (proper prior) did. Two analysis scripts mirrored the same wrong constant.
- **Ensemble cold-started alm (2026-08-28, job 11887897):** φ inflated 10³–10⁵× and froze, passing a mixing gate. A gate must run the production initialisation path, and mixing diagnostics measure movement, not correctness.
- **`run_gibbs_chain(seed=...)` never seeded TF (2026-08-31):** HMC momenta came from the global stream.
- **Return-tuple arity depends on the enabled blocks**; a hardcoded unpack crashed after a completed chain with SLURM reporting `COMPLETED 0:0`.
- **Coverage statistic ranks the truth against its conditional's mode**, which is non-uniform for a correct sampler; read it against `validate_coverage_rank_nulls.py`.
- **Block 4 PIT control passing vacuously at lag 1**; controls are now required to fail at lags 1, 10 and 50.
- **Rank grid off by one in three scripts (2026-09-25):** `fig1_validation.py`, `diagnose_calibration_stationarity.py` and `null_alm_power_rank_flat_prior.py` set `n_draws = M − 1` for M draws, although the rank of the truth runs over 0..M. So u = (r+0.5)/M reached (M+0.5)/M > 1. Figure 1's simulated uniform null never produced rank M. And figure 1 refused the production a_ℓm null the moment a truth ranked above every draw (job 12062385). The shift in mean_u is +0.5/M: +0.004 at 120 draws, +0.0014 at 360. That is immaterial to every verdict, but every null file was regenerated on the corrected grid (job 12062823). `aggregate_coverage_ranks.py` and `validate_coverage_rank_nulls.py` were already right. Tests: `test_rank_grid_counts_every_draw_and_accepts_the_top_rank`, `test_null_ranks_live_on_the_observed_grid`, and the corrected `test_trace_rank_equals_rank_of_power_against_truth`, which had encoded the bug.
- Smaller:
  - m>0 alm precision weight;
  - SHT m-weights;
  - `jit_compile` vs `tf.py_function`;
  - `IndexedSlices` cotangents;
  - a hardcoded `atol` faking ESS = 100 %;
  - stale MCLMC/lmax≈128 claims in `main.tex`;
  - a correlation number attached to the wrong claim.

## Validated foundations

- **Phase 0:** unlensed Gibbs baseline on real Planck data (lmax = 300, float64), converged and trusted.
- **Matrix-free ducc0 SHT** behind `tf.custom_gradient`, ~500× the dense path, full-sky and masked-sky validated.
- **HMC + matrix-free SHT alm sampler:** the production path, with MAP initialisation required.
- **Beam + pixel window** (`beam_fwhm_arcmin`) and **anisotropic noise** (`noise_map`), validated against independent truths.
- **Block 4 exact inverse-Gamma draw** with an optional proper conjugate prior (ν > 0 enforced). The prior's fiducial is snapshotted before the loop, and the SBC truth is drawn from the same joint prior.
- **Exact lensing operator (2026-09-23)**, validated as above.
- **φ-block memory leak** fixed (traced `bootstrap_results` + `one_step`).
- **Joint (C_ℓ, C_L^φφ) correlation machinery:** per-chain standardisation, chain bootstrap, within-chain permutation null. The legacy result was a null (0/16 cells).

## Closed-out sampler routes (do not revisit without new evidence)

- Diagonally-preconditioned CG on the masked sky: degrades ~10⁴ under a realistic mask.
- Messenger field: critical slowing down at n_alm ≈ 90k.
- Unlensed-operator exact Block-2 shortcut: 652 % C_ℓ bias.
- Nyström `block` φ mass matrix: loses to `prior` with Block 4 on; the rank-deficiency hypothesis was falsified.
- NUTS for φ: lag-1 0.985.
- MCLMC for φ: fails the stationarity gate twice.
- Lensing amplification above A_φ ≈ 3000 at lmax = 64: Gibbs stalls or inflates φ (pilots above).

## Positioning (settled)

- **The novelty is the joint (C_ℓ, C_L^φφ) posterior as a capability** no competing method produces (MUSE, QE, Commander, diffusion). The evidence is validation. Never lead with the differentiable machinery (Flinch).
- **The legacy S1 framing ("bias reduction leads as validation, 93.7 %") is void** with the retraction. The new figure 2 shows a blind fit absorbing exactly the lensing each sky received while the joint fit does not, under amplified lensing, stated as such. The "recovers A_L = 1 without a template" framing survives if the new figure supports it.
- **The narrow form of the blind-model claim still holds:** against map-based Gibbs methods (Commander does not model lensing), not against Planck's likelihood (lensed spectra plus an A_L nuisance).
- **Literature:** no curved-sky joint sampler and no curved-sky MUSE as of 2026-09-22 (`literature.md`). Citation IDs and author lists were verified 2026-09-22: `1708.06753` (Millea, Anderes & Wandelt 2019, flat-sky T+P joint sampler), `2111.07664` (Ducrocq et al.), `0708.2989`, `1803.03462`, `1705.01893`, and the Commander trio `astro-ph/0209560` / `0310080` / `0407028`.

## Engineering gotchas worth remembering

- **Eager `tf.py_function` in a hot loop leaks memory**; wrap in `@tf.function`, and pass changing values as `tf.Variable`.
- **`/cosma/apps/durham/dc-hick2` has its own NFS quota** (100 GB / 10M files, `quota -s`).
- **`/cosma5` is not mounted on dine2/cosma7/cosma8 compute nodes.** Large artifacts go to `/cosma8/data/dp004/dc-hick2/diffcmb_results_archive/`, symlinked into `results/`.
- **SLURM `COMPLETED`/exit 0 does not mean the script succeeded**; read `.err` and the script's own verdict line.
- **dine2 nodes need a job-private `$TMPDIR`** (autograph cache collisions).
- **Packing several TF tasks per dine2 node:** drop `--exclusive` and `--mem=0`, and give each task one GPU (`CUDA_VISIBLE_DEVICES = task % 4`) with `TF_FORCE_GPU_ALLOW_GROWTH=true`. By default TF reserves all four GPUs' memory. `--exclusive` with 8 cores per task turned a 16 h campaign into a ~2-day queue.
- **`set -u` in SLURM scripts:** write `${PYTHONPATH:-}`. `PYTHONPATH` is unset on compute nodes, and all 72 tasks failed in 1 s.
