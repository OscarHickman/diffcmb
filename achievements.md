# Achievements — DiffCMB

*Condensed record of what is validated, closed out, retracted or fixed — current state, not a change log. Full history is in git. Forward plan: `ROADMAP.md`.*

**Read first.** Everything produced before 2026-09-23 was made with the **legacy forward model**: bilinear-interpolation lensing on an nside = lmax grid, deflection by −∇φ, and a defective fiducial cosmology. Those results remain valid as statements about *the sampler inverting the model it was given*, and they are reproducible (`lensing_operator='bilinear'`, chains read their own spectra). They are **not** statements about lensing on the sky. Every paper number is re-derived from the exact-operator ensembles; the adopted one is `ens_exact_l64_A3000_n30_nu30_long` (D5; section "2026-09-25/27" below). The open defect is the a_ℓm band-edge offset (first section).

## Decisions taken

| | decision | date |
|---|---|---|
| D1 | one paper, all extensions included | 2026-09-22 |
| D1b | all figures final before any paper text | 2026-09-22 |
| D2 | journal PRD (the `paper_style.py` default) | 2026-09-22 |
| D3 | exact lensing operator | 2026-09-23 |
| D4 | amplified-lensing validation at A_φ = 3000, σ = 30 μK/pixel, lmax = 64, chosen by the pilots below | 2026-09-23 |
| D5 | the paper's Block-4-ON ensemble uses the proper hyperprior **ν = 30**, 1000 + 3600 sweeps (`ens_exact_l64_A3000_n30_nu30_long`), the only Block-4-ON configuration that passes every calibrated test. ν = 6 is reported as a limitation. Rejected alternatives: a non-centred φ parameterisation (sampler development, 2–3 days, needs φ-tuning sign-off) and ~25k-sweep ν = 6 chains (~100 h per task) | 2026-09-25 |

---

## 2026-09-27 → 10-03 — the a_ℓm `[60,64)` band-edge offset (T0.1, still open)

*Tables: `docs/dashboard.md` (the 2026-09-27 → 10-03 sections). Next steps: `ROADMAP.md` T0.1.*

### The offset is real, and it is ℓ = 63

- **It reproduces on fresh skies.** Against the effective exact-sampler null it is +2.55 on r000–r023, **+2.33 on the fresh r024–r047 alone**, and +3.45 combined (nominal null: +2.80). Every other a_ℓm bin is at |z| ≤ 0.6. By the pre-registered rule (survives more realizations), it is a defect to diagnose, not sky noise.
- **It sits at ℓ = 63, the band limit** (job 12066643). Per-ℓ z is +3.5 to +3.8 at thin 50/100/150 on the ν = 30 ensemble, +2 to +4 in every other exact ensemble, and ≤ +1.5 at ℓ = 60–62. m = 0 is clean; real parts carry more of it than imaginary parts.

### Band-edge diagnosis (2026-09-27): what is established

- **The diagonal a_ℓm null is validated against an exact dense reference.** `scripts/exact_dense_alm_reference.py` builds the full lensed operator at φ_true from `lens_map_tf` (reproduces it to 6e-15) and rebuilds the production data from the production seeds. It then samples the (C_ℓ, a_ℓm) posterior exactly: a Cholesky a | C draw, then Block 1's InvGamma C draw (job 12066700, 48 skies). Result: no a_ℓm rank offset at any top ℓ (ℓ = 63 z +0.3/+0.8), sd(z) 0.99–1.02 (1.006 at ℓ = 63), and a shrinkage slope equal to W (0.891 against 0.894). So pixelisation and the lensing geometry do not bias the statistic, and the null's diagonal model is adequate **when φ is known**.
- **In production, posterior C_63 sits ~6 % low and drifts by chain quarter** (0.934 → 0.947 of S_true/(k−4), against 0.99 in the dense reference). ℓ = 62 started low too and recovered (0.973 → 0.993).
- **The MAP start sets every C_ℓ low**, at about e^−1 × truth: `find_map_estimate`'s joint (ln C, a) MAP is the usual hierarchical-MAP shrinkage (replay job 12066670, `/cosma5/.../scratch_r036/`).
- **The slow direction is collective.** The single-coordinate τ_int(ln C_63) ≈ 42 does not show it.
- **A rank-mean quarter test missed the drift; C_ℓ / (S_true/(k−4)) by quarter caught it.** `diagnose_alm_band_edge.py` prints both.
- **Broader, milder:** per-mode sd(z) of a_ℓm is 1.03–1.08 across all ℓ at ν = 30 (1.18–1.25 in the 1200-sweep chains) against 1.00 exact. Posteriors are slightly too narrow, and less so in longer chains. The rank test cannot see this, because its effective null is calibrated from the chains' own variance. Quote sd(z), not only the rank.
- **Tools, all tested:**
  - `diagnose_alm_band_edge.py` (per-ℓ/m/part ranks against the null, thinning, quarters, per-mode sd(z), shrinkage slope, C by quarter; `tests/test_diagnose_alm_band_edge.py`, including a shrunk-sampler mutation);
  - `exact_dense_alm_reference.py` (`tests/test_exact_dense_alm_reference.py`: a | C mean and covariance, and C | a against the InvGamma median);
  - the ensemble wrapper takes `LMAX`/`NSIDE`.

### C_ℓ collapse (a separate defect, open)

- In ν = 30 r036, C_62 is ~2e-5 of the realized power for the whole chain (a_62 sd ≈ 3e-3 against 1.1 prior). ν = 6 long r021 collapsed the same way at ℓ = 62 and recovered only late. Block-4-OFF at N = 48 has one collapsed sky too (ℓ = 62 sd(z) 100.6), so it is not specific to Block 4.
- It is seeded by a low MAP start (r036's MAP put C_62 at e^−2, the lowest of any ℓ). Once C is tiny, HMC can hardly move the stiff a_ℓm.
- It was caught only by a per-mode z (|z| ≈ 700); the power ranks shift by only ~0.3σ.

### Band-edge localisation (2026-09-30): lmax 64 only, and not caused by Block 4

Jobs 12066584 (Block-4-OFF `_nocl4` extended to N = 48), 12066585 (lmax = nside = 32) and 12066667 (lmax 32, nside 64), all 48/48, run through `diagnose_alm_band_edge.py` by job 12074645 (log `logs/alm_band_edge_12074645.out`).

- **lmax 32 is clean at both pixel scales.** Top bin `[28,32)` z −0.5 (nside 32) / −0.3 (nside 64); ℓ = 31 z −1.2 / −0.3; C_31/(S_true/(k−4)) is flat at 0.99–1.04 across chain quarters. So the top-multipole lag is not a generic band-edge property of the sampler, nor of nside = lmax pixelisation.
- **Block-4-OFF shows the same offset:** `[60,64)` z +3.4 to +3.9 (thin 50–150), ℓ = 63 z +3.1 to +3.7, C_63 at 0.926–0.943 of S_true/(k−4) in every quarter. It is not caused by Block 4 or the ν prior.
- **Block-4-OFF figure 1 at N = 48, thin 50** (scratch, not the paper figure): φ p_bin **0.180** ✓ (0.190 at N = 24), a_ℓm min p_bin **0.000** ✗ from `[60,64)` (0.094 at N = 24), 50 % power at 0.314σ. The a_ℓm failure is the T0.1 defect, not something specific to Block-4-OFF; the caption number waits for T0.1.

### `--cl_init data` pilot (2026-10-03): the start is not the cause

**Hypothesis tested:** the joint-MAP start leaves C_ℓ further from equilibrium at lmax 64, so C_63 is still climbing at the end of the chain.

**What ran:** job 12078506, skies r032–r039, the production configuration (Block-4-ON, ν = 30, 1000 + 3600, lmax = nside = 64), with `--cl_init data`: ln C_ℓ starts at S_ℓ(a_MAP)/(2ℓ−3). It wrote to `ens_exact_l64_A3000_n30_nu30_clinit_data/`. All 8 tasks completed with `phi_calibration_ok` set. They ran CPU-only on cosma8-shm (18.8–19.0 h; the `cuInit` line in `.err` is harmless). The diagnostic was job 12090562 (log `logs/alm_band_edge_12090562.out`). The control is the same 8 skies of `_nu30_long`, symlinked as `ens_exact_l64_A3000_n30_nu30_long_r032_r039/`. The skies match: identical `alm_true_packed` and `cl_phiphi_true`. Only 8 skies, so every comparison below is paired by sky.

- **The data start removes the early transient but not the deficit.** C_63 / (S_true/(k−4)), the mean over the 8 skies by quarter:

  | | Q1 | Q2 | Q3 | Q4 |
  |---|---|---|---|---|
  | pilot | 0.946 | 0.944 | 0.921 | 0.935 |
  | control | 0.904 | 0.883 | 0.942 | 0.935 |
  | paired difference | +0.04 ± 0.02 | +0.06 ± 0.02 | −0.02 ± 0.03 | 0.00 ± 0.03 |

  The two starts converge on the same late level. Against the φ-fixed dense reference on the same skies (r036 excluded), Q4 C_63 is −0.07 ± 0.06 low in the pilot and −0.085 ± 0.05 in the control.
- **The rank offset shrinks by about a third, but it does not go away.** The `[60,64)` z against the effective null is +1.86 in the pilot and +2.82 in the control (thin 50; +1.44 vs +2.71 at thin 150). The paired Δu is −0.10 ± 0.04 at thin 50 and −0.13 ± 0.06 at thin 150. The ℓ = 63 sd(z) is 1.07 in the pilot and 1.09 in the control, against 0.97–0.99 exact. The shrinkage slope at ℓ = 63 is 0.656 in the pilot and 0.645 in the control, against W_eff 0.74.
- **It does not prevent the r036 collapse.** C_62 was already 1.2e-6 of the realized power at the first saved sweep, lower than the control's 2e-5, because S_ℓ(a_MAP) inherits the MAP's shrinkage of a_62. Unlike the control, it recovered by itself in Q3–Q4 (0 → 0.17 → 0.72), which is the same stochastic late recovery seen in ν = 6 r021.
- **Reading.** Neither criterion is met: C_63 is not flat near 1.0, and r036 still collapses. Since two different starts reach the same late C_63, a slow climb from the start does not explain the late-chain deficit. The candidates are a stationary effect of marginalising φ at the band edge, or mixing that is too slow for either start to reveal. The dense reference conditions on φ = truth, and the lensing effect is largest at the band edge (−41 % at `[45,64)`, figure 2). The discriminating test is in `ROADMAP.md` T0.1.

### Figure outputs from the diagnosis

- The figure 1 power table (`validation_power_table.{csv,tex}`, mutation-checked) and the zoomed aware-only figure 2 panel (`bias_aware_zoom.pdf`; job 12066719). The Block-4-OFF certification goes in figure 1's caption as a number, not a panel.

---

## 2026-09-25/27 — the adopted ensemble (ν = 30), extended to N = 48

### What was run

| step | job(s) | result |
|---|---|---|
| lensing-blind pair on the ν = 30 skies r000–r023 (figure 2) | 12062768 | 24/24 |
| nulls regenerated on the corrected rank grid | 12062823 | every z unchanged to 2 d.p. |
| `make figures` pointed at `_nu30_long` (Makefile `ENSEMBLE`) | 12062824 | all figures rebuilt |
| a_ℓm null at thin 50 (`--thin`, null file `_thin<N>`, Makefile `FIG1_THIN=50`) + figures | 12065179, 12065180 | figure 1 passes at N = 24 |
| 24 fresh skies r024–r047, Block-4-ON + blind pair | 12065119, 12065120 | 48/48, Block-4-ON tasks 14.0–15.1 h |
| N = 48 harvest (`scripts/submit_harvest_t01e.slurm`, with a completeness guard) | 12065737 | log `logs/harvest_t01e_12065737.out`; the N = 24 files are kept as `*_n24.npz` |

### N = 48 results (thin 10 unless stated; uniform mean_u = 0.5 ± 0.042)

- **Joint-likelihood SBC passes:** 0.528, KS_p 0.45 at thin 40; 0.512, KS_p 0.77 on the second half.
- **Strict C_L^φφ SBC passes:** pooled 0.495, KS_p 0.65. Per bin, KS_p is 0.85 / 1.00 / 0.22 / 0.064; the `[60,64)` low reading of N = 24 (0.397, KS_p 0.015) relaxed to 0.437.
- **Block 4 PIT passes:** aligned KS_p 0.22. The lag-10/50 controls are rejected (KS_p 0); lag 1 passes vacuously (φ lag-1 correlation +0.85).
- **Figure 1 (thin 50):** φ p_bin 0.386 ✓, C_L^φφ 0.501 ✓, **a_ℓm 0.005 ✗** (the band-edge offset above). 50 % power is at 0.281σ (0.436σ at N = 24).
- **φ power bias** per bin (median): 1.007 / 1.006 / 0.991 / 1.017.
- **Figure 2 (48 pairs):** mean |bias| 12.4 % blind vs 0.80 % aware; corr(blind bias, lensing received) in `[45,64)` is 0.892. Aware z against the exact-sampler expectation (nominal / effective): `[2,10)` −0.59/−0.83, `[10,20)` +1.99/+0.40, `[20,30)` +1.57/−0.23, `[30,45)` +1.47/−1.25, `[45,64)` +1.31/**−2.97**. All bins are inside the bracket except `[45,64)` (−3.04 at N = 24), which probably shares the band-edge cause.
- **Figure 3:** 0.606 / 0.621 / 0.663 / 0.721 of QE⊕prior at L ~ 15 / 25 / 38 / 55.
- **Figure 4:** 1/16 cells above null (0.8 expected by chance), 0/16 at thin 150, all |r| ≤ 0.054. The one cell, `C_ℓ[30,60)×C_L^φφ[10,30)` (ratio 1.03, chain z +2.34), is the same cell as at N = 24.
- **Convergence:** R̂ > 1.01 for 28.6 % / 20.1 % of (chain, multipole) pairs (Blocks 1/4). Median ESS per 3600 sweeps is 252 / 74 / 181 / 342 (Blocks 1–4).
- **Maps:** r 0.910, per-mode z sd 1.028 pooled over 48 skies.

### Figure work at N = 24 (2026-09-26) that carries over

- **Figure 2's reference is not zero.** Under the flat C_ℓ prior an *exact* sampler's posterior-mean C_ℓ exceeds S_true/(k−4), because noise inflates E[S|d]. This is the figure 1 a_ℓm offset seen from the other side. `scripts/expected_aware_bias_flat_prior.py` computes the expected bias for nominal and chain-calibrated effective noise, and saves `expected_aware_bias_flat_prior.npz`. `fig2_bias_reduction.py` shades that bracket when the file covers exactly the figure's skies, and otherwise skips it with a warning.
- **Figure 4's flagged cells were chance** (`scripts/inspect_fig4_cells.py`). A chain-level z, robust to autocorrelation, finds none significant over 16 cells, and none clear the null at thin 150.
- **Block-4-OFF caption number** (`_nocl4`, thin 50, N = 24, own null): p_bin φ 0.190 / a_ℓm 0.094.

---

## 2026-09-22/25 — operator retraction, physics fixes, the exact-operator re-run and its diagnosis

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

| mode | job | output | state (2026-10-03) |
|---|---|---|---|
| Block-4-ON, ν = 6 | 12038494 | `results/analysis/ens_exact_l64_A3000_n30/chain_rNNN.npz` | 24/24 complete |
| lensing-blind (same data) | 12038496 | same dir, `blind_rNNN.npz` | 24/24 complete |
| Block-4-OFF | 12038495 + 12066584 | `..._nocl4/chain_rNNN.npz` | 48/48 (400 + 1200 sweeps) |
| Block-4-ON, ν = 6, 1000 + 3600 | 12047785 | `..._n30_long/` | 24/24 (T0.1c2) |
| **Block-4-ON, ν = 30, 1000 + 3600 (adopted, D5)** | 12047786 + 12065119 | `..._n30_nu30_long/chain_rNNN.npz` | **48/48** (r000–r047) |
| lensing-blind on the ν = 30 skies | 12062768 + 12065120 | same dir, `blind_rNNN.npz` | 48/48 |
| lmax = nside = 32, ν = 30 (band-edge control) | 12066585 | `ens_exact_l32_A3000_n30_nu30_long/` | 48/48 |
| lmax 32, nside 64, ν = 30 (band-edge control) | 12066667 | `ens_exact_l32_A3000_n30_ns64_nu30_long/` | 48/48 |
| ν = 30 pilot, `--cl_init data`, skies r032–r039 | 12078506 | `..._n30_nu30_clinit_data/` | 8/8 (pilot, not production) |

The ν = 30 skies share the unlensed T with the ν = 6 skies but not φ_true, so a blind chain pairs only with its own ν.

**QE validated at the production configuration** (job 12038409, `results/analysis/qe_noise_validation_exact_A3000_n30.npz`): noise ratio **0.995**, response **0.971** (1.09 / 1.00 / 0.94 / 0.93 by bin). The QE weights and filter now use the lensed spectrum *measured* from exact-operator simulations. That, together with the sign fix, resolves the old 0.883 response.

### First harvest (2026-09-23, 400 + 1200 sweeps, ν = 6): physics clean, calibration failed

Sources: `make figures` (`logs/make_figures_exact.log`) and harvest job 12040798 (`logs/harvest_exact_l64_12040798.out`). All 72 tasks completed, and `phi_calibration_ok` was 24/24.

- **Physics clean (figure 2):** on 24 same-sky pairs the blind fit landed on the lensing each sky received: −45.6 vs −46.0 % at `[45,64)`, −15.7 vs −16.0 % at `[30,45)`. Mean |bias| was 13.9 % blind vs 1.37 % aware.
- **Calibration failed on Block-4-ON:**
  - the joint-likelihood SBC rejected (0.716, KS_p 0.001);
  - the φ field ranks failed (p_bin 0.004);
  - the a_ℓm ranks failed (`[30,60)`, p_bin < 0.001).
- **Block-4-OFF:** the joint likelihood and φ passed; a_ℓm `[30,60)` failed, the same bin as Block-4-ON.
- **Mixing:** median bulk ESS per 1200 sweeps was 123 / **35** / 65 / 80 (Blocks 1–4), below the rank-test draws.

That set up the T0.1 diagnosis below. Every number here is superseded by the adopted ensemble above.

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

  Adopted as D5. At N = 48 it still passes all of these except a_ℓm `[60,64)` (section above).
- a_ℓm `[60,64)` sat at +2.3 to +2.9σ against the null in all four ensembles, which share the same 24 unlensed skies. The fresh skies reproduced it (section above).

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
- `tests/test_qe.py`: QE sign test on exact-lensed skies (response 0.85–1.15); N_L's absolute normalisation frozen at lmax = 16 (`test_nl_absolute_normalisation_is_frozen_at_lmax16`), from an independent sympy-3j loop that matches `qe_tt_noise_nl` to 4e-16 (2026-09-26).
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
- **`--cl_init data` wrote into a read-only MAP vector (2026-09-30, job 12074650):** `np.asarray` on `find_map_estimate`'s result returned a read-only view of a TF tensor, and all 8 pilot tasks died 37 s in with `ValueError: assignment destination is read-only`. The helper `data_driven_ln_cl` had unit tests; the assignment in `main()` had none. Fixed with `np.array` (a copy); `test_cl_init_data_runs_end_to_end_and_is_recorded` runs the script with the flag and was confirmed to fail before the fix. Chains now record `cl_init` (absent = `map`). Resubmitted as job 12078506.
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
- **SLURM wrappers and `/cosma5` (2026-09-24/25).** `results/analysis` is a symlink into `/cosma5` (commit `115e6da`), which dine2 does not mount. All 48 tasks of the first c2/c3 attempt (12041916/7) died at their first checkpoint while `sacct` said COMPLETED. Fixes:
  - the 14 CPU-only wrappers run on the `/cosma5`-mounting partitions (cosma5, cosma8-shm/shm2/shm3, cosma8-ska, bluefield1), with `--mem` sized from past MaxRSS;
  - the 57 GPU wrappers stay on dine2 and now fail at start if `results/analysis` is not mounted;
  - every wrapper exits with Python's status.
- **Thread caps (2026-09-26):** the ensemble wrapper caps the OMP/OPENBLAS/MKL/DUCC0/TF pools. Spot-check: 52 threads per process, ~145 % CPU against 800 % allocated.
