# Achievements — DiffCMB

*Condensed record of what is validated, closed out, retracted or fixed — current state, not a change log. Full history is in git. Forward plan: `ROADMAP.md`. Live numbers: `docs/dashboard.md`.*

**Read first.** Everything produced before 2026-09-23 was made with the **legacy forward model**: bilinear-interpolation lensing on an nside = lmax grid, deflection by −∇φ, and a defective fiducial cosmology. Those results remain valid as statements about *the sampler inverting the model it was given*, and they are reproducible (`lensing_operator='bilinear'`, chains read their own spectra). They are **not** statements about lensing on the sky. Every paper number is re-derived from the exact-operator ensembles: `ens_exact_l64_A3000_n30_nu30_long` (D5) for figures 2–4, maps and convergence, and `ens_exact_l64_A3000_n30_nu30_strict` (D7) for figure 1. T0.1, the a_ℓm band-edge offset, is closed by D7 (section "T0.1" below). The φ `[30,60)` offset that remains is a stated limitation.

## Decisions taken

| | decision | date |
|---|---|---|
| D1 | one paper, all extensions included | 2026-09-22 |
| D1b | all figures final before any paper text | 2026-09-22 |
| D2 | journal PRD (the `paper_style.py` default) | 2026-09-22 |
| D3 | exact lensing operator | 2026-09-23 |
| D4 | amplified-lensing validation at A_φ = 3000, σ = 30 μK/pixel, lmax = 64, chosen by the pilots below | 2026-09-23 |
| D5 | the paper's Block-4-ON ensemble uses the proper hyperprior **ν = 30**, 1000 + 3600 sweeps (`ens_exact_l64_A3000_n30_nu30_long`), the only Block-4-ON configuration that passes every calibrated test. ν = 6 is reported as a limitation. Rejected alternatives: a non-centred φ parameterisation (sampler development, 2–3 days, needs φ-tuning sign-off) and ~25k-sweep ν = 6 chains (~100 h per task) | 2026-09-25 |
| D6 | T1.1: compare against a **strict** exact ensemble (proper ν = 30 priors on both spectra), **spectra-level NPE first**, **TFP in the existing venv** (rejected: importance-reweighting to the flat C_ℓ^TT prior; a field-level HEALPix generative model first; a new torch env). Scoping, and why `2603.04535` cannot be reused: `docs/notes/t1_1_learned_posterior_scoping.md` | 2026-10-07 |
| D7 | **figure 1 is the strict SBC of all four blocks on the strict ensemble** (`_nu30_strict`; proper ν = 30 on C_ℓ^TT and C_L^φφ, truths drawn from them), read against uniform with no simulated null. The φ-fixed run certifies the a_ℓm/C_ℓ blocks separately. The flat-prior figure 1 (`_nu30_long`, a_ℓm against its exact-sampler null) is kept as `figure1_flatprior` for the caption. Rejected: a φ-marginal reference for the flat-prior ensemble (no exact method at this size); the band edge as a bare limitation | 2026-10-14 |
| D8 | **r036 stays in `_nu30_long`**: its C_62 collapse (flat prior + MAP start) is stated in the caption. Dropping it moves figure 2 by ≤ 0.26 points per bin (reduction 93.6 % → 93.2 %), and a post-hoc exclusion would condition the ensemble on its outcome | 2026-10-14 |
| D9 | **T1.1 headline stays the baseline flow**: no width-gap route (regularisation, capacity, v2 summaries, compressor) beats it; the residual width is reported as the information cost of the summaries | 2026-10-15 |

---

## Ensembles on disk (identical skies across modes)

`scripts/submit_ensemble_exact_lmax64.slurm` (tasks packed per node, one GPU each, TF memory growth). All complete as of 2026-10-14:

| mode | job | output | tasks |
|---|---|---|---|
| Block-4-ON, ν = 6 | 12038494 | `results/analysis/ens_exact_l64_A3000_n30/chain_rNNN.npz` | 24 |
| lensing-blind (same data) | 12038496 | same dir, `blind_rNNN.npz` | 24 |
| Block-4-OFF | 12038495 + 12066584 | `..._nocl4/chain_rNNN.npz` | 48 (400 + 1200 sweeps) |
| Block-4-ON, ν = 6, 1000 + 3600 | 12047785 | `..._n30_long/` | 24 (T0.1c2) |
| **Block-4-ON, ν = 30, 1000 + 3600 (adopted, D5)** | 12047786 + 12065119 | `..._n30_nu30_long/chain_rNNN.npz` | **48** (r000–r047) |
| lensing-blind on the ν = 30 skies | 12062768 + 12065120 | same dir, `blind_rNNN.npz` | 48 |
| lmax = nside = 32, ν = 30 (band-edge control) | 12066585 | `ens_exact_l32_A3000_n30_nu30_long/` | 48 |
| lmax 32, nside 64, ν = 30 (band-edge control) | 12066667 | `ens_exact_l32_A3000_n30_ns64_nu30_long/` | 48 |
| ν = 30 pilot, `--cl_init data`, skies r032–r039 | 12078506 | `..._n30_nu30_clinit_data/` | 8 (pilot, not production) |
| ν = 30 skies, φ fixed at truth (Blocks 1 + 2), r032–r039 | 12099339 | `..._n30_nu30_phifixed/` | 8 (T0.1 a) |
| ν = 30 skies, truth start (`--start_truth`), r032–r039 | 12100312 | `..._n30_nu30_truthstart/` | 8 (T0.1 b; diagnostic, never an SBC ensemble) |
| **strict: proper ν = 30 on C_ℓ^TT and C_L^φφ, truths drawn from them** | 12100331 | `..._n30_nu30_strict/` | **48** (T1.1 reference, T0.1 c) |
| strict, on the real observing geometry (Planck common mask, 5′ beam + pixwin, ecliptic noise surrogate; `submit_ensemble_realsky_lmax64.slurm`) | 12172310 (pilot r000–r003) + 12183261 (r004–r047, queued 2026-10-17) | `ens_realsky_l64_A3000_n30_nu30_strict/` | 4 of 48 (T1.2a; same truths as `_nu30_strict`) |

The ν = 30 skies share the unlensed T with the ν = 6 skies but not φ_true, so a blind chain pairs only with its own ν. **The strict skies share φ_true and C_L^φφ_true with `_nu30_long`** (same realization index), and their a_ℓm are the same Gaussian draws rescaled by the drawn C_ℓ^TT (correlation 0.99; checked 2026-10-14). So the two ensembles are paired, not independent.

---

## 2026-10-20 — T1.2a real data: at A_φ = 1 one start's C_L^φφ runs away (sampler, not data)

- **SMICA A_φ = 1, 4 chains (12187757, 1000 + 3600):** C_ℓ^TT is clean (R̂ ≤ 1.001 in every bin; Planck 2018 / posterior median 0.97, 1.00, 1.03 in `[10,30)`, `[30,60)`, `[60,64)`; 0.83 at ℓ < 10, the known ν = 30 prior dominance; 1.16 at ℓ = 63, ≈ 1 sd). **C_L^φφ is not:** chains 1–3 sit at the prior centre (median ln(C/C₀) 0.01–0.08 in every bin), but **chain 0 drifts away during sampling**: in `[30,60)` ln(C/C₀) goes 0.07 → 0.35 → 1.1 → 2.0 (samples 0, 100, 400, 800) and plateaus at ≈ 3 (20× the prior), `[60,64)` at 3.9. 4-chain R̂ on C_L^φφ is 1.56–1.67 for L ≥ 10.
- **The same start (`--realization 0`) does the same on SEVEM 2018** (12188089_0: 0.95, 2.81, 4.01, 3.74 in the four L ≥ 10 bins); PR4 SEVEM chain 0 does not (0.03–0.04). The trajectory is start- and seed-driven, not a feature of one map.
- **It is not a mode the data support.** Chain 0's data log-likelihood is the *lowest* of the four (−18652 vs −18643 to −18648, 12 thinned states each); its φ buys 34 in log-likelihood at fixed a_ℓm against ≈ 5 in the other chains, while the ν = 30 InvGamma prior costs ≈ 30 nats per L at ln(C/C₀) = 3, ~10³ over the L ≥ 30 range. The A_φ = 3000 runs show the data constrain C_L^φφ only at ≈ 2000× physical (all 4 chains at 0.69–0.71 of the prior centre in `[30,60)`, agreeing). A π-invariant kernel started in the typical set does not walk 60 prior sd out for no likelihood: this is a sampler failure at A_φ = 1, the configuration that was never certified (SBC is at A_φ = 3000). φ acceptance (0.44–0.63) and the Block-4 conditional (exact conjugate draw) look normal; no rescaling move is on.
- **Test (no tuning, ROADMAP sign-off rule):** the certified real-sky ensemble at A_φ = 1 on simulated skies, 8 skies, 1000 + 3600 (**12189451**, `ens_realsky_l64_A1_n30_nu30_strict/`). A runaway there from a prior-drawn truth confirms the sampler; none points at the real data.
- **PR4 SEVEM A_φ 1, all 4 chains (12187764):** no runaway, but slow C_L^φφ mixing: chain medians in `[30,60)` ln(C/C₀) 0.005–0.087 against a posterior sd ≈ 0.06 (R̂ 1.24, bulk ESS 435); chain 3 creeps upward (0.07 → 0.13 by sixths), possibly the same drift at a slower rate. C_ℓ^TT R̂ ≤ 1.001. PR4 SEVEM / SMICA posterior D_ℓ: ≤ 0.7 % for ℓ < 60, 1.07 in `[60,64)` (0.97 sd), 1.10 at ℓ = 63 (0.67 sd): the method systematic sits at the band edge, as the map budget predicted.
- **SMICA A_φ 3000, all 4 chains (12187756, summary 12187765):** C_L^φφ 0.70 (`[30,60)`, 68 % 0.673–0.725), 0.73 (`[10,30)`), 0.87 (`[2,10)`) of the prior centre, i.e. ≈ 2100× physical. **But the chains sit at different levels:** per-chain `[30,60)` means −0.345, −0.366, −0.346, −0.381 in ln, each flat to ≈ 0.01 across its sixths with a per-sample sd of 0.034, so R̂ 1.117 and an offset of up to 1.3 posterior sd that 3600 sweeps do not relax. A slow C_L^φφ direction the within-chain ESS (73–457) does not see, plausibly the parked low-L φ mixing mode; the pooled interval is the honest one until the extension (below) says whether it closes. **Its C_ℓ^TT is not A_φ 1's:** posterior D_ℓ in `[30,60)` is 1365 μK² against 1305 at A_φ 1 (+4.6 %, ≈ 1.6 sd), Planck/median 0.952 against 0.992–0.996. Lensing assumed 2000× stronger than the sky's moves the inferred unlensed spectrum, so the A_φ 3000 run is a C_L^φφ consistency test, not a C_ℓ^TT measurement.
- **2018 vs PR4 and SMICA vs SEVEM, posterior D_ℓ^TT at A_φ 1 (4 chains each; `logs/realdata_compare_20261020.log`):** SEVEM 2018 → PR4 changes the posterior median by ≤ 0.2 % below ℓ = 60 and ≤ 2 % at the band edge (≤ 0.24 sd), the null the map budget predicted. SMICA → SEVEM: ≤ 0.7 % below ℓ = 60 (≤ 0.18 sd), then 1.05–1.07 in `[60,64)` (0.7–1.0 sd) and 1.08–1.10 at ℓ = 63: the component-separation systematic is confined to the band edge and stays under 1 sd. SEVEM 2018 C_ℓ^TT: Planck/median 0.96, 0.99, 0.98 for ℓ ≥ 10; its C_L^φφ has chain 0's runaway too (R̂ 1.64). Summary 12188090 had failed on a false alarm (the chains' data differ by 5.7e-14 μK, an ulp of the noise map between Intel and AMD nodes); `realdata_summary.py` now compares the data to 1e-9 relative, as the likelihood replay does, and was rerun by hand.
- **Extensions (2026-10-20, no tuning):** the SMICA A_φ 3000 and A_φ 1 chains resumed from their final checkpoints to 7200 samples in copies (`…_clnu30_ext/`, **12189472** A_φ 3000, **12189473** A_φ 1, 16 h): do the A_φ 3000 offsets relax, and does A_φ 1 chain 0 return? (The resume re-seeds with the chain's original seed, as the r029 resume did.)
- `fig_realdata.py` now refuses a run whose R̂ on either spectrum exceeds 1.05, and `realdata_summary.py` prints a NOT CONVERGED line, so the pending figure job (12189426) will not plot this C_L^φφ band.

## 2026-10-19 — T1.2a: the real-sky production (N = 48) passes every calibration test

Harvest 12187701 (`logs/harvest_realsky_12187701.out`; 12183262 had stopped at its completeness guard because r029 timed out at 16 h on mad02, resumed from its checkpoint as 12187700). The strict configuration on the Planck common mask, 5′ beam + pixwin and the ecliptic noise surrogate, 48 skies:

- **Strict SBC passes.** p per bin, thin 20 / second half thin 10 / thin 50:
  - C_ℓ^TT: `[2,10)` 0.96 / 0.58 / 0.36, `[10,30)` 0.87 / 0.41 / 0.53, `[30,60)` 0.095 / 0.66 / 0.73, `[60,64)` 0.47 / 0.26 / 0.18, ℓ = 63 0.43 / 0.66 / 0.79;
  - C_L^φφ: `[2,10)` 0.87 / 0.87 / 0.85, `[10,30)` 0.99 / 0.41 / 0.029, `[30,60)` 0.22 / 0.47 / 0.42, `[60,64)` 0.17 / 0.22 / 0.82, ℓ = 63 0.051 / 0.96 / 0.025.
  - The two thin-50 values near 0.03 are both C_L^φφ, out of ten bins (Bonferroni 0.005), and neither reappears in the second half. No collapsed C_ℓ.
- **Joint-likelihood SBC passes:** mean_u 0.513, KS p 0.78 (thin 40).
- **Field ranks pass in every bin:** φ pooled cal_p 0.29 (`[30,60)` mean_u 0.581, cal_p 0.94), a_ℓm pooled 0.95 (min bin cal_p 0.090 at `[60,64)`).
- The ℓ < 10 interval-coverage rows flag (C_ℓ^TT mean_u 0.405, C_L^φφ 0.355), but they are labelled "NOT calibration", and the full-sky strict ensemble flags the same bins with nearly the same ranks (0.407, 0.348; re-run 2026-10-19). That is the coverage statistic under a proper prior, not the mask.
- **Paired with the full sky:** sd ratio 1.02–1.38 for C_ℓ^TT and 1.07–1.28 for C_L^φφ (f_sky^−1/2 = 1.15, more at the band edge). rms z 0.80–1.19 (full sky 0.82–1.20); drift ≤ 0.33 sd; ESS 2–3× lower at ℓ ≥ 60 and L < 10.
- **Verdict:** the joint sampler is calibrated on the real observing geometry. The real-data runs on SMICA (both A_φ) and PR4 SEVEM were submitted on the 47-sky preview (12187756/57/64), and this confirms that gate.

## 2026-10-17 — T1.2a: the real-sky pilot passes its gate

The strict configuration on the Planck common temperature mask (f_sky 0.751), a 5′ beam + pixel window and an ecliptic noise surrogate, 4 skies (job 12172310), compared sky by sky with `_nu30_strict` on identical truths. Posterior widths grow by 1.01–1.15 below ℓ = 60 (f_sky^−1/2 = 1.15) and 1.25–1.46 at the band edge; no more bias than the full sky (rms z ≤ 1.24, against ≤ 1.67 full-sky) and half-to-half drift ≤ 0.33 sd. The cost is mixing: ESS falls 2–4× at ℓ = 63 (C_ℓ) and L < 10 (C_L^φφ). Table: `docs/dashboard.md` → CURRENT. Production r004–r047 queued (12183261), harvest 12183262. The pipeline needed one fix first: `psi_lensed` gathered `unmasked_idx` a second time (identity on a full sky, out of range on a general mask).

## 2026-10-17 — T1.2a data preparation: band-limit SMICA in harmonic space

The simulation test draws band-limited truths, so its data hold no ℓ ≥ 64 power; the real sky does. `scripts/real_data_bandlimit_budget.py` (log `logs/realdata_bandlimit_budget_20261017.log`) measures, on SMICA R3.00 at nside 64 and the common mask, what the lmax = 64 model cannot explain (in-band signal rms 53.6 μK, σ = 30 μK):

| preparation | ℓ ≥ 64 residual rms | mask-region leak (data / CMB-only sim) | band-edge S/N per mode |
|---|---|---|---|
| `ud_grade` (model.py's real-data path) | **54.8 μK (1.83σ)** | — | 10.2 |
| Gaussian 5′, no cut | 59.6 μK | 6.0 / 5.6 μK | 11.2 |
| Gaussian 120′ | 16.8 μK | 5.2 / 4.7 μK | 4.6 |
| Gaussian 200′ | 4.8 μK | 4.7 / 4.2 μK | 1.0 |
| **harmonic cut at ℓ < 64 (any beam)** | **0** | as the Gaussian of the same beam | as the Gaussian |

- **`ud_grade` would have misfit the data by ~4× the noise variance.** Smoothing alone cannot fix this without erasing the band edge (S/N ≤ 1 by 200′).
- **The harmonic cut removes all of it.** What remains is masked-region content (2 ≤ ℓ < 64) spread into the observed pixels: 6.0 μK for SMICA against 5.6 μK for one pure-CMB realisation of its own spectrum. The CMB part is sky the model describes, so the non-CMB excess is ≈ √(6.0² − 5.6²) ≈ 2.2 μK, 0.07σ (one simulation, so a rough figure).
- **SMICA's noise in the band is ≤ 0.26 μK rms** (high-ℓ raw plateau ≤ 2 × 10⁻⁴ μK²), so the σ = 30 noise the chain adds is the noise model and its likelihood is exact; neither half-mission maps nor FFP10 are needed at this scale.
- Built and tested (`tests/test_real_sky_setup.py`, `test_data_map_replaces_the_simulated_sky_and_hides_the_truth`; both mutation-checked): `real_sky_setup.bandlimited_map` (drops ℓ < 2 and ℓ ≥ lmax), `read_data_map` (refuses a map made for another nside, band or beam), `scripts/prepare_real_data.py`, and `coverage_ensemble_chain.py --data_map` (needs `--mask_file`; writes `data_chain_rNNN` / `data_blind_rNNN` with NaN truths, so no harvest glob or simulated checkpoint meets a real-data chain; an independent review caught the blind save missing this before commit). Prepared: `results/analysis/realdata_prep/smica_l64_ns64_b5.fits` (rms 55.1 μK).

### 2018 vs PR4 and SMICA vs SEVEM at lmax = 64 (2026-10-17)

PR4 (NPIPE) has component-separated maps only from SEVEM and Commander; `COM_CMB_IQU-sevem_2048_R4.00.fits` and its 2018 counterpart `COM_CMB_IQU-sevem_2048_R3.01_full.fits` are now in `/cosma8/data/dp004/dc-hick2/Plank` (from the PLA `product-action?MAP.MAP_ID=` endpoint), with the Planck 2018 TT spectrum in `Plank/spectra/`. All three maps band-limited by `prepare_real_data.py --map` (`results/analysis/realdata_prep/{smica,sevemR3,sevemR4}_l64_ns64_b5.fits`), then compared on the common mask at nside 64. Pseudo-C_ℓ ratio per bin, against the posterior's fractional width (cosmic variance at f_sky 0.75):

| bin | SMICA / SEVEM 2018 | SEVEM 2018 / PR4 | posterior sd of C |
|---|---|---|---|
| [2,10) | 1.021 | 0.990 | 0.167 |
| [10,30) | 0.994 | 0.997 | 0.058 |
| [30,60) | 0.999 | 0.998 | 0.031 |
| [60,64) | 0.963 | 0.988 | 0.073 |
| ℓ = 63 | 0.931 | 0.983 | 0.145 |

- **2018 and PR4 agree to ≤ 1.7 % in every bin (≤ 0.15 posterior sd); the map difference is 1.1 μK rms on the observed sky.** At lmax = 64 a 2018-vs-PR4 posterior comparison is a null by construction; it can only be a reproducibility check.
- **The component-separation method matters more:** SMICA vs SEVEM differ by 5.4 μK rms on the observed sky (≤ 2 % in power in the bulk, 4–7 % at ℓ ≥ 60, ~0.5 sd). Most of it is in-mask content (the two differ by 50 μK rms inside the mask) leaking through the ℓ < 64 cut. The leak is global, not at the mask edge: growing the mask by 1–6 rings leaves the SMICA–SEVEM leak difference at 3.7 μK while f_sky falls from 0.75 to 0.07. A SEVEM run alongside SMICA measures this systematic.
- The real-data MAP objective on SMICA (smoke job 12184339) is 21 583, inside the 18 800–30 600 range of the simulated real-sky skies.

### Real-data smoke runs on SMICA (2026-10-17/18; one chain each, short, not science)

- **Flat C_ℓ^TT prior collapses the band edge on real data** (12184339, 20 + 40 sweeps, A_φ 3000): `find_map_estimate` leaves ln C_ℓ at its −12 clip for ℓ ≥ 60 (e^−12.9 below the fiducial), and ℓ = 63 never left it (D_63 ≈ 0 against Planck's ~1700 μK²); ℓ = 54 collapsed too. The certified real-sky pilot starts from the same clip and recovers under its ν = 30 prior. `submit_realdata_lmax64.slurm` now defaults to CLNU = 30, and `realdata_summary.py` names any ℓ whose median is below e^−3 × the fiducial.
- **ν = 30 on both spectra, 100 + 200 sweeps** (12184369 A_φ 3000, 12184370 A_φ 1; 1.0–1.1 h): no collapsed ℓ; Planck 2018 D_ℓ / posterior median 0.95–0.99 in `[10,30)` and `[30,60)`, 0.96–1.04 in `[60,64)`, 1.08–1.16 at ℓ = 63.
- **At ℓ < 10 the posterior sits 19 % above Planck (ratio 0.84 in both runs).** The data carry f_sky(2ℓ+1) ≈ 4–14 modes per ℓ there against the prior's ν = 30, so the prior holds 70–90 % of the weight and pulls the known low-ℓ power deficit toward ΛCDM. A real-data C_ℓ^TT figure at ν = 30 must say the low ℓ are prior-dominated.
- **C_L^φφ:** at A_φ = 3000 the posterior median is 0.68–0.87 of the prior centre (e.g. `[30,60)` 0.683, 68 % 0.659–0.710), already several posterior sd below the amplified hypothesis after 300 sweeps; at A_φ = 1 it is 1.0 ± 0.1 of the prior centre (the prior returned, as the S/N of 0.002 predicts).

## 2026-10-15 — T1.1 width gap: no route closes it (D9)

Seven variants scored on the 48 strict skies (`logs/npe_gap_12164772_*`, `_12164773_*`, `_12170564_*`; route c rescored 2026-10-15 on the correct summaries). Held-out SBC passes for every variant except the compressor. Width ratio, learned/exact, C_ℓ^TT [30,60) and C_L^φφ [30,60):

| variant | C_ℓ^TT [30,60) | C_L^φφ [30,60) | held-out SBC |
|---|---|---|---|
| baseline `maf5x256` (v1) | 1.24 | 1.55 | pass |
| wd 1e-4 / wd 1e-3 (c) | 1.25 / 1.27 | 1.54 / 1.52 | pass |
| 5x128 / 3x128 wd 1e-4 (c) | 1.25 / 1.24 | 1.55 / 1.51 | pass (3x128 C_L^φφ [30,60) p 0.025) |
| v2 summaries, wd 0 / 1e-4 (a) | 1.26 / 1.26 | 1.57 / 1.56 | pass / **fails** (C_L^φφ [60,64) p 0.013) |
| learned compressor (b) | 1.80 | 1.67 | **fails** (C_ℓ^TT [30,60) p < 1e-4) |

Every variant sits within ±0.03 of baseline in every pooled bin, and mean shifts stay within ~1–3 SE of zero. Regularisation, capacity, richer summaries and a learned compressor do not move the width. The compressor is worse and mis-calibrated (its NLL diverged at epoch 2). **D9: the baseline `flow_maf5x256` stays the T1.1 headline; the residual 3–55 % is reported as the information cost of the spectra-level summaries.** 

**Learning curve (job 12171988, 2026-10-16): flat, so the gap is summary-limited, not data-limited.** Same flow trained on 50k / 100k / 200k sims, width ratio C_ℓ^TT [30,60) 1.26 / 1.24 / 1.24 and C_L^φφ [30,60) 1.55 / 1.55 / 1.55 (C_L^φφ [10,30) 1.33 / 1.34 / 1.33). Mean shifts stay within ~3 SE of zero at every size, and held-out SBC passes at every size (lowest p 0.023, C_ℓ^TT [60,64) at 50k). More simulations will not close the gap.

**C2ST (job 12173707, 2026-10-16; `scripts/npe_c2st.py`, `c2st_strict.npz`).** Per sky, a cross-validated ridge-quadratic classifier separates learned from exact draws in the 6 pooled bin-averaged coordinates (band-edge bins excluded), with one exact draw per autocorrelation time (about ESS draws per class, 134–450 per sky) and the same number of learned draws. Mean accuracy over the 48 strict skies (0.5 = indistinguishable):

| test | accuracy |
|---|---|
| raw (width and shift count) | 0.765 (range 0.66–0.89) |
| shape (each class standardised by its own mean and sd) | 0.472 |
| exact vs exact, raw / standardised | 0.501 / 0.437 |

The learned posterior is cleanly separable from the exact one by its width, as the width ratios say. With width and shift removed it is not separable by correlations: 0.472 is below chance, and +0.034 ± 0.008 above the standardised null, which is itself biased low (0.437: standardising the pooled sample anticorrelates the folds). Read it as a small residual at most, not a detected correlation mismatch. The quadratic features test means, widths and correlations, not higher moments, and this is a per-sky, 6-coordinate test, not a full-space one (the chains' ESS does not support more). The φ-mode correlations are not scored: the NPE is spectra-level and has no φ field.

## 2026-10-07/08 — T1.1: the learned posterior (spectra-level NPE) is calibrated, unbiased against the exact posterior, and wider

Decisions: D6.

**Pipeline** (all tested):
- `npe_simulate.py`: the strict-mode generative process of `coverage_ensemble_chain.py`, bit for bit (the test compares truths against a strict chain). θ = (ln C_ℓ^TT, ln C_L^φφ), ℓ = 2..63; x = (ln data power, ln normalised-QE power), using the QE filter validated at this configuration. 0.05 s/sim.
- `npe_flow.py`: TFP conditional MAF, checked first against an exact linear-Gaussian posterior (`tests/test_npe_flow.py`).
- `npe_train.py`, `npe_sbc.py`, `npe_vs_exact.py`: training, held-out SBC, sky-by-sky comparison against the strict chains. Each refuses inputs from another configuration.
- `strict_sbc_ranks.py`: strict SBC of both spectra against their drawn truths, plus the C_ℓ collapse table (T0.1 c).

**Bugs caught while building it** (each now has a test):
- TFP caches `inverse()` results by input-tensor identity. Reusing one validation tensor made the recorded validation loss stale, so early stopping restored the wrong weights. Fix: fresh tensors each epoch.
- Glorot initialisation gave a first-epoch NLL of ~100 (identity ≈ 4) on the toy problem. A near-identity init plus a soft-clamped log-scale reach the exact NLL in 5 epochs.
- `cl_phiphi_samples` are stored as ln C_L^φφ; taking another log gave NaN.

**Training** (jobs 12100390 sims, 12100400 train + SBC; 190 000 sims, MAF 5 × 256): best validation NLL 162.65 at epoch 12, stopped at 27. It overfits early (train 161.2 against val 163.2 by the end), which is a lever to pull.

**Held-out SBC passes** (`flow_maf5x256/sbc_heldout.npz`; 2000 sims × 199 draws; uniform mean 99.5, spread 0.2887):

| spectrum | bin | mean rank | spread | p |
|---|---|---|---|---|
| C_ℓ^TT | [2,10) | 98.87 | 0.2882 | 0.71 |
| C_ℓ^TT | [10,30) | 99.10 | 0.2831 | 0.26 |
| C_ℓ^TT | [30,60) | 100.07 | 0.2832 | 0.70 |
| C_ℓ^TT | [60,64) | 100.56 | 0.2889 | 0.51 |
| C_L^φφ | [2,10) | 98.47 | 0.2851 | 0.32 |
| C_L^φφ | [10,30) | 98.30 | 0.2835 | 0.53 |
| C_L^φφ | [30,60) | 98.89 | 0.2842 | 0.18 |
| C_L^φφ | [60,64) | 102.55 | 0.2942 | 0.15 |

**Learned vs exact on the 48 strict skies** (job 12100439, `flow_maf5x256/vs_exact_strict.npz`; 4000 learned draws per sky; shift in units of the exact posterior sd, SE = sd(shift)/√48):

| block | [2,10) | [10,30) | [30,60) | [60,64) (band edge, not pooled) |
|---|---|---|---|---|
| C_ℓ^TT shift ± SE | +0.02 ± 0.02 | +0.11 ± 0.06 | −0.02 ± 0.08 | +0.10 ± 0.15 |
| C_ℓ^TT width ratio | 1.03 | 1.12 | 1.24 | 1.59 |
| C_L^φφ shift ± SE | +0.03 ± 0.07 | +0.10 ± 0.13 | +0.12 ± 0.18 | −0.06 ± 0.16 |
| C_L^φφ width ratio | 1.16 | 1.33 | 1.55 | 1.52 |

Mean correlation matrix, learned − exact: max |diff| 0.106.

- **Reading:** the learned posterior is calibrated for what it conditions on (held-out SBC), **unbiased against the exact posterior** (no bin shifted significantly; the largest is z ≈ 2.0), and **wider**, by 3–24 % for C_ℓ^TT and 16–55 % for C_L^φφ, growing with ℓ. The width gap is the information the spectra-level summaries lose, as predicted from the weak QE–C_L^φφ tracking (per-L correlation ≈ 0.17 on 20 sims). It is not a calibration failure.

**Width-gap work (2026-10-14; all three routes queued, results in `docs/dashboard.md` → In flight):**
- (c) overfitting: `npe_train.py --weight_decay` (L2 on the MADE kernels, training objective only; every recorded loss stays the plain NLL), plus smaller flows.
- (a) richer summaries: `npe_simulate.py --summary_set v2` (372 numbers per sky against 124). The v1 entries come first and are bit-identical. v2 appends the ln QE power from low-ℓ legs [2,32) and from high-ℓ legs [32,64) (QE filters zeroed outside the band); the correlation of those two reconstructions per L; and the ln power of the data delensed by the Wiener-filtered QE φ (`delens_map`, the exact operator with −φ). 0.13 s/sim against 0.07.
  - The delensing sign is pinned on a noise-free lmax = 32 sky. The upper-band correlation with the unlensed sky is 0.94 lensed, 0.99 after delensing by the true φ, and 0.78 with the wrong sign; the test was mutation-checked by flipping the sign.
  - Power alone is the wrong check: remapping a band-limited field loses in-band power smoothly with ℓ.
- (b) learned map compressor: `npe_compress.py`. It whitens the data a_ℓm per ℓ and lays them on the (ℓ, m, re/im) grid, then applies a 5 × 5 convolution and a square (a learned local quadratic form), averages over m and takes the log, then dense layers that also take x. It is trained with MSE on standardised θ, and its estimate is appended to x (`load_conditioning`, used by `npe_train --compressor`, `npe_sbc` and `npe_vs_exact`).
  - Toy tests: it learns power (correlation > 0.9). It also learns a (ℓ, ℓ+1) coupling that no per-ℓ power can see (correlation > 0.8, against < 0.2 for power).

---

## 2026-09-27 → 10-14 — the a_ℓm `[60,64)` band-edge offset (T0.1): closed by D7

*Tables: `docs/dashboard.md` (the 2026-09-27 → 10-14 sections).*

**Outcome.**
- The production a_ℓm/C_ℓ blocks are exact at the band edge when φ is known (φ-fixed run).
- The ~6–9 % C_63 deficit against the φ-known dense reference appears only when φ is sampled, and it is **stationary**: a truth start falls to the same level as the MAP start (two-start test).
- The **strict** ensemble, whose ranks need no reference at all, passes strict SBC at ℓ = 63 and in the a_ℓm `[60,64)` field rank.
- So the deficit is a property of the φ-marginal posterior measured against a φ-known reference, not a demonstrated sampler defect. **D7** makes the strict SBC figure 1. The same ensemble shows a φ `[30,60)` offset (next subsection).

### Figure 1 under D7 (2026-10-14): three blocks pass; φ `[30,60)` fails

`fig1_validation.py` now detects a strict ensemble (`cl_prior_nu` set). It then reads the a_ℓm row against uniform, refuses the flat-prior null, and adds `cltt_sbc.pdf`, the strict SBC of C_ℓ^TT (tests in `tests/test_paper_figures.py`; a biased synthetic sampler is rejected). The Makefile builds `figure1/` from `_nu30_strict` and `figure1_flatprior/` from `_nu30_long` (`logs/fig1_d7_20261014.log`). Each entry is `p_bin`: the Bonferroni-corrected minimum of the per-bin rank-mean and rank-spread p-values, over 4 bins × 2 statistics, thin 50, N = 48:

| block | strict (figure 1) | flat prior (`figure1_flatprior`) |
|---|---|---|
| C_ℓ^TT | **1.00 ✓** | — (no strict rank under a flat prior) |
| a_ℓm field power | **0.178 ✓** (`[60,64)` mean p 0.27) | 0.005 ✗ (`[60,64)` against the φ-known null) |
| C_L^φφ | **0.185 ✓** | 0.501 ✓ |
| φ field power | **0.014 ✗** (`[30,60)` mean p 0.002, spread p 0.002) | 0.386 ✓ |

50 % power at a 0.28σ posterior-mean shift in both.

- **φ `[30,60)`:** mean_u 0.626 (+3.0σ), sd_u 0.225 (a central clump), the same at thin 10 and 50. On the same 48 φ truths the flat-prior ensemble reads 0.558–0.567, which passes. It is the bin of the ν-dependent funnel offset (ν = 6: 0.685; ν = 30 at N = 24: 0.574), so it is read as **residual mixing of the centred (φ, C_L^φφ) hierarchy at ν = 30**, stated as a limitation. Finding out why the strict C_ℓ^TT prior makes it worse would need φ-sampler work, which needs sign-off.

### Two-start test (T0.1 b, 2026-10-08): the deficit is stationary

**What ran:** job 12100312, the production configuration on r032–r039 started at (a_ℓm, φ) = truth (`--start_truth`; ln C_ℓ = truth only seeds the first mass matrix, since the sweep redraws C before use). 12.3–13.5 h per task on cosma5. Harvest 12100313 (`logs/harvest_t01b_12100313.out`) compared it with the MAP-start `_nu30_long` chains on the same skies.

- **The truth start falls to the MAP start's level and stays there.** C_63/(S_true/(k−4)) by quarter is 0.934 / 0.909 / 0.935 / 0.925, against 0.904 / 0.883 / 0.942 / 0.935 for the MAP start. Truth − MAP, paired, Q4: −0.017 ± 0.036 at ℓ = 63 (7 skies, r036 out), and |Δ| ≤ 0.025 ± 0.03 at ℓ = 60–62. Against the dense φ-known reference, Q4 is −0.102 ± 0.059 (truth start) and −0.085 ± 0.052 (MAP start).
- The drop happened within the 1000 burn-in sweeps, before the first saved quarter.
- The truth start's rank and width are a little cleaner: `[60,64)` z +1.94 against +2.82 (effective null, thin 50); per-mode sd(z) 1.01–1.07 against 1.07–1.11.
- **Reading:** the pre-registered criterion for a stationary deficit is met. Two starts meeting within burn-in cannot exclude a slow collective mode that both enter quickly, so the independent check is the strict SBC below.

### Strict exact ensemble (T0.1 c and T1.1 reference, 2026-10-08)

**What ran:** job 12100331, 48 skies, `--cl_prior_nu 30 --cl_phiphi_prior_nu 30`, truths drawn from those priors, 1000 + 3600 sweeps, 12.3–15.1 h per task. Harvest 12100517 plus the 2026-10-14 re-run of its strict-SBC step (`logs/harvest_strict_12100517{,_rerun}.out`).

- **Strict spectrum SBC passes in every bin**, full chain (thin 20) and second half (thin 10): C_ℓ^TT p 0.17–0.96, C_L^φφ p 0.051–0.87. At ℓ = 63, C_ℓ^TT has mean rank 98.4/180 (p 0.17) and 99.7/180 (p 0.96), the sign of a low posterior but consistent with uniform.
- **a_ℓm field ranks pass** (SBC-supported in strict mode): `[60,64)` mean_u 0.555 (thin 10, cal_p 0.040) / 0.545 (thin 50, cal_p 0.20); pooled 0.501.
- **Joint-likelihood SBC passes:** 0.528, KS_p 0.19 (thin 40).
- **T0.1 c, collapse: 0 collapsed (chain, ℓ) pairs on 48 skies**, against 1/48 under the flat prior in both `_nu30_long` (r036) and `_nocl4`. The strict r036 has the same φ and the same Gaussian a_ℓm draws, rescaled by its drawn C_ℓ, so this is close to r036 re-run under the proper prior.
- **φ `[30,60)` field rank mean_u 0.626, sd_u 0.225, KS_p < 0.001** (thin 10 and 50). Figure 1's calibrated test fails it (`p_bin` 0.014; subsection above).
- Full tables: `docs/dashboard.md` → CURRENT.

### φ-fixed run (T0.1 a, 2026-10-07): the a_ℓm/C_ℓ blocks are exact; the deficit needs φ sampling

**What ran:** job 12099339, the production sampler with φ held at φ_true (`--phi_fixed_truth`, Blocks 1 + 2 on the lensed likelihood), skies r032–r039, otherwise the production configuration. It took 35 min per task, against ~19 h with the φ block. Harvest: job 12099555 (`logs/harvest_t01a_12099555.out`).

- **It matches the dense exact reference on the same skies.** At ℓ = 60/61/62/63 the paired Q4 differences are −0.007 ± 0.002 / −0.002 ± 0.002 / +0.000 ± 0.004 / **+0.006 ± 0.006** (r036 excluded). The control on the same skies is −0.085 ± 0.052 at ℓ = 63. C_ℓ is flat across quarters.
- **Ranks and widths are clean.** `[60,64)` z −0.46 against the effective null (control +2.82); real parts +0.67 (control +3.41); per-mode sd(z) 0.97–1.03; shrinkage slope at ℓ = 63 0.913 against W 0.918 (control 0.645 against 0.740).
- **Reading.** Block 1, Block 2 and the lensed likelihood at ℓ = lmax − 1 are exact. The deficit needs φ to be sampled.
- The ℓ = 60 offset (0.7 %, z −2.7) is too small to matter at figure level. It may be the dense reference's Monte Carlo error, which the paired SE leaves out.

### `--cl_init data` pilot (2026-10-03): the start is not the cause

**What ran:** job 12078506, skies r032–r039, the production configuration (Block-4-ON, ν = 30, 1000 + 3600, lmax = nside = 64), with `--cl_init data`: ln C_ℓ starts at S_ℓ(a_MAP)/(2ℓ−3). Output `ens_exact_l64_A3000_n30_nu30_clinit_data/`; 18.8–19.0 h CPU-only on cosma8-shm. Diagnostic: job 12090562 (`logs/alm_band_edge_12090562.out`). Control: the same 8 skies of `_nu30_long`, symlinked as `ens_exact_l64_A3000_n30_nu30_long_r032_r039/` (identical `alm_true_packed` and `cl_phiphi_true`); every comparison is paired by sky.

- **The data start removes the early transient but not the deficit.** C_63 / (S_true/(k−4)), the mean over the 8 skies by quarter:

  | | Q1 | Q2 | Q3 | Q4 |
  |---|---|---|---|---|
  | pilot | 0.946 | 0.944 | 0.921 | 0.935 |
  | control | 0.904 | 0.883 | 0.942 | 0.935 |
  | paired difference | +0.04 ± 0.02 | +0.06 ± 0.02 | −0.02 ± 0.03 | 0.00 ± 0.03 |

- **The rank offset shrinks by about a third but stays.** `[60,64)` z +1.86 (pilot) against +2.82 (control) at thin 50; ℓ = 63 sd(z) 1.07 against 1.09 (exact 0.97–0.99); shrinkage slope at ℓ = 63 0.656 against 0.645 (W_eff 0.74).
- **It does not prevent the r036 collapse.** C_62 started at 1.2e-6 of the realized power (S_ℓ(a_MAP) inherits the MAP's shrinkage) and recovered by itself in Q3–Q4 (0 → 0.17 → 0.72).

### Band-edge localisation (2026-09-30): lmax 64 only, and not caused by Block 4

Jobs 12066584 (Block-4-OFF extended to N = 48), 12066585 (lmax = nside = 32) and 12066667 (lmax 32, nside 64), all 48/48, diagnosed by job 12074645 (`logs/alm_band_edge_12074645.out`).

- **lmax 32 is clean at both pixel scales.** Top bin `[28,32)` z −0.5 / −0.3; ℓ = 31 z −1.2 / −0.3; C_31/(S_true/(k−4)) flat at 0.99–1.04. The lag is not a generic band-edge property of the sampler, nor of nside = lmax pixelisation.
- **Block-4-OFF shows the same offset:** `[60,64)` z +3.4 to +3.9, ℓ = 63 z +3.1 to +3.7, C_63 at 0.926–0.943 of S_true/(k−4) in every quarter. It is not caused by Block 4 or the ν prior.
- **Block-4-OFF figure 1 at N = 48, thin 50** (scratch, not the paper figure): φ p_bin **0.180** ✓, a_ℓm min p_bin **0.000** ✗ from `[60,64)`, 50 % power at 0.314σ.

### Band-edge diagnosis (2026-09-27)

- **The offset is real, and it is ℓ = 63.** Against the effective exact-sampler null it is +2.55 on r000–r023, **+2.33 on the fresh r024–r047 alone**, and +3.45 combined; every other a_ℓm bin is at |z| ≤ 0.6. Per-ℓ (job 12066643): z +3.5 to +3.8 at ℓ = 63 on the ν = 30 ensemble, +2 to +4 in every other exact ensemble, ≤ +1.5 at ℓ = 60–62. m = 0 is clean; real parts carry more of it than imaginary parts.
- **The diagonal a_ℓm null is validated against an exact dense reference.** `scripts/exact_dense_alm_reference.py` builds the full lensed operator at φ_true from `lens_map_tf` (reproduces it to 6e-15), rebuilds the production data from the production seeds, and samples (C_ℓ, a_ℓm) exactly (job 12066700, 48 skies): no rank offset at any top ℓ (ℓ = 63 z +0.3/+0.8), sd(z) 0.99–1.02, shrinkage slope equal to W (0.891 against 0.894). So pixelisation and the lensing geometry do not bias the statistic **when φ is known**.
- **The MAP start sets every C_ℓ low**, at about e^−1 × truth: `find_map_estimate`'s joint (ln C, a) MAP is the usual hierarchical-MAP shrinkage (replay job 12066670).
- **The slow direction is collective**; the single-coordinate τ_int(ln C_63) ≈ 42 does not show it. A rank-mean quarter test missed the drift; C_ℓ / (S_true/(k−4)) by quarter caught it.
- **Broader, milder:** per-mode sd(z) of a_ℓm is 1.03–1.08 across all ℓ at ν = 30 (1.18–1.25 in the 1200-sweep chains) against 1.00 exact. The rank test cannot see this, because its effective null is calibrated from the chains' own variance. Quote sd(z), not only the rank.

### C_ℓ collapse under the flat C_ℓ prior

- In ν = 30 r036, C_62 is ~2e-5 of the realized power for the whole chain (a_62 sd ≈ 3e-3 against 1.1 prior). ν = 6 long r021 collapsed the same way at ℓ = 62 and recovered only late. Block-4-OFF at N = 48 has one collapsed sky (ℓ = 62 sd(z) 100.6).
- It is seeded by a low MAP start (r036's MAP put C_62 at e^−2). Once C is tiny, HMC can hardly move the stiff a_ℓm. It is caught only by a per-mode z (|z| ≈ 700); the power ranks shift by ~0.3σ.
- **Not caused by φ:** with φ fixed, r036's C_62 starts collapsed and recovers by Q4. It comes from the MAP start in Blocks 1 + 2.
- **The proper ν = 30 C_ℓ^TT prior removes it** on 48 strict skies (0 collapsed; above).
- **r036 stays in `_nu30_long` (D8).**

### Tools built for T0.1 (all tested)

- `fig1_validation.py` strict mode (D7): the a_ℓm row is read against uniform, and `cltt_sbc.pdf` is added.
- `diagnose_alm_band_edge.py`: per-ℓ/m/part ranks against the null, thinning, quarters, per-mode sd(z), shrinkage slope, C by quarter (`tests/test_diagnose_alm_band_edge.py`, including a shrunk-sampler mutation). `cl_ratio_per_chain` is the shared per-chain ratio.
- `exact_dense_alm_reference.py` (`tests/test_exact_dense_alm_reference.py`: a | C mean and covariance, and C | a against the InvGamma median).
- `compare_band_edge_paired.py`: C_ℓ/(S_true/(k−4)) by quarter for several ensembles on the same skies, paired against a reference, with SE and z; refuses to pair chains whose `alm_true_packed` differ (`tests/test_compare_band_edge_paired.py`, mutation-checked).
- `--phi_fixed_truth` (`run_gibbs_chain(phi_fixed=True)`, wrapper `MODE=phifixed`), `--start_truth` (`MODE=truthstart`; refuses `--blind` and `--cl_init data`), `--cl_init data`.
- `--cl_prior_nu` (strict mode; `model.sample_cl_given_alm(prior_nu, cl_fid)`, `run_gibbs_chain(cl_prior_nu, cl_prior_fid)`): a proper conjugate InvGamma(ν/2, ν C_ℓ^fid/2) prior on C_ℓ^TT, the Block 1 twin of Block 4's. The truth is drawn from it on its own seed stream; with `--cl_phiphi_prior_nu` every block has a strict SBC rank. Default `None` reproduces the old draw bit for bit (`tests/test_strict_sbc_priors.py`).
- The ensemble wrapper takes `LMAX`/`NSIDE`.

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

### Test suite extended to the scripts (2026-09-22/23)

- `tests/test_lensing_exact.py`: φ = 0 identity, a brute-force Y_ℓm sum at independently remapped points, geodesic distance, FD gradients, sign conventions against `hp.alm2map_der1`.
- `tests/test_paper_figures.py`: every figure script end to end on synthetic ensembles; R̂/ESS against AR(1) theory; rank-test calibration and power (figure 1 passes a correct and rejects a biased synthetic sampler).
- `tests/test_scripts_pipeline.py`: `coverage_ensemble_chain.py` aware and blind end to end, replay through the chain's own operator, the Block 4 PIT with a failing misaligned control.
- `tests/test_qe.py`: QE sign on exact-lensed skies; N_L's absolute normalisation frozen at lmax = 16 against an independent sympy-3j loop (agreement 4e-16).
- Mutation checks reintroduced the alm-ordering bug, the Block 4 −1, the tight bbox, the beamed prior and the old QE sign; each is caught.

The lessons from this period are now in `ROADMAP.md` → Standing discipline.

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
- **The joint-likelihood replay assumed the full-sky, beamless, uniform-noise geometry (2026-10-17, caught before any real-sky harvest):** `sbc_joint_likelihood.py` rebuilt y = lens(truth) + N(0, σ) on every pixel, so on a T1.2a chain it scored the sampler against a data map the chain never saw: χ²/n 2.43 for a posterior draw on r000, against 1.005 with the chain's own mask, beam and noise (truth 1.009). Its truth check could not catch this, because the truths do not depend on the geometry. The replay now builds the chain's geometry (shared `real_sky_setup.apply_mask`), and chains save `data_masked`, which the replay must reproduce to 1e-9 or it refuses. A test runs a masked, beamed chain and checks that the replay refuses a wrong beam, a wrong noise map and a wrong mask.
- **`psi_lensed` gathered the mask twice (2026-10-16):** both lensing operators already return the map at `unmasked_idx`, and the matrix-free likelihood gathered it again. That is the identity on a full sky (every earlier ensemble) and on a leading-block mask, and an out-of-range index on a real mask. Fixed in `ec0df2d`, with `test_masked_beamed_anisotropic_chain_runs_and_records_its_geometry`.
- **`strict_sbc_ranks.py` read the checkpoints as chains (2026-10-08, harvest 12100517):** its `chain_r*.npz` glob matched `chain_rNNN_ckpt.npz`, so it refused the ensemble as non-strict and the strict-SBC step exited 1 while the rest of the harvest ran. `aggregate_coverage_ranks.py` already filtered `_ckpt`; no other script has the bare glob. Fixed, with `test_cli_skips_checkpoint_files` (fails before the fix).
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
- **The legacy S1 framing ("bias reduction leads as validation, 93.7 %") is void** with the retraction. Do not rebuild it on physical lensing: at these ℓ the physical C_ℓ^TT effect is +0.05–0.2 %. The new figure 2 shows a blind fit absorbing exactly the lensing each sky received while the joint fit does not, under amplified lensing, stated as such. The "recovers A_L = 1 without a template" framing survives if the new figure supports it.
- **The narrow form of the blind-model claim still holds:** against map-based Gibbs methods (Commander does not model lensing), not against Planck's likelihood (lensed spectra plus an A_L nuisance).
- **Doeser & Jasche (`2606.10023`)** is cited in `main.tex`'s introduction.
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
