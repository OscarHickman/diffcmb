# Sampling & Validation Dashboard
*Last updated: 2026-09-24*

Live status of the production chains. Forward plan: `ROADMAP.md`. Closed-out
results and the bug record: `achievements.md`.

## CURRENT (2026-09-24): exact-operator ensembles — calibration fails; T0.1 diagnosis in progress

Everything below the line further down was made with the **legacy forward model**:
bilinear-interpolation lensing on an nside = lmax grid, lensing by −∇φ, and a
defective fiducial cosmology. It is reproducible and kept as the record, but it
is **not** a statement about lensing on the sky, and its bias-reduction numbers
(93.7 % / 98.4 %) are **retracted** (`achievements.md`). Cite nothing below the
line as a current result.

**Configuration (decisions D3/D4):**
- exact geodesic lensing operator (`lensing_operator='exact'`, physical +∇φ);
- corrected fiducial (`power.fiducial_spectra`);
- amplified lensing **A_φ = 3000**, pixel noise **σ = 30 μK**;
- lmax = nside = 64;
- 400 burn-in + 1200 saved sweeps;
- `phi_n_lfs=240`, `phi_mass_matrix=prior`.

Every chain file records `lensing_operator`, `fiducial`, `phi_amplitude` and
`noisesig`; every replay reads them.

| ensemble | job | directory | state | chain health |
|---|---|---|---|---|
| Block-4-ON, ν = 6 (title's object) | 12038494 | `ens_exact_l64_A3000_n30/chain_rNNN.npz` | **24/24 COMPLETED** | `phi_calibration_ok` 24/24; φ power / truth median 0.87 (0.41–1.70); alm accept 0.62 |
| lensing-blind, same data | 12038496 | `ens_exact_l64_A3000_n30/blind_rNNN.npz` | **24/24 COMPLETED** | — |
| Block-4-OFF (three-block core) | 12038495 | `ens_exact_l64_A3000_n30_nocl4/chain_rNNN.npz` | **24/24 COMPLETED** | `phi_calibration_ok` 24/24; φ power / truth median 1.07 (0.54–2.89); alm accept 0.64 |

No task `.err` contains a traceback.

**QE validated at this configuration** (job 12038409,
`qe_noise_validation_exact_A3000_n30.npz`): noise ratio **0.995**, response
**0.971**.

**In flight (submitted 2026-09-24, ROADMAP T0.1c):**

| job | what | output | expected |
|---|---|---|---|
| 12041984 | exact-sampler null for the a_ℓm power rank under the flat C_ℓ prior (`scripts/null_alm_power_rank_flat_prior.py`), nominal + chain-calibrated effective noise, vs both ensembles | `logs/null_alm_rank_12041984.out`; `<ensemble>/null_alm_power_rank_flat_prior.npz` | minutes |
| 12041916 | Block-4-ON, ν = 6, **1000 burn-in + 3600 samples** (same skies r000–r023) | `ens_exact_l64_A3000_n30_long/` | ~13 h/task at ~10 s/sweep |
| 12041917 | Block-4-ON, **ν = 30**, same length, same skies | `ens_exact_l64_A3000_n30_nu30_long/` | ~13 h/task; tasks 4–23 were queued at submission |

**How to read them:**
- **Null (12041984).** Compare each bin's observed mean_u with the *effective*
  null. `|z| < 3` in `[30,60)` ⇒ the a_ℓm offset is the statistic (fixed-spectrum
  truth vs flat C_ℓ prior), not the sampler, and the three-block core stands.
  `|z| ≫ 3` ⇒ sampler defect; stop and diagnose Block 2.
- **Long ν = 6 (12041916).** Re-run `diagnose_calibration_stationarity.py`,
  `sbc_joint_likelihood.py` and the null on it. φ `[2,10)` should stop drifting
  (drift z < 3 and a flat quarter profile) and logp should stop falling. If φ
  `[2,10)` still drifts at 3600 samples: T0.1d (lower A_φ).
- **ν = 30 (12041917).** If the φ `[30,60)` rank offset (0.70 at ν = 6) shrinks
  toward 0.5 as ν rises, it comes from the Block-4 hierarchy / prior at ν = 6. If
  it stays, it is a Block-4-ON sampler problem (Block-4-OFF has no φ offset).

### T0.1 (a, b) result — job 12040935 (4 min, harvested 2026-09-24)

Log: `logs/diag_calib_exact_12040935.out`. **Two different failures, not one.**

**1. a_ℓm `[30,60)`: stationary offset in both ensembles, probably the statistic.**

| rank mean_u, a_ℓm `[30,60)` | thin 10 | thin 40 | quarters 1→4 | drift z |
|---|---|---|---|---|
| Block-4-ON | 0.280 | 0.301 | 0.264 / 0.282 / 0.284 / 0.298 | −0.85 |
| Block-4-OFF | 0.273 | 0.286 | 0.229 / 0.263 / 0.251 / 0.285 | −2.08 |

It survives thin 40 and is flat across quarters, so it is not thinning and not
burn-in. The neighbours are offset the other way: `[10,30)` about 0.36–0.39 and
`[60,64)` about 0.60–0.65, both marginal. Candidate cause: the truth is drawn at
the **fixed** C_ℓ^fid while Block 1 uses the **flat** prior. The truth is then
not a draw from the sampler's prior, and the power rank need not be uniform. The
effect is negligible where the data pin a_ℓm down, which is why the
legacy ensembles passed this row. They used the script default σ = 1 μK and
A_φ = 1. It is large where A_φ = 3000 lensing
removes the information. The sign agrees (posterior power above the truth ⇒
mean_u < 0.5). The null job 12041984 tests this.

**2. Block-4-ON φ: two parts.**
- **Low-L φ `[2,10)` is not stationary.** Quarter mean_u 0.771 / 0.717 / 0.614 / 0.661;
  drift z of log(power/truth) **+4.05**; τ_int **173 sweeps** (about 7 effective
  draws per 1200-sweep chain); **logp still falling** (drift z −3.56). This is
  burn-in from the MAP start, so run longer (job 12041916). Block-4-OFF φ `[2,10)`
  has τ_int 138 but no significant drift (z +1.24).
- **φ `[30,60)` is a stationary offset:** 0.696 / 0.706 / 0.696 / 0.715, drift z
  −1.32, survives thin 40 (0.707). It is absent in Block-4-OFF (0.466) and
  matches the strict C_L^φφ `[30,60)` 0.666. So it is Block-4-specific and not
  burn-in (job 12041917).

**Joint-likelihood SBC:**

| | thin 40 | 2nd half, thin 10 |
|---|---|---|
| Block-4-ON | 0.724, KS_p 0.0011 — **REJECTED** | 0.695, KS_p 0.0029 — **REJECTED** |
| Block-4-OFF | 0.492, KS_p 0.91 — pass | 0.472, KS_p 0.91 — pass |

Block-4-OFF passes every φ bin and the joint likelihood at every thinning; its
only failures are the a_ℓm rows above. **If the null explains them, the three-block
core is calibrated on the exact operator.**

The earlier harvest job 12040798 (7 min) and `make figures`
(log `logs/make_figures_exact.log`) both completed 2026-09-23.

### First harvest (2026-09-23) — physics clean, calibration FAILS

| test | Block-4-ON (ν = 6) | Block-4-OFF |
|---|---|---|
| joint-likelihood SBC (thin 10) | **0.716, KS_p 0.0011 — REJECTED** | 0.480, KS_p 0.67 — pass |
| φ field ranks (fig 1 `p_bin`) | **0.004** (mean_u 0.625: posterior φ power low) | 0.039 (mean_u 0.497) |
| a_ℓm field ranks (`p_bin`) | **< 0.001** (`[30,60)` mean) | **< 0.001** (`[30,60)` mean, mean_u 0.426) |
| strict C_L^φφ SBC | pooled 0.552, KS_p 0.067; `[30,60)` 0.666, KS_p 0.020; `p_bin` 0.038 | n/a |
| Block 4 PIT | aligned KS_p 0.937; lag-10/50 controls rejected (lag-1 vacuous, φ lag-1 +0.877) | n/a |
| φ power bias by bin | 0.85 / 0.95 / 0.98 / 1.02 | 1.07 / 0.99 / 1.00 / 1.00 |

**Convergence (Block-4-ON, 1200 sweeps, 120 rank draws):**
- R̂ > 1.01 for 48 % (Block 1) and 55 % (Block 4) of (chain, ℓ) pairs;
- median bulk ESS: C_ℓ 123, **a_ℓm 35**, φ 65, C_L^φφ 80;
- per-mode φ residual sd 1.108 over 24 skies (≈1.03–1.05 expected), i.e. posteriors too narrow.

**Figure 2** (24 same-sky pairs, per bin `[2,10)` … `[45,64)`):

| | `[2,10)` | `[10,20)` | `[20,30)` | `[30,45)` | `[45,64)` |
|---|---|---|---|---|---|
| blind bias (%) | +0.90 | +6.65 | +0.55 | −15.72 | −45.64 |
| lensing received (%) | +1.70 | +6.42 | +0.03 | −15.96 | −46.02 |
| aware bias (%) | −2.96 ± 1.05 | +1.50 ± 0.69 | +0.76 ± 0.82 | +0.85 ± 0.56 | +0.75 ± 0.50 |

The blind fit tracks the lensing received; mean |bias| is 13.9 % blind vs 1.37 % aware.

**Other figures:**
- **Figure 3:** width ratio 0.61 / 0.63 / 0.68 / 0.74 at L ~ 15 / 25 / 38 / 55. **Not interpretable** until the chains calibrate.
- **Figure 4:** 0/16 cells above null.
- **Maps:** r = 0.82 (realization 0).

**Next:** read jobs 12041984 / 12041916 / 12041917 as described under "How to
read them" above. The first-harvest numbers in this table are superseded by
the T0.1 result section once the long chains land.

**Pilots that chose the configuration** (jobs 12037774/5, 12037979–81;
directories `pilot_exact_lmax64_*`):
- At σ = 1 μK Gibbs crawls (φ|alm too sharp).
- A_φ = 10,000 stalls or inflates φ.
- A_φ = 3000 at σ = 30 recovers φ (corr 0.77–0.82, power 0.94–0.98 of truth).

---

# LEGACY RECORD (bilinear operator, pre-2026-09-23) — do not cite as current

## ✅ STATUS 2026-09-13: re-scored at ~60 draws/chain — no known open defect

Supersedes the scores throughout this page, which were taken at ~6–8 rank
draws/chain where the test is badly miscalibrated. Jobs 11986716 / 11986722.
Full account: `achievements.md`.

| statistic | thin=90 (~8 draws) | **thin=10 (~60 draws)** |
|---|---|---|
| φ field, Block 4 OFF | 0.4688, KS_p 0.0537, cal_p 0.2444 | 0.4585, **KS_p 0.2766**, cal_p 0.4038 |
| alm field, Block 4 OFF | 0.5039, KS_p 0.2319, cal_p 0.8659 | 0.4956, **KS_p 0.9987**, cal_p 0.7150 |
| φ field, Block 4 ON | 0.4661, KS_p 0.0913, cal_p 0.4741 | 0.4488, KS_p 0.1355, cal_p 0.1985 |
| alm field, Block 4 ON | 0.4896, KS_p 0.3460, cal_p 0.8928 | 0.4995, **KS_p 0.8439**, cal_p 0.5464 |
| **strict `C_L^φφ` SBC** | 0.4544, **KS_p 0.0009** | **0.4518, KS_p 0.0567, no bin rejects** |
| joint likelihood, B4 OFF | 0.4569 (thin 10) | 0.4587 (thin 5, KS_p 0.738) |
| joint likelihood, B4 ON | 0.3771 (thin 30) | **0.4219** (thin 5, KS_p 0.468) |

**`KS_p` and `cal_p` now agree** — the confirmation that rank discreteness was
the entire calibration problem. **The strict `C_L^φφ` p-value moved 60× while
its mean barely moved** (0.4544 → 0.4518): a measurement artifact, not a sampler
change. The "open sampling question" is closed.

**The autocorrelation cost of thinning less did not materialise** — `sd_u`
0.276–0.310 against uniform's 0.2887, despite τ_int up to 42.5 having motivated
thin=90. **Use `--thin 10` for rank scoring from here.**

**No field bin is a live defect.** The last notable one migrated: φ `[2,10)`
cleared (cal_p 0.015 → 0.401) and φ `[30,60)` took its place (0.052 → 0.011).
A real ℓ-localised defect persists under re-scoring; one that hops does not.

Unchanged: the `C_l^TT` / `C_L^φφ` **coverage** rows still flag against uniform
and still must be read against `validate_coverage_rank_nulls.py`'s simulated
null — the documented rank-vs-mode artifact. And a pass remains a **bound**, not
a proof of exactness, though a tighter bound than at 8 draws.

---

## ⚠⚠ READ FIRST (2026-09-12): the rank test was miscalibrated; scores below are re-derived

Two corrections that touch almost every number on this page. Full account:
`achievements.md`.

1. **`ks_uniform_p` over-rejects.** It compared *discrete* ranks to a
   *continuous* uniform. Fed ranks from a provably correct sampler it returns
   p<0.01 **12.5%** of the time on the pooled N=96 / 8-draw row, and p<0.05
   **30.3%** of the time. The inflation **grows with N** at fixed granularity,
   so doubling realizations made pooled p-values look more significant with no
   change in the sampler. Use the new `cal_p` column (simulation-calibrated at
   each run's own N and rank granularity), never `KS_p`. The cure is more draws
   **per chain**, not more chains — at 60 draws/chain `KS_p` is fine.
2. **A "pass" here is a bound, not a proof.** Calibrated power at N=24 with ~8
   draws/chain: a posterior mean shift of 0.1/0.2/0.3/0.5σ is detected
   7.5/14.3/26.5/**57.5%** of the time, and under-dispersion essentially never
   (≤10.7% even at sd×0.5). **Our non-rejections exclude only defects larger
   than roughly a half-σ posterior mean offset.** Say that, not "exact".

**Net effect: the sampler looks BETTER, and the open defect largely evaporates.**
Re-scored, **no field-rank row in either ensemble is flagged**. Headline φ pooled
KS_p 0.0537 → **cal_p 0.2444**; alm 0.2319 → **0.8659**. The Block-4-ON strict
`C_L^φφ` pooled p, the basis of the "open sampling question" since 2026-09-01,
moves **0.0009 → ~0.04** (and that row is separately anti-conservative). The φ
`[30,60)` bin chased for two weeks relaxes to cal_p 0.052; only φ `[2,10)`
remains notable at 0.015. `C_l^TT` coverage rows still flag and are still the
documented rank-vs-mode artifact — read them against
`validate_coverage_rank_nulls.py`, not against uniform.

---

## Current headline — simulation-based calibration

**lmax=64, nside=64, 12 independent chains, `phi_mass_matrix='prior'`, Block 4 OFF**
(job **11955622**, restored 2L+1 packing; supersedes job 11903181, pre-restoration).
Block 4 off pins `C_L^φφ` at the fiducial spectrum, so the φ prior is proper
*and identical to the process that generated the truth* — which is what makes
the φ rank a genuine calibration test.

| Field rank, **N=24 realizations** (pooled, N=96) | mean_u | KS_p | verdict |
|---|---|---|---|
| φ | **0.4688** | **0.0537** | consistent with uniform |
| alm | **0.5039** | **0.2319** | consistent with uniform |

*(N=12 stage, job 11955622 alone: φ 0.4792 / KS_p 0.4078, alm 0.5130 / KS_p 0.6369.)*
Extended to N=24 by job **11965828** (realizations 12–23, same outdir, same
config), harvested 2026-09-11.

**The φ `[30,60)` flag was chased to N=24 and the small-N-artifact reading
held.** It relaxed rather than sharpened (KS_p 0.005 → 0.014, mean_u 0.260 →
0.339, i.e. toward 0.5), and a *second* bin flagged at the same weak
significance in the **opposite** direction (φ `[2,10)`, mean_u 0.568, KS_p
0.014). Two bins straddling 0.5 at p~0.014 across 8 tests is bin-level noise,
not a coherent one-bin bias — which is precisely what doubling N was run to
distinguish. Neither bin is treated as a live defect. At N=24 all four
`C_l^TT` coverage FLAGs again sit inside their null bands (obs
0.094/0.089/0.120/0.349 vs null 0.095/0.096/0.115/0.342), and φ power bias
per bin has median 0.988–1.025 (one realization reaches 2.23 in `[2,10)`). `--thin 90`, matching the pre-restoration
ensemble's derived value (τ_int max there was 42.5). All four `C_l^TT`
coverage FLAGs sit inside their `validate_coverage_rank_nulls.py` null bands
(observed 0.104/0.073/0.125/0.333 vs null 0.098/0.096/0.114/0.348) — the usual
rank-vs-mode artifact, not bias. φ power bias per bin stays near 1 (median
0.996–1.047, max 1.49 in `[2,10)`). Every realization's phi-power/truth ratio
lands in `[0.735, 1.320]`.

**Pre-restoration reference (job 11903181, 2L packing, superseded 2026-09-08):**
φ mean_u 0.4688 (KS_p 0.124), alm mean_u 0.5312 (KS_p 0.235), no flagged bin
in either field row. Thin-robust at the time: φ 0.475 (p=0.44) at `--thin 30`,
0.453 (p=0.24) at 45. Pre-fix-shape pair (job 11900600) was 0.4534/0.5367.

Reference: the same configuration with Block 4 **on** (flat improper prior on
`C_L^φφ`, job 11899585, pre-restoration) gives φ mean_u = 0.367, KS_p = 0.0040
— a statement about the prior, not about the sampler (see below). This
*flat-improper-prior* Block-4-ON comparison has not been re-run under the
restored packing and remains pre-restoration. The **proper-prior** (ν=6)
Block-4-ON ensemble *has* been re-run under the restored packing and is the
certification of the paper's four-block claim — jobs 11965813 + 11980637,
see the banner at the top of this page.

Mixing, pre-restoration runs: τ_int median 4.7–27 per bin with Block 4 off, vs
24–56 with it on. R̂ ≤ 1.07 outside the lowest and highest bins.

---

## ⚠→✅ The `[10,30)` residual DISSOLVED at N=24 (2026-09-12, job 11980637)

Doubling the Block-4-ON proper-prior ensemble to N=24 (realizations 12–23 of
job 11980637 appended to job 11965813's 0–11) **removed the localisation that
defined this issue.** Strict `C_L^φφ` SBC rank, per bin:

| bin | N=12 | **N=24** |
|---|---|---|
| `[2,10)` | 0.4062 (KS_p 0.154) | 0.4688 (KS_p 0.137) |
| `[10,30)` | **0.2812 (KS_p 0.0047)** ← flagged | **0.4219 (KS_p 0.137)** — not flagged |
| `[30,60)` | 0.5417 (KS_p 0.727) | 0.5000 (KS_p 0.933) |
| `[60,64)` | 0.4688 (KS_p 0.727) | 0.4271 (KS_p 0.137) |
| POOLED | 0.4245 (KS_p 0.00395) | 0.4544 (KS_p 0.00090) |

**No individual bin rejects at N=24.** The `[10,30)` bin relaxed from 0.281 to
0.422 exactly as the Block-4-OFF `[30,60)` flag relaxed when chased the same
way — so "a localised, ℓ-dependent defect in `[10,30)`", the framing this
investigation ran on since 2026-09-01, was a small-N artifact. Every diagnosis
built on that localisation (the cross-L Hessian-coupling story, the Nystrom
mass-matrix attempts) was chasing a bin that no longer stands out.

**What remains is smaller, global, and differently shaped:** a uniform downward
offset of ~0.045 in mean_u across all four bins. The pooled KS_p *fell*
(0.00395 → 0.00090) while every bin individually passed — not a contradiction:
the pooled test is explicitly anti-conservative (bins within a realization share
a chain and are correlated), so doubling N makes a small consistent offset
"significant" there without any bin rejecting. Read the pooled value as
indicative of a small global offset, not as a localised defect.

**The joint-likelihood SBC agrees and localises the difference to the
configuration, not the ℓ-range** (`scripts/sbc_joint_likelihood.py`, Modrak et
al. test quantity):

| ensemble | thin=10 | thin=30 |
|---|---|---|
| Block 4 **OFF** (headline) | 0.4569 (KS_p 0.738) | 0.4417 (KS_p 0.468) |
| Block 4 **ON** (proper prior) | 0.4028 (KS_p 0.256) | 0.3771 (KS_p 0.067) |

Both are "consistent with uniform", with Block-4-ON sitting lower.

⚠ **The gloss first written here — "mean_u < 0.5 means the draws fit the data
better than the truth, i.e. a too-narrow posterior" — was wrong on both counts
and is corrected 2026-09-12.** Simulation (`scripts/validate_sbc_statistic_power.py`)
shows **under-dispersion does not move mean_u at all** (posterior sd ×0.5 leaves
it at 0.499; it shows up as rank *spread* instead), and a posterior mean biased
*low* pushes mean_u **above** 0.5. A sub-0.5 mean therefore means the posterior
sits **high** relative to the truth — consistent with the measured φ over-power
(medians 1.033–1.122) — and says nothing about width. Width lives in `sd_u`,
now reported: measured at 0.215–0.314 against uniform's 0.2887, i.e. the
posteriors are mildly too **wide** (conservative), not overconfident.
Block 4's own conditional remains exact (PIT aligned KS_p 0.339, lag-10/50
controls rejected at KS_p=0 — a genuine pass).

**Current reading: the Block-4-ON funnel carries a small, global,
under-dispersion consistent with incomplete φ mixing — not a wrong conditional
and not an ℓ-localised defect.** Report it as a bounded caveat on that
configuration; the Block-4-OFF headline is unaffected and clean on every
statistic including the new one.

---

## ✅ RESOLVED-AS-NOT-THE-CAUSE: the packing does not explain the `[10,30)` residual

**Job 11965813, harvested 2026-09-11** — job 11903182's exact configuration
(lmax=64, ν=6 proper `C_L^φφ` prior, Block 4 ON, `phi_n_lfs=240`) re-run under
the restored 2L+1 packing. This was ROADMAP "Next actions" #1: every number
for this configuration was pre-restoration, so it had to be re-established
before any further Block 3 work.

| Statistic | pre-restoration (11903182) | **restored packing (11965813)** |
|---|---|---|
| strict `C_L^φφ` SBC rank | 0.3802 (KS_p 0.0013) | **0.4245 (KS_p 0.00395)** |
| worst bin | `[10,30)` = 0.292 | **`[10,30)` = 0.281 (KS_p 0.0047)** |
| other three bins | pass individually | pass (KS_p 0.15 / 0.73 / 0.73) |
| φ power bias in `[10,30)` | median 1.08–1.14 | median **1.162** |
| Block 4 PIT (aligned) | 0.4999 (KS_p 0.42), genuine | **0.5001 (KS_p 0.248), genuine** |

> ⚠ **This section records the N=12 state and its "still localised to
> `[10,30)`, still a firm rejection" verdict did NOT hold up.** At N=24 the
> localisation dissolved and under a calibrated test the pooled rejection went
> with it (section above). The conclusion that *the packing is not the cause*
> stands; the framing of what it is not the cause **of** does not. Kept because
> the packing-exclusion argument is still needed.

**Verdict (N=12, superseded): the residual survives the restoration, essentially
unchanged in location, direction and size.** The rank is the best of the three measurements
on both axes (vs 0.3802/0.0013 at the same trajectory length and 0.4196/0.00049
at doubled length) but is still a firm rejection of uniformity, and it is still
localised to `[10,30)` with φ over-powered in that same bin. Since the
restoration changed *both* the alm dof (2L→2L+1) and Blocks 1/4's inverse-Gamma
shape (L−1 → L−0.5), and moved the rank only marginally, **the packing is now
excluded as the cause** — which retrospectively answers the Step 0 question the
restoration scoping plan skipped: the missing dof and the `[10,30)` residual
were not linked. Remaining candidate is unchanged: Block 3 (φ|alm,C_ℓ)
mixing/conditioning in `[10,30)`.

The Block 4 PIT control behaved as the checklist requires: lag-10 and lag-50
rejected at KS_p=0, so the aligned pass has power. Note lag-1 alone would have
been near-vacuous — φ lag-1 decorrelation is +0.915 (lag-10 +0.531, lag-50
+0.095). Always pass `--control_lags 1,10,50`.

---

## ⚠ Everything below this line predates the 2026-09-06 `Im(a_{L,1})` restoration

The Block-3-mixing / proper-prior investigation below (jobs 11899585 through
11913324) was run entirely under the old 2L packing (one fewer real dof per
multipole than a real sky has). None of it has been re-run under the restored
packing yet. Kept as historical record and as the likely starting point if
this investigation resumes, not as current status.

---

## ⚠ Open: the proper-prior configuration is NOT yet SBC-validated

**Job 11903182** (Block 4 ON, proper conjugate prior ν=6, corrected shape) is
the source for the paper's joint `(C_ℓ, C_L^φφ)` differentiator figure, and its
strict `C_L^φφ` SBC rank does **not** clear:

| Statistic | Result | Verdict |
|---|---|---|
| alm field rank | 0.5052 (KS_p 0.408) | clean |
| Block 4 PIT given φ | 0.4999 (KS_p 0.42) | **genuine pass** — lag-10/50 controls rejected at KS_p=0 (fixed 2026-09-01) |
| φ field rank | 0.4115 (KS_p 0.0264) | low, worst bin `[10,30)` = 0.292 |
| **strict `C_L^φφ` SBC rank** | **0.3802 (KS_p 0.0013)** | **improved from 0.25–0.28 pre-fix, still not uniform** |

The φ and `C_L^φφ` deficits are the same deficit: Block 4 is exact given φ, so
if `C_L^φφ` ranks low its conditioning `S_L(φ)` must be high — measured φ
power/truth median 1.08–1.14, concentrated in the same `[10,30)` bin. Not
burn-in (0.400 on the 2nd half, 0.396 on the last quarter; φ power *rises*
1.002 → 1.060 across the chain). Leading hypothesis is Block 3 mixing under the
Block-4-ON funnel (τ_int max 92.7 vs 42.5, split-R̂ max 1.64 vs 1.31), not a
wrong conditional — and that hypothesis is now better supported than it was,
because the Block 4 PIT's pass is no longer vacuous (see the row above). Any
figure sourced from this job must carry the caveat.

**HARVESTED 2026-09-02: job 11912088 (12/12 COMPLETED), doubled φ trajectory
(`phi_n_lfs` 240 → 480) — intermediate result, neither predicted branch.**

| Statistic | 240-traj (11903182) | 480-traj (11912088) |
|---|---|---|
| strict `C_L^φφ` SBC rank | 0.3802 (KS_p 0.0013) | **0.4196 (KS_p 0.00049)** |
| worst rank bin | `[10,30)` = 0.292 | `[10,30)` = 0.262 (KS_p 0.0001) |
| φ field rank | 0.4115 (KS_p 0.0264) | 0.4345 (KS_p 0.0741) |
| Block 4 PIT (aligned) | 0.4999 (KS_p 0.42), genuine pass | 0.5025 (KS_p 0.342), genuine pass |
| τ_int, per-bin max across chains | max 92.7 (bin unspecified) | `[2,10)` 100.2, `[10,30)` 48.3, `[30,60)` **107.8**, `[60,64)` 86.8 |

The rank moved *toward* 0.5 (0.38 → 0.42) but the KS_p got *smaller*
(0.0013 → 0.00049) — still a firm rejection of uniformity, not the "→0.5"
branch the roadmap called a pass. τ_int fell in 3 of 4 bins (most sharply
in the previously-worst `[60,64)` bin, 326 → 87) but **did not fall** in
`[30,60)`, and Geyer's estimator truncated early (window exhausted before
finding a non-positive pair) in all 48 chain×bin combinations, so all these
τ_int numbers are lower bounds, not converged estimates. The failure is
now concentrated almost entirely in one bin, `[10,30)` (KS_p 0.0001; the
other three bins individually pass at KS_p 0.16–0.57) — a localized,
ℓ-dependent residual is a different shape of evidence than "the whole
spectrum is under-mixed," and doubling trajectory length again is not
obviously the next lever. **Recommendation carried to `ROADMAP.md`: stop
scanning `phi_n_lfs` and look at Block 3 (φ|alm,C_ℓ) mechanics specifically
in the `[10,30)` range** — e.g. whether the HMC step size/mass matrix is
comparably well-conditioned there vs the bins that do pass.

**HARVESTED 2026-09-02: pilot job 11913324 (`block` mass matrix, $n_{\mathrm{probes}}=24$, Block 4 ON, ν=6 proper prior, 600 sweeps):**
Tested whether Nystrom rank deficiency explained previous block mass matrix failures. Result: **falsified**.
- `[10,30)`: $\tau_{\mathrm{int}}=25.3$ vs $14.8$ for baseline `prior` (no improvement in target bin).
- `[2,10)`: $\tau_{\mathrm{int}}$ severely regressed to $113.7$ (vs $48.3$), $\hat{R}=1.951$, drift $-2.39\sigma$.
- Sweep time $+44\%$ ($35.1\text{s}$ vs $24.3\text{s}$).
- **Conclusion:** The non-diagonal Nystrom mass matrix route is closed post-fix. Baseline remains `phi_mass_matrix='prior'`.

---

## ⚠ How to read the spectrum rows

`aggregate_coverage_ranks.py` FLAGs the `C_l^TT` and `C_L^φφ` rows in every run.
**Those flags are not evidence of bias.** The statistic ranks the truth's
realized power `S_L/k_L` (`k_L` = the packed dof, `2L` for every run on this page, `2L+1` after the 2026-09-06 restoration) against posterior draws — and that is exactly the
*mode* of the inverse-Gamma conditional. An inverse-Gamma is right-skewed, so
`P(draw < mode) < 0.5` for a *correct* sampler, and bin-averaging shrinks the
spread while preserving the offset, driving the mean rank toward zero.

Always compare against the simulated null:

```bash
PYTHONPATH=diffcmb .venv/bin/python scripts/validate_coverage_rank_nulls.py \
    --indir results/analysis/<ensemble dir> --thin <matching thin>
```

Measured under the **corrected shape** — observed vs null, all inside the 95%
band (`C_l^TT` from job 11903181, `C_L^φφ` from job 11903182):

| bin | `C_l^TT` obs | null | `C_L^φφ` obs | null (φ-trajectory) |
|---|---|---|---|---|
| [2,10) | 0.083 | 0.086 | 0.260 | 0.279 |
| [10,30) | 0.083 | 0.092 | 0.302 | 0.339 |
| [30,60) | 0.094 | 0.114 | 0.427 | 0.457 |
| [60,64) | 0.333 | 0.344 | 0.406 | 0.376 |

Note this row is *interval coverage*, distinct from the **strict** `C_L^φφ` SBC
rank in the open-issue section above — that one ranks `cl_phiphi_true` (drawn
from the sampler's own prior) among the Block 4 samples, is uniform under a
correct sampler with no null needed, and is the statistic that does not clear.

The `C_L^φφ` null must retain the chain's own sweep-to-sweep φ scatter; a null
that freezes φ at truth is far too narrow and makes a correct sampler look
biased. `C_l^TT` needs no such correction because alm is pinned at cosine 0.9998.

---

## In flight

**Nothing.** The queue holds no diffcmb jobs. The last three to land:

| Job | What | Result |
|---|---|---|
| 12015546 (2026-09-18) | `submit_validate_qe_noise.slurm` — Monte Carlo validation of `diffcmb/qe.py`'s QE noise `N_L^φφ` against the estimator's own definition, 64 sims | **PASS**: noise ratio 0.992, response 0.883. Artifact `results/analysis/qe_noise_validation.npz`. **Valid only at (lmax=64, nside=64, σ_pix=1.0)** — `fig3` checks this and refuses on a mismatch |
| 12015488 (2026-09-17) | `submit_compare_cl_bias_reduction_lmax192.slurm` — lmax=192 bias-reduction harvest | **98.4%** reduction over five reliable bins; blind deficit to −8.0% at `[160,192)`. **Bias reduction only — NOT an exactness claim** (φ gate NO-GO at 192) |
| 12015538 (2026-09-18) | First, **invalid** QE validation design (vs `estimate_phi_diag_fisher`) | **Superseded — do not cite.** Compared two different quantities (fixed-alm Fisher vs CMB-marginalised noise); ratio 132, trend +0.66 in ln L. `achievements.md` records why |

**Previously in flight, now complete: job 11980637** — `scripts/submit_coverage_ensemble_lmax64_prior_cl4_properprior_packingv2_extendN.slurm`,
extending the Block-4-ON proper-prior ensemble from N=12 to N=24 (realizations
12–23, same outdir as job 11965813, byte-identical config — not a φ-tuning
run). Doubles the chain count for the joint (C_ℓ^TT, C_L^φφ) differentiator
figure, whose error bar is a *chain*-level bootstrap, and tests whether the
`[10,30)` strict-rank residual relaxes or sharpens at N=24. ~5h/realization.

**Completed 2026-09-11: job 11980570** — `scripts/submit_lensing_blind_baseline_packingv2.slurm`,
re-running the Commander-style lensing-blind `C_l^TT` baseline under the
restored packing. Required because the existing
`lensing_blind_baseline_lmax128.npz` (2026-08-12) is packed at the OLD 2L width
(`alm_true_packed` 16254 vs `packed_length(128)`=16380, a `lmax-2`=126
shortfall) and predates both the ordering fix and the dof restoration — it is a
different model from the lensing-aware chain it is meant to be the reference
for, so the bias-reduction figure could not be built from it. Same config
(seed=0, lmax=128, nside=128, noisesig=1.0), new output path; the stale file is
kept, not overwritten. Ran in **4 minutes**, not the ~3h the header guessed —
with no φ block the sampler is 0.1 s/sweep. `alm_true_packed` verified at 16380.
**The bias-reduction figure is now built and positive** — see below.

**Harvested and closed 2026-09-11:** jobs 11965813 (Block-4-ON proper prior,
above), 11965828 (N=12→24 extension, headline section) and 11966631 (lmax=128
lensing-aware chain, below). Previously in flight: Both the packing-v2 pilot (job 11951115) and the packing-v2
12-chain ensemble (job 11955622) completed and were harvested 2026-09-08 —
see "Current headline" above. `restore-im-alm-l1-dof` is merge-ready, not yet
merged into `main`; `docs/paper/main.tex` is clear to update onto the
confirmed 2L+1 numbers. Both are pending explicit user sign-off (`ROADMAP.md`).

The pre-existing open question from before the dof restoration is unchanged
and unaddressed by any of this: Block 3's conditioning in the `[10,30)` bin.
Nothing links the restored dof to it (Step 0 of the scoping plan was
skipped) — see the historical section below.

Standing harvest checklist for any future ensemble: `.err` for tracebacks
(SLURM `COMPLETED` is not sufficient), per-realization φ/truth power ratio
O(1), re-derive `--thin` from each run's own τ_int, and **check the Block 4
PIT's verdict line reports a rejected control** before quoting the aligned
pass (`--control_lags 1,10,50`; lag-1 alone is not enough).

---

## ✅ `C_l^TT` bias reduction — DEMONSTRATED, 93% (2026-09-11)

`scripts/compare_cl_bias_reduction.py` →
`results/analysis/figures/cl_bias_reduction_lmax128.png`. Lensing-aware joint
sampler (job 11966631) vs Commander-style lensing-blind Gibbs (job 11980570),
**identical simulation**, both packing-v2.

| ℓ bin | blind/expected | aware/expected | blind pull | aware pull |
|---|---|---|---|---|
| `[2,10)` | 0.916 | 0.891 | −2.6 | −2.9 |
| `[10,30)` | 0.9986 | **1.0048** | −0.9 | +2.3 |
| `[30,60)` | 0.9914 | **1.0002** | −13.5 | +0.3 |
| `[60,100)` | 0.9738 | **1.0005** | −71.1 | +1.4 |
| `[100,128)` | 0.9456 | **0.9992** | −172.0 | −2.2 |

Mean |fractional bias| over the four reliable bins **0.0226 → 0.0016 (93%
reduction)**. The lensing-aware posterior is consistent with unbiased in every
reliable bin (|pull| ≤ 2.3); the blind deficit grows monotonically with ℓ to
−5.4% — the physically expected signature. `[2,10)` is excluded (φ not
equilibrated at lmax=128, below).

⚠ **The reference is `S_l/(k_l−4)`, the expected posterior mean — not the
realized power and not the fiducial.** Under the flat improper prior Block 1's
posterior mean sits high by `k_l/(k_l−4)` (+15% at ℓ~20, +2% at ℓ~110), an
ℓ-dependent offset common to both chains that is larger than the lensing signal.
Against the realized power this comparison reports *no* bias reduction and names
the lensing-aware chain the more biased one; against the fiducial it adds cosmic
variance on top (ratio 3.66 at `[2,10)`). Both wrong references reverse the
conclusion. See `achievements.md`.

---

## lmax=128 lensing-aware chain — NO-GO, but informative (job 11966631)

First lmax=128 run since the ordering fix and the dof restoration, so it is the
first *confirmed* post-fix data point on the low-L φ mode (2300 sweeps,
48.6 s/sweep, 25.7h; HMC, `phi_mass_matrix='prior'`, `phi_n_lfs=240`, seed=0).

| ℓ-bin | lag-1 | lag-10 | lag-50 | first \|r\|<0.2 | drift σ |
|---|---|---|---|---|---|
| `[2,10)` | **0.967** | 0.854 | 0.515 | lag 150 | −0.52 |
| `[10,30)` | 0.919 | 0.639 | 0.155 | lag 50 | −0.60 |
| `[30,60)` | 0.801 | 0.403 | 0.180 | lag 50 | −0.25 |
| `[60,100)` | 0.750 | 0.236 | 0.069 | lag 25 | +0.04 |
| `[100,128)` | 0.819 | 0.195 | −0.028 | lag 10 | −0.17 |

**NO-GO** on the gate (worst lag-1 0.967 ≥ 0.9), *re-confirming* the pre-fix
"genuine, low-L-specific long-lived mode" verdict rather than overturning it —
marginally better than the pre-fix 0.996, same bin, same character. The chain is
healthy otherwise: φ acceptance 0.691, alm 0.744, drift ≤ 0.60σ everywhere, and
every bin above `[2,10)` decorrelates within 10–50 lags. **So the mode is
narrowly low-L, not a global mixing failure — and it is now the sole blocker on
lmax=128 being ensemble-ready.** Per ROADMAP's standing rule, no further
φ-equilibration tuning launched without sign-off.

Do **not** read this as a licence to build the bias-reduction figure's low-ℓ
bins from this chain: `[2,10)` has ~15 effective samples at best. The mid/high-ℓ
bins, where the lensing suppression of `C_l^TT` is largest anyway, are usable.

---

## Invalid output — do not aggregate or cite

| Directory | Job | Why |
|---|---|---|
| `coverage_ensemble_lmax64/` | 11848757 | Pre-dates the 2026-08-24 alm ordering fix |
| `coverage_ensemble_lmax64_prior_cl4/` | 11887897 | Cold-start alm; φ frozen 1e3–1e5× above truth |
| `lensing_blind_baseline_lmax128.npz` | (2026-08-12) | Packed at the OLD 2L width (`alm_true_packed` 16254 vs 16380) — pre-ordering-fix *and* pre-dof-restoration. Superseded by `..._packingv2.npz` (job 11980570) |

Both `..._prior_cl4_mapfix/` (11899585) and `..._prior_nocl4/` (11900600) are
valid but were produced with the pre-2026-08-31 inverse-Gamma shape; their
field-rank conclusions stand, their spectrum values shift slightly at low ℓ.
**Superseded** by `..._prior_nocl4_doffix/` (11903181) and
`..._prior_cl4_properprior_doffix/` (11903182) — cite those.

⚠ Any aggregation run *before* 2026-08-31's analysis-script fix is also stale:
`aggregate_coverage_ranks.py` and `validate_coverage_rank_nulls.py` both still
hardcoded the `2L+1` dof assumption after the samplers were corrected, so a
harvest from that window compared corrected chains against a stale reference.
Re-run both scripts rather than quoting an older printout.

---

## Differentiator figure — built, and currently a null

`scripts/plot_joint_cl_clpp_posterior.py` measures the **within-posterior**
correlation between `C_ℓ^TT` and `C_L^φφ` on job 11903182 (each chain
standardised before pooling, so this is not the cosmic-variance scatter of the
12 truths). Output: `results/analysis/figures/joint_cl_clpp_posterior.png`.

**1 of 16 bin-pair cells exceeds its permutation null, against 0.8 expected by
chance — no detection.** Strongest cell is `C_ℓ^TT [2,10) × C_L^φφ [10,30)` at
r = +0.174 against a 95% null of 0.15, i.e. marginal.

This is a statement about sample size, not about physics. At the 168 pooled
draws that 12×600 sweeps supply after thinning by τ_int, the estimator throws
|r| ~ 0.15 on *uncorrelated* data — the same size as the effect being looked
for. Resolving |r| = 0.10 at 2σ needs ~2.4× this ensemble; |r| = 0.05 needs
~9.5×. A small correlation is also the physically expected outcome: `C_ℓ^TT` is
the *unlensed* spectrum and `C_L^φφ` depends on φ alone given φ, so the two
couple only through the data via the lensing likelihood.

Significance is against a within-chain permutation null (pairing destroyed,
marginals kept), and the error bars are a **chain-level** bootstrap — only the
12 chains are independent, not the 600 sweeps.

---

## Historical — pre-pivot dense-SHT era (2026-06-27)

Kept because the float32 result is the project's standing counterexample for the
fp64 discipline rule, not because these runs are current.

| Run | Precision | Accept | R-hat C_l (med/max) | ESS C_l | Status |
|---|---|---|---|---|---|
| lmax300 Gibbs | **float64** | 71% | 1.026 / 1.085 | 385 | C_l converged |
| lmax300 Gibbs | float32 | 38% | 1.000 / 1.001 | 1553 | alm **frozen** |
| lmax200 Gibbs | float32 | 64% | 1.000 / 1.001 | 1551 | alm **frozen** |
| lmax200 HMC | float32 | 64% | 2985 / — | 5 | diverged |
| lmax64 NUTS | float32 | 100% | 1.180 / — | 12.5 | underlength |

**The float32 trap, worth re-reading before any mixed-precision proposal:**
gradient noise in the SHT matmul accumulated across ~607k unmasked pixels and
drove the HMC step size to ~1e-7. Acceptance looked healthy at 38–65%, but the
chains moved <2e-6 in whitened alm space per step — effectively frozen. `C_l`
R̂ then read a *perfect* 1.000, because each chain sat in its own frozen alm
realisation and converged rapidly within that stuck mode. R̂ on one block is not
convergence.
