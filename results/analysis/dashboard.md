# Sampling & Validation Dashboard
*Last updated: 2026-09-08*

Live status of the production chains. Forward plan: `ROADMAP.md`. Closed-out
results and the bug record: `achievements.md`.

> **✅ 2026-09-08: the headline exactness claim is CONFIRMED under the
> restored `Im(a_{L,1})` packing** (`k_L = 2L` → `2L+1`, `achievements.md`).
> Job 11955622 (12 realizations, restored packing) reproduces job 11903181's
> pre-restoration result: φ 0.4792 / alm 0.5130, both uniform. Everything
> below marked "pre-restoration"/job ≤11913324 was measured through the old
> 2L packing (one fewer real dof per multipole) and is kept as historical
> reference, not current status.

---

## Current headline — simulation-based calibration

**lmax=64, nside=64, 12 independent chains, `phi_mass_matrix='prior'`, Block 4 OFF**
(job **11955622**, restored 2L+1 packing; supersedes job 11903181, pre-restoration).
Block 4 off pins `C_L^φφ` at the fiducial spectrum, so the φ prior is proper
*and identical to the process that generated the truth* — which is what makes
the φ rank a genuine calibration test.

| Field rank (pooled over 4 ℓ-bins, N=48) | mean_u | KS_p | verdict |
|---|---|---|---|
| φ | **0.4792** | **0.4078** | consistent with uniform |
| alm | **0.5130** | **0.6369** | consistent with uniform |

One flagged bin: φ `[30,60)` (mean_u 0.260, KS_p 0.005, N=12) — the same bin
the pre-restoration ensemble also flagged; read as a small-N artifact, not a
reopened defect (`achievements.md`). `--thin 90`, matching the pre-restoration
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
Block-4-ON comparison has not yet been re-run under the restored packing.

Mixing, pre-restoration runs: τ_int median 4.7–27 per bin with Block 4 off, vs
24–56 with it on. R̂ ≤ 1.07 outside the lowest and highest bins.

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

**Nothing.** Both the packing-v2 pilot (job 11951115) and the packing-v2
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

## Invalid output — do not aggregate or cite

| Directory | Job | Why |
|---|---|---|
| `coverage_ensemble_lmax64/` | 11848757 | Pre-dates the 2026-08-24 alm ordering fix |
| `coverage_ensemble_lmax64_prior_cl4/` | 11887897 | Cold-start alm; φ frozen 1e3–1e5× above truth |

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
