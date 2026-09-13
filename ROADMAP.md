# Research Roadmap: Differentiable Bayesian CMB Analysis

*Forward-looking plan only. Completed/closed-out work is in `achievements.md`; positioning/novelty argument is in `literature.md`; full detail in git history.*

**The claim:** the first full-sky, curved-sky (HEALPix), differentiable joint Gibbs sampler over (alm_unlensed, C_ℓ, φ). Flat-sky joint sampling exists (CMBLensing.jl); full-sky methods are point-estimate or marginal (MUSE, QE). The window is finite (curved-sky MUSE could appear at any time) — the coverage test and the differentiator figures are the critical path; everything else waits.

**Why this scope.** A competing paradigm — diffusion/score-based generative lensing reconstruction — markets uncorrelated samples in ~0.2s and discards the two things this project built: a differentiable forward model and a sampler (`literature.md`). That sets the bar:

- The product is *demonstrated* exactness, not asserted exactness — a convincing coverage/rank test outranks any additional scale. Since 2026-09-12 read "demonstrated" strictly: the honest form is a **bound** on undetected bias, stated with the test's power.
- The differentiator is the joint (C_ℓ, C_L^φφ) posterior with propagated correlations — an object no competing method (MUSE, QE, Commander, diffusion) produces.
- Position vs learned inference is offensive, not defensive: an exact sampler is the reference standard learned posteriors get validated against — but only as strong as the scale actually demonstrated.
- Deprioritised: CMBLensing.jl benchmark is a citation, not a science result; lmax scaling is not the route to impact.

**Positioning (decided 2026-08-06, full reasoning `literature.md`).** Broader scope, accepting scoop risk: the real-data Planck run and Phase 2b ΛCDM-parameter section are in scope, sequenced *after* the critical path below. Never lead with the differentiable machinery (Flinch made it table stakes). ⚠ The second half of this — "lead with the joint (C_ℓ, C_L^φφ) posterior" — is **under review as decision S1 below**, because that figure is a null while the bias reduction is a 93% effect.

---

## Current state (2026-09-13)

Detail and numbers in `achievements.md` and `results/analysis/dashboard.md`.

**Headline:** no bias detected at lmax=64, N=24 chains, now scored at **~60 rank draws/chain** where the test is properly calibrated — φ 0.4585 (KS_p 0.2766, cal_p 0.4038), alm 0.4956 (KS_p 0.9987, cal_p 0.7150), no bin flagged, joint-likelihood 0.4587 (KS_p 0.738). `KS_p` and `cal_p` now agree, which confirms the discreteness was the whole calibration problem. Still state it as a **bound**, not exactness — but a tighter one than before.

**Second result, now replicated:** `C_ℓ^TT` bias reduction **94.3% across 3 independent skies** (sd 1.3%), blind deficit −5.3±0.1% at `[100,128)` and monotone in ℓ on 3/3, aware within ±0.12% of unbiased everywhere. Seed 2 outstanding.

**There is no known open sampler defect.** The last one closed 2026-09-13: the strict `C_L^φφ` SBC rank went **KS_p 0.0009 → 0.0567** purely from resolving the rank at 60 draws instead of 8, with the mean essentially unchanged (0.4544 → 0.4518) and no bin rejecting. Together with the earlier N=24 and calibration findings, the "open sampling question" is closed as a measurement artifact.

**Still genuinely open:** lmax=128 is blocked for *calibration* work by the low-ℓ φ mode (a mixing limitation, not a correctness one — re-confirmed on all 3 replication skies, worst lag-1 0.964–0.975); the joint (C_ℓ, C_L^φφ) correlation is a null at achievable sample size; and the paper has not been updated for any of the last week's findings.

**In flight:** job **11984844_2** — the last bias-reduction replication sky (seed 2), ~30h elapsed of a 72h cap. On landing, re-run `scripts/submit_bias_reduction_seeds.slurm` (it skips absent seeds and reports how many it found) to publish the result as N=4.

## Next actions

1. **Harvest seed 2** and re-run the across-sky summary for the N=4 number.
2. **Update `docs/paper/main.tex`** — now the critical path. It predates the entire recalibration: its exactness claims read as demonstrated exactness rather than a bound, it quotes the miscalibrated `KS_p` values, and it does not carry the bias-reduction replication. See "Rank-test remediation" below.
3. **Decide S1** (below) — which result leads.
4. Then Todo §1–§2.

## Open decision — needs your sign-off

### S1. Lead with the bias reduction, not the (C_ℓ, C_L^φφ) correlation — RECOMMENDED

The standing instruction is "lead with the joint (C_ℓ, C_L^φφ) posterior." The evidence no longer supports that ordering:

| | joint (C_ℓ, C_L^φφ) correlation | C_ℓ^TT bias reduction |
|---|---|---|
| status | **null** — 1/16 cells vs 0.8 expected by chance | **93%**, 172σ vs 2σ |
| expected size | small *by construction* — C_ℓ^TT is the *unlensed* spectrum, coupling only through the data | large, growing with ℓ |
| cost to strengthen | ~2.4× ensemble for \|r\|=0.10, ~9.5× for 0.05 | done; N=4 replication in flight |
| legibility | needs the object explained first | "ignoring lensing costs 5% at ℓ~110; we remove it" |

Leading with a null because it is novel is the weaker play. **Proposed:** lead with the bias reduction as the demonstration of what joint sampling buys, and keep the joint posterior as a **capability claim** — "no competing method (MUSE, QE, Commander, diffusion) produces this object at all" — rather than a detection claim. More honest about the null, and more impactful.

**The vulnerability this creates, and the answer.** A referee will say "nobody analyses lensed data with an unlensed model." Both halves must be in the text: (i) Commander genuinely does not model lensing — a real, widely-used pipeline, not a strawman — so the claim is well-posed *against map-based Gibbs methods*; (ii) it must be stated that narrowly and **not** implied against Planck's cosmological likelihood, which uses lensed spectra and an A_L nuisance. Optional strengthening if pushed: add a lensed-template baseline (fit with the *lensed* spectrum, φ fixed), which isolates "we propagate φ uncertainty" from "we know about lensing at all".

**Cheap framing win:** the result is an A_L statement in disguise. "We recover A_L = 1 without a template or a nuisance parameter" is far more legible to the CMB community and costs only a rewrite.

## Todo, priority order

### 1. Exactness evidence
- [x] Raise draws/chain to ~60 and re-score — done 2026-09-13; tightened the bound and closed the strict `C_L^φφ` question (`achievements.md`).
- [ ] Report the power of the validation as a table in the paper. No competing method states the sensitivity of its own validation; doing so is a strengthening, not a concession.
- [x] Re-check the φ `[2,10)` bin at thin=10 — done: it **clears** (cal_p 0.015 → 0.401) and φ `[30,60)` takes its place (0.052 → 0.011). The flagged bin migrates with rank resolution, so neither is a live defect (`achievements.md`).
- [x] Joint-likelihood SBC on the Block-4-ON ensemble at ~120 draws/chain — done 2026-09-13: 0.4219 (KS_p 0.468), up from 0.3771 at thin=30, consistent with the low reading having been small-sample.
- [ ] Optional: lmax≈128 exactness only if a referee asks — **blocked** by the low-ℓ φ mode.

### 2. Differentiator figures (what the paper is *for*)
- [ ] Per-mode uncertainty-propagation figure: what joint sampling buys over marginal methods.
- [ ] Push the bias-reduction demonstration to lmax=256 — **decided 2026-09-12**: spend scale here, not on the null correlation. The effect *grows* with ℓ, and it needs only a converged C_ℓ marginal at mid/high ℓ — **not** the calibration gate that blocks lmax=128. Sequenced after the N=4 replication, since replication beats scale.
- [ ] Write the position vs learned/amortised inference into the paper explicitly (intro + subsection) — the most likely referee question.
- [ ] Write the position vs **Flinch and Almanac** explicitly too — a separate referee question, answered by the φ/C_L^φφ block. Draft language and citations already in `main.tex`.
- [ ] Decide whether to chase the (C_ℓ, C_L^φφ) correlation at all. Under S1 it becomes a capability claim and needs no detection; ~2.4× the ensemble buys at best a marginal one. Recommend not chasing.

## Phase 2a — real-data run (end-to-end demonstration)

In scope for this paper. Supporting evidence, not the headline (the A_L anomaly that originally motivated it is no longer live per Planck PR4/ACT DR6). Pitch as either an A_L post-mortem on Planck 2018 vs PR4, or the joint posterior as a lensing-consistency test for SO/LiteBIRD-class data. Sequenced after the Todo §1/§2 items above.

- [ ] Run the joint sampler on real Planck data; report the joint (C_ℓ, φ) posterior's lensing-consistency verdict.

## Phase 2b — ΛCDM parameters from C_ℓ

In scope for this paper. Routine, cheap robustness section — derive standard ΛCDM parameter constraints from the posterior C_ℓ chains once the Todo §1/§2 items are done. Sequenced after those.

- [ ] Parameter-inference pass on the posterior C_ℓ^TT chains from the lmax≈128 (and, if run, real-data) chains; report against Planck/ACT/SPT baselines.

## Phase 3 — polarization / LiteBIRD delensing (the science paper)

Full TQU joint analysis, after Phase 2 submits. Target reference: LiteBIRD lensing forecast (arXiv:2507.22618, QE/iterative pipeline — a sampling-based result fills a real gap). Benchmark against `2511.21949` (~47% delensing at 30≤ℓ≤300) and `2608.06343` (A_lens^res≈0.48), not only the CMB-S4 forecast.

- [ ] Spin-2 extension of alm utilities and the lensing operator (ducc0 spin-2 transforms).
- [ ] TQU joint likelihood (TT, TE, EE, BB); C_ℓ^TE breaks inverse-Gamma conjugacy → 2×2 inverse-Wishart or HMC.
- [ ] Simulated lensed TQU at LiteBIRD-like noise: delensing efficiency vs QE/iterative baselines, recovered r constraint.

## Parked (not started; recorded so the platform argument isn't lost)

- Phase 4 — lmax≥1000 scaling: tuning, not rearchitecture; profile only when Phase 2/3 need it. Scaling route if unparked: cuHPX (arXiv:2510.01785) or cunuSHT (arXiv:2406.14542).
- Phase 5 — non-Gaussian extensions (fNL, mask in-painting, learned priors, systematics): separate papers after Phases 2-3.
- Re-tune matrix-free-HMC step-size adaptation: current regime mixes ~4× less efficiently per-sample than the old dense-SHT reference. Skip unless Phase 2 chains show it matters.

## Literature actions (from the 2026-09-03 rescan; full annotations `literature.md`)

- [ ] Re-check the full `main.tex` reference list for further ID errors, the same way the three referee-visible ones (MUSE, CMBLensing/SPTpol, MCHMC title) were found and fixed.
- [ ] Cite `2603.04535` (learned CMB-delensing sampler) in the competing-paradigm section, replacing JADE as the lead example; read its body first to confirm sky geometry. Concede `2606.12255` honestly as the counterpoint (implicit/explicit field-level inference agreeing in a neighbouring problem).
- [ ] Cite Doeser & Jasche (`2606.10023`) in the introduction — external statement of why an exact reference posterior is needed.
- [ ] Add `2210.13260` (masked-sphere Almanac companion) alongside `2305.16134`; correct the draft's "all-sky, noiseless" characterisation of Almanac.
- [ ] Cite Modrák et al. (`2211.02383`) for the joint-likelihood test quantity, now adopted and passing (`achievements.md`). Reading it also falsified this item's original premise — it is about test-quantity *sensitivity*, not about valid quantities being non-uniform under a correct sampler.
- [ ] Adopt nested R̂ (`2110.13017`) alongside rank-normalised split-R̂ (`1903.08008`); report τ_int as a lower bound citing `2408.13411`; cite the corrected (2017) Cook, Gelman & Rubin form.
- [ ] State the demonstrated scale precisely: **exactness at lmax=64, lensing-bias removal at lmax=128** (the bias-reduction figure is a legitimate lmax=128 result — it needs no calibration gate). Give the comparison set (MUSE/Almanac/Bayer/FLI, all 10²-10⁴× larger) and the scaling route. Never claim *exactness* at lmax≈128.
- [ ] Add the "what the deficit is not" paragraph (not N0/N1, not mean-field, not non-Gaussian deflection, not foregrounds) — note the missing `Im(a_{ℓ,1})` dof is now *removed as a candidate* (restored 2026-09-06), not excluded by evidence.
- [ ] Locate an arXiv/proceedings version of the SFNO CMB-delensing paper (OpenReview `I8k3wwwm9l`) or drop it — currently `[UNVERIFIED]`.
- [ ] Still unverified before citing: author lists for `1708.06753`, `2111.07664`, `0708.2989`; Papež et al. 2018 and Huffenberger & Næss 2018 IDs; the Eriksen/Jewell/Wandelt 2004 Commander trio IDs.
- [ ] Re-run the named-author arXiv scan (Millea, Seljak, Bayer, Loureiro) and the citation-hygiene ID grep before every submission milestone (below).

## Rank-test remediation — remaining items

The rank test was found miscalibrated and underpowered on 2026-09-12; the fixes
already applied (`discrete_uniform_p`, `rank_spread`, both ensembles re-scored,
suite green) are recorded in `achievements.md`. What is left:

- [x] **Raise draws per chain from ~8 to ~60** — done 2026-09-13 (jobs 11986716,
      11986722). Closed the last open defect; `sd_u` 0.276–0.310 vs uniform's
      0.2887 shows the expected autocorrelation cost did **not** materialise, so
      `--thin 10` is the default for rank scoring from here (`achievements.md`).
- [ ] Re-state every claim in `docs/paper/main.tex` as a *bound* ("no bias
      detected above ~0.5σ posterior mean offset at N=24") rather than as
      demonstrated exactness. Pre-empts "what would your test have caught?"
- [ ] Add the power statement to the paper as a table (also listed under Todo §1).
- [ ] Re-read every pre-2026-09-12 flag and pass in `dashboard.md` against
      `cal_p` before citing any of them; `KS_p` is retained for continuity only.

## Standing discipline

- **One critical path**: Phase 2 gates ✓ → coverage/rank test → joint-posterior differentiator figures → paper. Anything not on this waits. The manuscript runs in parallel rather than at the end.
- **Scope**: broader scope, accepting scoop risk. Real-data Planck run and Phase 2b are in scope, sequenced after the critical path, not instead of it.
- **Never lead with the differentiable machinery.** Every abstract, talk, and intro leads with the joint (C_ℓ, C_L^φφ) posterior.
- **Watch authors, not only keywords.** Millea, Seljak, Bayer and Loureiro are the highest-probability source of a scoop; any φ or lensing extension of Flinch, Almanac, or CMBLensing.jl changes the plan. Named-author arXiv check + citation-hygiene grep before every submission milestone.
- **Demonstrated beats asserted.** Prefer a converged result at a smaller scale over a non-converged one at a larger scale — every time, and say which one you have.
- **Precision**: fp64 end-to-end unless a mixed scheme is validated against fp64 chains (a float32 false-convergence trap is the standing counterexample).
- **Dense-reference discipline**: validate every new sampler/operator against an exact small-scale reference before production — has caught real bugs repeatedly (`achievements.md`).
- **A gate must run the production script's own initialisation path.** A gate and the thing it gates being separate scripts is how the cold-start bug reached a full 12-chain ensemble (`achievements.md`). Diff their setup before trusting a verdict.
- **Check calibration, not just mixing.** R̂/ESS/τ_int measure whether a chain is moving, not whether it is in the right place — always also compare a recovered quantity against its known truth scale.
- **A PIT check against the sampler's own stated conditional validates the draw, not the derivation.** It passes for any shape parameter, since code and reference share the assumption. To test a derivation you need an independent generative draw, which for a spectrum block requires a proper prior.
- **Derive constants from the structure, don't hardcode them** — and when you do fix one, grep for every script that mirrors the same derivation (`achievements.md` — this slipped three times in one week before the lesson stuck).
- **A control that passes makes its check vacuous** — a misalignment/mutation control must itself fail before the aligned pass counts as evidence.
- **A saved `.npz` carries no packing version.** `PACKING_VERSION` guards checkpoints only. Before differencing or pooling any pre-2026-09-06 analysis product against a current chain, check its array widths against `packed_length(lmax)` — this caught the stale lensing-blind baseline that would otherwise have rendered a packing artifact as a physics result (`achievements.md`).
- **An unbiased `C_l` posterior does not centre on the truth's realized power.** Under the flat improper prior Block 1's posterior *mean* is `S_l/(k_l−4)`, not `S_l/k_l` — +15% at ℓ~20. Difference any spectrum comparison against `S_l/(k_l−4)`; against the realized power (or worse, the fiducial, which adds cosmic variance) the artifact is larger than the signal and reverses conclusions (`achievements.md`).
- **A localised anomaly is noise until it survives more realizations AND a finer rank resolution AND a calibrated test.** Three separate "ℓ-localised defects" on this project (φ `[30,60)` at N=12, the Block-4-ON `[10,30)`, φ `[2,10)` at thin=90) each dissolved under one of those, and the flagged bin demonstrably migrates between bins when the resolution changes. Weeks of Hessian-coupling and mass-matrix work were spent on the second one. Check all three before chasing.
- **Calibrate a goodness-of-fit test at the granularity you use it at, by feeding it output from a provably correct sampler and measuring the false-positive rate — before trusting either a flag OR a pass.** A continuous KS test on discrete ranks rejected a correct sampler 12.5% of the time at p<0.01 and drove a two-week investigation into a defect that was largely the test (`achievements.md`). And always report the POWER a pass carries: at N=24 ours could not see a 0.3σ posterior mean shift 3 times in 4, so a pass is a bound, not a proof.
- **A rank/coverage statistic needs its own simulated null before any flag is read as bias** — it can ranks the truth against its conditional's mode, which is non-uniform by construction even for a perfect sampler.
- **An intermittent test failure is a hypothesis, not a flake.**
- **Claims hygiene**: every "first" carries scope qualifiers and nearest-prior-work citations.
- **R-hat on C_ℓ alone is not convergence** — check the alm block and tail-ESS trends too.
- **No further φ-equilibration tuning without user sign-off** — this track has produced a negative or ambiguous result on almost every attempt (`achievements.md`); report and wait rather than launching the next idea unilaterally.
- **Cluster/storage operational rules** (job-submission caps, `/cosma8` quota, checkpoint placement, `$TMPDIR`): see the global `~/.claude/CLAUDE.md` COSMA entries and `achievements.md`'s "Engineering gotchas" — cluster-account facts, not project-plan facts.
