# Research Roadmap: Differentiable Bayesian CMB Analysis

*Forward-looking plan only. Completed/closed-out work is in `achievements.md`; positioning/novelty argument is in `literature.md`; full detail in git history.*

**The claim:** the first full-sky, curved-sky (HEALPix), differentiable joint Gibbs sampler over (alm_unlensed, C_ℓ, φ). Flat-sky joint sampling exists (CMBLensing.jl); full-sky methods are point-estimate or marginal (MUSE, QE). The window is finite (curved-sky MUSE could appear at any time) — the coverage test and the differentiator figures are the critical path; everything else waits.

**Why this scope.** A competing paradigm — diffusion/score-based generative lensing reconstruction — markets uncorrelated samples in ~0.2s and discards the two things this project built: a differentiable forward model and a sampler (`literature.md`). That sets the bar:

- The product is *demonstrated* exactness, not asserted exactness — a convincing coverage/rank test outranks any additional scale.
- The differentiator is the joint (C_ℓ, C_L^φφ) posterior with propagated correlations — an object no competing method (MUSE, QE, Commander, diffusion) produces.
- Position vs learned inference is offensive, not defensive: an exact sampler is the reference standard learned posteriors get validated against — but only as strong as the scale actually demonstrated.
- Deprioritised: CMBLensing.jl benchmark is a citation, not a science result; lmax scaling is not the route to impact.

**Positioning (decided 2026-08-06, full reasoning `literature.md`).** Broader scope, accepting scoop risk: the real-data Planck run and Phase 2b ΛCDM-parameter section are in scope, sequenced *after* the critical path below. Never lead with the differentiable machinery (Flinch made it table stakes) — lead with the joint (C_ℓ, C_L^φφ) posterior.

---

## Current state (2026-09-11)

The `Im(a_{L,1})` dof restoration (`k_L = 2L → 2L+1`) is merged into `main` and the headline SBC claim is confirmed under it. All three campaigns launched 2026-09-09 completed and were harvested 2026-09-11; full numbers in `achievements.md` and `results/analysis/dashboard.md`.

**What the harvest settled:**

1. **Headline SBC, now at N=24** (job 11965828 extending 11955622): φ 0.4688 (KS_p 0.0537), alm 0.5039 (KS_p 0.2319), both uniform. The lone flagged φ `[30,60)` bin was chased and **relaxed** rather than sharpening (KS_p 0.005→0.014, mean_u 0.260→0.339), with a second bin flagging weakly in the opposite direction — bin-level noise, closed as not a defect.
2. **The packing is excluded as the cause of the `[10,30)` residual** (job 11965813): strict `C_L^φφ` SBC rank 0.4245 (KS_p 0.00395) under the restored packing vs 0.3802 (0.0013) before. Marginally better, still a rejection, and **still localised to `[10,30)`** with φ over-powered there (median 1.162). Since the restoration changed both the alm dof and Blocks 1/4's shape and moved the rank barely at all, it is not the explanation — this retrospectively answers the Step 0 question the restoration plan skipped. Block 4's own PIT is a genuine pass (lag-10/50 controls rejected).
3. **lmax=128 is blocked solely by the low-L φ mode** (job 11966631, the first post-ordering-fix/post-restoration data point): NO-GO at worst lag-1 0.967 in `[2,10)`, *re-confirming* the pre-fix verdict rather than overturning it. Every bin above `[2,10)` decorrelates within 10–50 lags and the chain is otherwise healthy (φ accept 0.691, drift ≤0.60σ), so the failure is narrowly low-ℓ, not global.

**The one open defect is unchanged and now better isolated:** Block 3 (φ|alm,C_ℓ) conditioning in `ℓ∈[10,30)` under the Block-4-ON funnel. Packing, Block 4's conditional, trajectory length and the Nystrom mass matrix are all now excluded. Per the standing rule, **no further φ-equilibration tuning is launched without user sign-off** — this track has returned a negative or ambiguous result on nearly every attempt.

**Also found during the harvest (a real defect, not a result):** the lensing-blind `C_l^TT` baseline the bias-reduction figure depends on was silently a *different model* — `lensing_blind_baseline_lmax128.npz` (2026-08-12) is packed at the old 2L width (16254 vs `packed_length(128)`=16380) and predates both the ordering fix and the restoration. `ROADMAP` had it marked done. Differencing it against the post-fix lensing-aware chain would have rendered a packing artifact as physics.

**In flight:** job **11980570**, `scripts/submit_lensing_blind_baseline_packingv2.slurm` — re-running that baseline under the restored packing at the identical simulation (seed=0, lmax=128, nside=128, noisesig=1.0) to a new output path, ~3h. This is the remaining blocker on the `C_l^TT` bias-reduction figure.

## Next actions

1. **Harvest job 11980570** — check `.err`, confirm `alm_true_packed` is 16380 long (not 16254), then difference its `cl_samples` mean bin-by-bin against the lensing-aware chain `pilot_coverage_lmax128_postfix_hmc.npz` for the bias-reduction figure. Build the figure from the mid/high-ℓ bins; `[2,10)` is not usable from the lensing-aware side (~15 effective samples under the low-L mode).
2. **Decide the `[10,30)` Block 3 question** — it is now cleanly isolated and is the last thing between the project and a clean exactness story. Needs a decision, not another unilateral tuning run (see standing rule). The honest alternative is to **report it**: the headline Block-4-OFF SBC passes, and the Block-4-ON proper-prior configuration carries a documented, localised, one-bin caveat.
3. **More effective samples for the differentiator figure** — the joint (C_ℓ^TT, C_L^φφ) correlation is still a null at the current ensemble size, and resolving |r|=0.10 at 2σ needs ~2.4× the samples. Job 11965813's 12 chains are now available under the correct packing and can be pooled with any future Block-4-ON ensemble.

## Todo, priority order

### 1. Exactness evidence (highest value)
- [x] Multi-realization rank/coverage test at lmax=64 — DONE and reconfirmed under the restored packing (`achievements.md`).
- [x] `docs/paper/main.tex` updated to the confirmed numbers (both the 2026-08-31 dof-shape fix and the 2026-09-06 packing restoration).
- [~] **Resolve the Block 4 ON / proper-prior `C_L^φφ` strict SBC rank** — re-measured under the restored packing (0.4245, KS_p 0.00395, job 11965813): the residual survives and is confirmed localised to `[10,30)`; the packing is excluded as its cause. Remaining candidate is Block 3 conditioning there. Decision needed (Next actions §2), not another tuning run.
- [ ] Optional strengthening: push to lmax≈128 only if a referee asks — **blocked**: job 11966631 confirms the low-L φ mode post-fix (NO-GO, lag-1 0.967 in `[2,10)`). Per "demonstrated beats asserted", a passing test at 64 outranks a partial one at 128.

### 2. Differentiator figures (what the paper is *for*)
- [x] Joint (C_ℓ^TT, C_L^φφ) posterior correlation figure — built 2026-09-01, result is a null at current sample size (`achievements.md`). Next step is more effective samples (a dedicated long run, or pooling job 11903182 + job 11912088's post-thin draws), not a new estimator — not yet done.
- [ ] Per-mode uncertainty-propagation figure: what joint sampling buys over marginal methods.
- [~] C_ℓ^TT bias reduction vs a lensing-blind (Commander-style) analysis. Lensing-aware side **done** (job 11966631, restored packing). Lensing-blind reference was found on 2026-09-11 to be packed at the old 2L width and is **not** usable — re-run in flight as job 11980570. Build from mid/high-ℓ bins only.
- [ ] Write the position vs learned/amortised inference into the paper explicitly (intro + subsection) — the most likely referee question.
- [ ] Write the position vs **Flinch and Almanac** explicitly too — a second, separate referee question, answered by the φ/C_L^φφ block. Draft language and citations already in `main.tex`.

### 3. Related-work obligation (not a science result)
- [x] CMBLensing.jl comparison — written 2026-09-01 as `sec:cmblensing` in `main.tex`, from their published numbers. Costs shown side by side, not reduced to one ratio. Remaining trigger for actually installing CMBLensing.jl: a referee demanding same-realization posterior overlays. Design notes: `docs/notes/cmblensing_benchmark_notes.md`.

## 2. Real-data run — end-to-end demonstration

In scope for this paper. Supporting evidence, not the headline (the A_L anomaly that originally motivated it is no longer live per Planck PR4/ACT DR6). Pitch as either an A_L post-mortem on Planck 2018 vs PR4, or the joint posterior as a lensing-consistency test for SO/LiteBIRD-class data. Sequenced after §1/§2 above.

- [ ] Run the joint sampler on real Planck data; report the joint (C_ℓ, φ) posterior's lensing-consistency verdict.

## 2b. Phase 2b — ΛCDM parameters from C_ℓ

In scope for this paper. Routine, cheap robustness section — derive standard ΛCDM parameter constraints from the posterior C_ℓ chains once §1/§2 are done. Sequenced after those.

- [ ] Parameter-inference pass on the posterior C_ℓ^TT chains from the lmax≈128 (and, if run, real-data) chains; report against Planck/ACT/SPT baselines.

## 3. Phase 3 — polarization / LiteBIRD delensing (the science paper)

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
- [ ] Read Modrák et al. (`2211.02383`) before finalising the coverage-test design — per-ℓ-bin φ-power is a non-innocent test quantity and the residual is localised to one bin.
- [ ] Adopt nested R̂ (`2110.13017`) alongside rank-normalised split-R̂ (`1903.08008`); report τ_int as a lower bound citing `2408.13411`; cite the corrected (2017) Cook, Gelman & Rubin form.
- [ ] State the demonstrated scale as lmax=64 everywhere, with the comparison set (MUSE/Almanac/Bayer/FLI, all 10²-10⁴× larger) and the scaling route. Never lmax≈128.
- [ ] Add the "what the deficit is not" paragraph (not N0/N1, not mean-field, not non-Gaussian deflection, not foregrounds) — note the missing `Im(a_{ℓ,1})` dof is now *removed as a candidate* (restored 2026-09-06), not excluded by evidence.
- [ ] Locate an arXiv/proceedings version of the SFNO CMB-delensing paper (OpenReview `I8k3wwwm9l`) or drop it — currently `[UNVERIFIED]`.
- [ ] Still unverified before citing: author lists for `1708.06753`, `2111.07664`, `0708.2989`; Papež et al. 2018 and Huffenberger & Næss 2018 IDs; the Eriksen/Jewell/Wandelt 2004 Commander trio IDs.
- [ ] Re-run the named-author arXiv scan (Millea, Seljak, Bayer, Loureiro) and the citation-hygiene ID grep before every submission milestone (below).

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
- **A rank/coverage statistic needs its own simulated null before any flag is read as bias** — it can ranks the truth against its conditional's mode, which is non-uniform by construction even for a perfect sampler.
- **An intermittent test failure is a hypothesis, not a flake.**
- **Claims hygiene**: every "first" carries scope qualifiers and nearest-prior-work citations.
- **R-hat on C_ℓ alone is not convergence** — check the alm block and tail-ESS trends too.
- **No further φ-equilibration tuning without user sign-off** — this track has produced a negative or ambiguous result on almost every attempt (`achievements.md`); report and wait rather than launching the next idea unilaterally.
- **Cluster/storage operational rules** (job-submission caps, `/cosma8` quota, checkpoint placement, `$TMPDIR`): see the global `~/.claude/CLAUDE.md` COSMA entries and `achievements.md`'s "Engineering gotchas" — cluster-account facts, not project-plan facts.
