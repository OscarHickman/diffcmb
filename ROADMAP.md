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

## Current state (2026-09-08)

The `Im(a_{L,1})` dof restoration (`k_L = 2L → 2L+1`) is merged into `main`. Re-validated at production scale (pilot job 11951115, then 12-chain ensemble job 11955622): headline SBC **CONFIRMED** under the restored packing — φ mean_u 0.4792 (KS_p 0.4078), alm 0.5130 (KS_p 0.6369), both uniform, superseding the pre-restoration pair (0.4688/0.5312). `docs/paper/main.tex` is updated onto these numbers. Full detail: `achievements.md`.

**What's still open, and predates the dof restoration:** with Block 4 ON and a proper `C_L^φφ` prior, the strict SBC rank does not clear (0.42, localised to ℓ∈[10,30)) — see `achievements.md`'s "Open sampling question" section for the full investigation (Hessian-coupling diagnosis, the falsified Nystrom-mass-matrix fix). None of that has been re-measured under the 2L+1 packing yet.

**In flight (2026-09-09):**
- Job 11965813, `scripts/submit_coverage_ensemble_lmax64_prior_cl4_properprior_packingv2.slurm` — 12-realization re-run of job 11903182's config (lmax=64, ν=6, `phi_n_lfs=240`, Block 4 ON) under the restored 2L+1 packing. ~5h/realization, 24h walltime. Compare the strict `C_L^φφ` SBC rank against 11903182's 0.3802/KS_p=0.0013.
- Job 11965828, `scripts/submit_coverage_ensemble_lmax64_prior_nocl4_packingv2_extendN.slurm` — extends the headline packing-v2 ensemble (job 11955622) from N=12 to N=24 realizations, appending into the same output dir, to chase the flagged φ `[30,60)` bin (KS_p=0.005 at N=12).
- Job 11966631, `scripts/submit_pilot_coverage_lmax128_postfix_hmc.slurm` — the lensing-aware lmax=128 chain needed for the C_ℓ^TT bias-reduction figure, run for the first time since the 2026-08-24 alm-ordering fix and 2026-09-06 dof restoration. **Caveat:** every prior lmax=128 φ-equilibration verdict (including the "genuine, low-L-specific long-lived mode" closure in `achievements.md`) predates the ordering fix and is unconfirmed post-fix; this run is the first post-fix data point on that question, not a re-confirmation. HMC explicitly (the script's own default is the closed-NO-GO `mclmc`), `phi_mass_matrix='prior'`, `phi_n_lfs=240`, seed=0/lmax=128/nside=128/noisesig=1.0 matching `lensing_blind_baseline_lmax128.npz` for direct comparability. 72h walltime, checkpoint every 50 sweeps.

Harvest instructions for all three are in their script headers.

## Next actions

1. **Harvest job 11965813** — check `.err` per array task, confirm φ power ratio O(1), then run `validate_coverage_rank_nulls.py` and `aggregate_coverage_ranks.py`. Re-establishes the Block-4-ON proper-prior configuration under the restored packing; needed before any further Block 3 investigation, since every existing number for this configuration was pre-restoration.
2. **Harvest job 11965828** — re-run `aggregate_coverage_ranks.py` over all 24 realizations, compare the φ `[30,60)` bin's KS_p at N=24 against the N=12 value.
3. **Harvest job 11966631** — read the GO/NO-GO equilibration verdict as the first post-fix lmax=128 data point (not a re-confirmation of the pre-fix one), then extract posterior C_ℓ^TT and compare against `lensing_blind_baseline_lmax128.npz` for the bias-reduction figure.
4. Otherwise proceed to the priority task list below.

## Todo, priority order

### 1. Exactness evidence (highest value)
- [x] Multi-realization rank/coverage test at lmax=64 — DONE and reconfirmed under the restored packing (`achievements.md`).
- [x] `docs/paper/main.tex` updated to the confirmed numbers (both the 2026-08-31 dof-shape fix and the 2026-09-06 packing restoration).
- [ ] **Resolve the Block 4 ON / proper-prior `C_L^φφ` strict SBC rank** — see Next actions §1 above.
- [ ] Optional strengthening: push to lmax≈128 only if a referee asks. Per "demonstrated beats asserted", a passing test at 64 outranks a partial one at 128.

### 2. Differentiator figures (what the paper is *for*)
- [x] Joint (C_ℓ^TT, C_L^φφ) posterior correlation figure — built 2026-09-01, result is a null at current sample size (`achievements.md`). Next step is more effective samples (a dedicated long run, or pooling job 11903182 + job 11912088's post-thin draws), not a new estimator — not yet done.
- [ ] Per-mode uncertainty-propagation figure: what joint sampling buys over marginal methods.
- [~] C_ℓ^TT bias reduction vs a lensing-blind (Commander-style) analysis. Lensing-blind reference chain done (`achievements.md`, `results/analysis/lensing_blind_baseline_lmax128.npz`); the lensing-aware side still needs the equilibrated joint chain.
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
- **A rank/coverage statistic needs its own simulated null before any flag is read as bias** — it can ranks the truth against its conditional's mode, which is non-uniform by construction even for a perfect sampler.
- **An intermittent test failure is a hypothesis, not a flake.**
- **Claims hygiene**: every "first" carries scope qualifiers and nearest-prior-work citations.
- **R-hat on C_ℓ alone is not convergence** — check the alm block and tail-ESS trends too.
- **No further φ-equilibration tuning without user sign-off** — this track has produced a negative or ambiguous result on almost every attempt (`achievements.md`); report and wait rather than launching the next idea unilaterally.
- **Cluster/storage operational rules** (job-submission caps, `/cosma8` quota, checkpoint placement, `$TMPDIR`): see the global `~/.claude/CLAUDE.md` COSMA entries and `achievements.md`'s "Engineering gotchas" — cluster-account facts, not project-plan facts.
