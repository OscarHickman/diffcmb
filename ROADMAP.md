# Research Roadmap: Differentiable Bayesian CMB Analysis

*Forward-looking plan only. Completed/closed-out work is in `achievements.md`; positioning/novelty argument is in `literature.md`; full detail in git history.*

**The claim:** the first full-sky, curved-sky (HEALPix), differentiable joint Gibbs sampler over (alm_unlensed, C_ℓ, φ, C_L^φφ). ⚠ The fourth block is in the claim, so it must be in the evidence — cite the Block-4-**ON** ensemble, not the Block-4-OFF one (see Current state). Flat-sky joint sampling exists (CMBLensing.jl); full-sky methods are point-estimate or marginal (MUSE, QE). The window is finite (curved-sky MUSE could appear at any time) — the coverage test and the differentiator figures are the critical path; everything else waits.

**Why this scope.** A competing paradigm — diffusion/score-based generative lensing reconstruction — markets uncorrelated samples in ~0.2s and discards the two things this project built: a differentiable forward model and a sampler (`literature.md`). That sets the bar:

- The product is *demonstrated* exactness, not asserted exactness — a convincing coverage/rank test outranks any additional scale. Since 2026-09-12 read "demonstrated" strictly: the honest form is a **bound** on undetected bias, stated with the test's power.
- The differentiator is the joint (C_ℓ, C_L^φφ) posterior with propagated correlations — an object no competing method (MUSE, QE, Commander, diffusion) produces.
- Position vs learned inference is offensive, not defensive: an exact sampler is the reference standard learned posteriors get validated against — but only as strong as the scale actually demonstrated.
- Deprioritised: CMBLensing.jl benchmark is a citation, not a science result; lmax scaling is not the route to impact.

**Positioning (decided 2026-08-06, full reasoning `literature.md`).** Broader scope, accepting scoop risk: the real-data Planck run and Phase 2b ΛCDM-parameter section are in scope, sequenced *after* the critical path below. Never lead with the differentiable machinery (Flinch made it table stakes). **Ordering settled 2026-09-16 (was decision S1):** the bias reduction leads as *validation*, the joint (C_ℓ, C_L^φφ) posterior carries the *novelty* as a capability claim with an honestly-reported null. Full reasoning and the referee-objection language in `achievements.md` → Positioning (settled).

---

## Current state (2026-09-16)

Full detail and numbers in `achievements.md` and `results/analysis/dashboard.md` — this section is a pointer, not a restatement.

**Headline exactness (3-block core):** no bias detected at lmax=64, N=24 chains, 60 rank draws/chain — stated as a **bound**, not exactness (`achievements.md`). This is the **Block-4-OFF** ensemble (jobs 11955622 + 11965828): it certifies (a_ℓm, C_ℓ, φ) only.

**Headline exactness (the title's object):** the full (a_ℓm, C_ℓ, φ, **C_L^φφ**) sampler is certified separately by the **Block-4-ON**, ν=6 proper-prior N=24 ensemble (jobs 11965813 + 11980637) — no bin rejects, pooled strict rank 0.4518 (KS_p 0.0567) at `--thin 10`. ⚠ **The paper must cite this one for any joint-sampler claim, not the Block-4-OFF numbers** — see `achievements.md`'s "Which configuration this certifies".

**Headline effect, replicated at N=4:** `C_ℓ^TT` lensing-bias reduction **93.7% ± 1.8% across 4 independent skies** (job 11991514) — aware beats blind 4/4, deficit deepens monotonically with ℓ 4/4. Per S1 this is presented as **validation**, not as the discovery.

**No known open sampler defect** — the last one (strict `C_L^φφ` SBC rank) closed 2026-09-13 as a measurement-resolution artifact.

**In flight: nothing.** Job 11987444 (lmax=192) completed 2026-09-15T22:25; the queue holds no diffcmb jobs.

**Still genuinely open:**
- The paper (`docs/paper/main.tex`) has not been touched since **2026-09-08** and predates the entire rank-test recalibration. This is the critical path.
- lmax=128 *and* lmax=192 are blocked for **calibration** work by the low-ℓ φ mixing mode (a mixing limit, not a correctness one). At 192 the gate returned **NO-GO** at lag-1 0.976 — so **no exactness claim above lmax=64 is available**, and none should be attempted.
- The lmax=192 **bias-reduction** harvest has not been run (below) — it is the one piece of unclaimed value sitting on disk.
- The joint (C_ℓ, C_L^φφ) correlation is a null at achievable sample size; per S1 this is now accepted, not a gap to close.

## Figure state

`papers/7_DiffCMB/plots/` holds the scoped figure sequence, tagged **`methods-paper-v1`** in both repos — the known-good methods-paper state and the revert point if Phase 2a stalls. Built: figure1 (validation), figure2 (bias reduction, N=4), figure4 (joint posterior / capability claim). **Not built: figure3** (per-mode uncertainty propagation vs QE) — deliberately not fabricated; `plots/STORY.md` says what it needs. Details in `achievements.md` → Paper deliverables.

**Phase 2a discipline:** the real-data run touches real Planck data, foregrounds and masking, none of which the validated simulation pipeline has been checked against. **Do it on a branch or worktree, never on `main`**, so a stalled attempt is abandoned with `git worktree remove` and `main` (== `methods-paper-v1`) is untouched. Fallback if it fails: submit the methods paper as scoped — nothing in the checkpoint depends on Phase 2a landing.

## Next actions — in order

1. **Harvest the lmax=192 bias-reduction figure.** `compare_cl_bias_reduction.py --lmax 192` against `results/analysis/pilot_coverage_lmax192_hmc.npz` and `lensing_blind_baseline_lmax192.npz`; bins extend to `[128,160)`, `[160,192)`. Cheap (both chains are on disk) and either strengthens figure2 or is dropped. **Read it with the low-ℓ caveat: `[2,10)` is UNRELIABLE as at 128, and at 192 that now extends to `[10,30)`** (lag-1 0.928, decorrelation lag 200). Do not present any lmax=192 *exactness* number.
2. **Rewrite `docs/paper/main.tex`** — the critical path, now unblocked (S1 is decided). Four things, in this order:
   - a. Recast every exactness statement as a **bound** ("no bias detected above ~0.5σ posterior mean offset at N=24"), not as demonstrated exactness.
   - b. **Separate the Block-4-OFF and Block-4-ON certifications explicitly**, and cite Block-4-ON for the joint claim in the title.
   - c. Replace the miscalibrated `KS_p` values with the thin=10 / `cal_p` numbers.
   - d. Carry the N=4 bias-reduction replication, framed per S1 as validation + the A_L framing.
3. **Add the power table** to the paper (§1 below). No competing method states the sensitivity of its own validation; doing so is a strengthening, not a concession.
4. Then the literature/citation actions below, then Phase 2a on a branch.

## Todo, priority order

### 1. Exactness evidence
- [ ] Report the power of the validation as a table in the paper — **this is Next action 3**; kept here so the section is self-contained.
- [ ] Optional: lmax≈128 exactness only if a referee asks — **blocked** by the low-ℓ φ mode, and now confirmed blocked at lmax=192 too (gate NO-GO, lag-1 0.976, job 11987444). Treat "exactness at lmax=64" as the demonstrated scale and say so; do not attempt to raise it for this paper.

### 2. Differentiator figures (what the paper is *for*)
- [ ] Per-mode uncertainty-propagation figure: what joint sampling buys over marginal methods.
- [ ] Write the position vs learned/amortised inference into the paper explicitly (intro + subsection) — the most likely referee question.
- [ ] Write the position vs **Flinch and Almanac** explicitly too — a separate referee question, answered by the φ/C_L^φφ block. Draft language and citations already in `main.tex`.

*Closed out of this section and recorded in `achievements.md`:* the lmax=192 scale-up (landed 2026-09-15, gate NO-GO — harvest is Next action 1) and the decision not to chase the (C_ℓ, C_L^φφ) correlation (S1, 2026-09-16).

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
(`discrete_uniform_p`, `rank_spread`, raising draws/chain to ~60, both ensembles
re-scored) are done and recorded in `achievements.md`. What is left:

- [→] Re-state every claim in `docs/paper/main.tex` as a *bound* rather than as
      demonstrated exactness — **now Next action 2a**; pre-empts "what would your
      test have caught?"
- [→] Add the power statement to the paper as a table — **now Next action 3**.
- [ ] Re-read every pre-2026-09-12 flag and pass in `dashboard.md` against
      `cal_p` before citing any of them; `KS_p` is retained for continuity only.

## Standing discipline

- **One critical path**: Phase 2 gates ✓ → coverage/rank test → joint-posterior differentiator figures → paper. Anything not on this waits. The manuscript runs in parallel rather than at the end.
- **Scope**: broader scope, accepting scoop risk. Real-data Planck run and Phase 2b are in scope, sequenced after the critical path, not instead of it.
- **Never lead with the differentiable machinery** (Flinch made it table stakes). Since S1 (2026-09-16): the **novelty** claim is the joint (C_ℓ, C_L^φφ) posterior, stated as a *capability* no competing method produces; the **evidence** that leads is the bias reduction, presented as validation. Don't sell a null as a detection, and don't sell a validation as a discovery.
- **Name the configuration, not just the result.** "Headline exactness" meant the Block-4-*OFF* ensemble for two weeks while the paper's title claimed a four-block joint sampler — the certification of the title's object existed, under a different job pair, and was simply not labelled as such. Every exactness number in the paper or the dashboard carries which blocks were sampled, or it is not citable.
- **Watch authors, not only keywords.** Millea, Seljak, Bayer and Loureiro are the highest-probability source of a scoop; any φ or lensing extension of Flinch, Almanac, or CMBLensing.jl changes the plan. Named-author arXiv check + citation-hygiene grep before every submission milestone.
- **Demonstrated beats asserted.** Prefer a converged result at a smaller scale over a non-converged one at a larger scale — every time, and say which one you have. **The demonstrated exactness scale for this paper is lmax=64 and will not rise**: the low-ℓ φ mixing mode failed the gate at 128 and again, worse, at 192 (lag-1 0.976). Bias *reduction* is separately demonstrable at 128 (and possibly 192) because it needs no calibration gate — never conflate the two scales in a sentence.
- **A scale-up that lands is not a scale-up that passes.** Job 11987444 ran 2d03h and exited `COMPLETED 0:0` while printing NO-GO. Read the script's own verdict line, not the SLURM state (the general form of this is already in `achievements.md`'s engineering gotchas).
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
