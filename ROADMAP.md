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

1. ~~Harvest job 11980570~~ **DONE 2026-09-11** — baseline re-ran clean (4 min, not the ~3h the header guessed: no φ block is 0.1 s/sweep), `alm_true_packed` verified at 16380, and the bias-reduction figure is **built and positive** (93% reduction; see §2 below). Remaining polish: the `[2,10)` bin dominates the figure's y-scale — the paper version should drop it rather than grey it.

2. **Harvest job 11980637** (launched 2026-09-11) — extends the Block-4-ON proper-prior ensemble from N=12 to N=24, the same small-N-vs-real-defect discriminator job 11965828 just applied to the Block-4-OFF `[30,60)` flag. Serves two ends at once: it doubles the chain count for the joint (C_ℓ^TT, C_L^φφ) differentiator figure (currently a null — the error bar is a *chain*-level bootstrap, so more chains is the only lever), and it tests whether the `[10,30)` strict-rank residual relaxes or sharpens at N=24. Config byte-identical to 11965813, so this is not a φ-tuning run. Harvest per the script header.

3. **Decide the `[10,30)` Block 3 question** — it is now cleanly isolated and is the last thing between the project and a clean exactness story. Needs a decision, not another unilateral tuning run (see standing rule). The honest alternative is to **report it**: the headline Block-4-OFF SBC passes, and the Block-4-ON proper-prior configuration carries a documented, localised, one-bin caveat.


## Impact strategy (2026-09-12) — proposed, needs your sign-off on §S1

The 2026-09-11 harvest changed what the strongest evidence is, and the stated
positioning no longer matches it. Four decisions follow; S1 is genuinely yours,
S2–S4 I have acted on.

### S1. Lead with the bias reduction, not the (C_ℓ, C_L^φφ) correlation — RECOMMENDED, your call

The standing instruction is "never lead with the differentiable machinery; lead
with the joint (C_ℓ, C_L^φφ) posterior." The evidence no longer supports that
ordering:

| | joint (C_ℓ, C_L^φφ) correlation | C_ℓ^TT bias reduction |
|---|---|---|
| status | **null** — 1/16 cells vs 0.8 expected by chance | **93%**, 172σ vs 2σ |
| physically expected size | small *by construction* — C_ℓ^TT is the *unlensed* spectrum and couples to C_L^φφ only through the data | large and growing with ℓ |
| cost to strengthen | ~2.4× ensemble for \|r\|=0.10, ~9.5× for 0.05 | already done; replication in flight |
| legibility to a referee | requires explaining what the object even is | "ignoring lensing costs you 5% at ℓ~110; we remove it" |

Leading with a null because it is novel is the weaker play. **Proposed:** lead
with the bias reduction as the demonstration of *what joint sampling buys*, and
keep the joint (C_ℓ, C_L^φφ) posterior as a **capability claim** — "no competing
method (MUSE, QE, Commander, diffusion) produces this object at all" — rather
than a detection claim. That is both more honest about the null and more
impactful. It also *decouples* the headline from the one open defect (see S2).

**The main vulnerability this creates, and the honest answer.** A referee will
say: "nobody analyses lensed data with an unlensed model." Two-part response,
both of which must be in the text: (i) Commander genuinely does not model
lensing — this is a real, widely-used pipeline, not a strawman, so the claim is
well-posed *against map-based Gibbs methods*; (ii) the claim must be stated
that narrowly, and must **not** be implied against Planck's cosmological
likelihood, which uses lensed spectra and an A_L nuisance. Optional
strengthening if a referee pushes: add a lensed-template baseline (fit with the
*lensed* spectrum, φ fixed) — that isolates "we propagate φ uncertainty" from
"we know about lensing at all", which is the sharper claim.

**Framing bonus, cheap:** the result is an A_L statement in disguise — "we
recover A_L = 1 without a template or a nuisance parameter" is far more legible
to the CMB community than a fractional-bias table, and costs only a rewrite.

### S2. The `[10,30)` residual: REPORT it — decided, and N=24 has since DISSOLVED the localisation

Rationale, now that the causes are enumerated: packing (excluded, job 11965813),
Block 4's conditional (excluded — PIT is a genuine pass), trajectory length
(excluded at 240 and 480), Nystrom mass matrix (falsified at two ranks). What
remains is Block 3 mixing, and that track has returned a negative or ambiguous
result on nearly every attempt. Under S1 the affected configuration (Block 4 ON)
supports a *capability* claim, not the headline, so a documented, localised,
one-bin caveat is proportionate. The N=24 extension (job 11980637) may yet
relax it exactly as it relaxed the `[30,60)` flag — **wait for that harvest
before writing the caveat**, but do not launch further φ work either way.

**That harvest landed 2026-09-12 and the localisation is gone.** At N=24 the
`[10,30)` bin moved 0.2812 → 0.4219 (KS_p 0.0047 → 0.137) and **no bin rejects
individually**; what remains is a small (~0.045) *global* downward offset whose
pooled KS_p is computed by an explicitly anti-conservative test. The new
joint-likelihood statistic agrees and attributes the difference to the
*configuration*: Block-4-ON 0.4028/0.3771 vs Block-4-OFF 0.4569/0.4417, i.e.
posterior draws fitting the data slightly better than the truth — mild
under-dispersion from incomplete φ mixing, with Block 4's conditional itself
exact (PIT genuine pass). **So the caveat to write is "a small global
under-dispersion in the Block-4-ON funnel", not "a localised `[10,30)`
defect"** — and note in the paper that the localised framing was a small-N
artifact, since the Hessian-coupling and Nystrom work was aimed at it.
Full numbers: `results/analysis/dashboard.md`.

**Modrák et al. `2211.02383` was read 2026-09-12 and cuts the OPPOSITE way to
the assumption recorded here.** The paper is *"SBC Checking for Bayesian
Computation: The Choice of Test Quantities Shapes Sensitivity"* (Modrák, Moon,
Kim, Bürkner, Huurre, Faltejsková, Gelman, Vehtari). Its thesis is that the
choice of test quantity governs **how sensitive SBC is to problems**, *not* that
some test quantities are non-uniform under a correct sampler. Under exact
posterior sampling every valid test quantity ranks uniformly. So per-ℓ-bin
φ-power is **not** excused as a "non-innocent" statistic, and the `[10,30)`
non-uniformity should be read as what the project already concluded it is — a
genuine, bounded, localised *computational* (mixing) deficiency, since MCMC with
finite chains is not exact posterior sampling. Report it as such; do not explain
it away. Delete the old "non-innocent test quantity" note — it was wrong.

### S3. Adopt the joint likelihood as an SBC test quantity — DONE 2026-09-12, and it PASSES

Modrák's central practical recommendation is that data-dependent test
quantities, **the joint likelihood especially**, detect failures that
parameter-wise ranks miss (including the posterior-equals-prior failure mode,
which parameter ranks cannot see). **Correction to this item as first written:** I claimed the saved `logp` made
it "a rank computation over existing files — no new sampling". That was wrong.
`logp` is `−psi` (the alm-block log-posterior given the chain's φ), not a
likelihood, and the truth's value is not saved; the data map is not saved
either. It needed a new script that replays the generative path. No new
*sampling* was required, which is what made it cheap, but it was not free.

**Result: `scripts/sbc_joint_likelihood.py`, run on the headline N=24
Block-4-OFF ensemble — consistent with uniform and thin-robust** (mean_u 0.4569,
KS_p 0.738 at `--thin 10`; 0.4417, KS_p 0.468 at `--thin 30`). All 24 replays
verified against the saved truth to 1e-12. Details and the CAMB
non-reproducibility gotcha the verification exposed: `achievements.md`.

Remaining: run the same statistic on the Block-4-ON ensemble once job 11980637
lands, where it is the more interesting test — that is the configuration with
the open `[10,30)` residual, and a data-dependent quantity is the one most
likely to say something the φ-power ranks cannot.

### S4. Spend scale on the bias reduction, not on the null — decided

lmax=64 exactness / lmax=128 bias reduction against a comparison set 10²–10⁴×
larger is the biggest referee target. The right place to spend is the
**bias-reduction** demonstration, because (i) the effect *grows* with ℓ (−5.4%
already at `[100,128)`), so lmax=256 makes the figure stronger, not merely
bigger; and (ii) it needs only a converged C_ℓ marginal at mid/high ℓ, where φ
mixes fine — it does **not** need the calibration gate that blocks lmax=128 for
SBC. Pushing the null correlation instead buys, at best, a marginal detection of
a quantity that is small by construction. Sequenced after the N=4 replication
lands, since replication beats scale (standing discipline: demonstrated beats
asserted).

## Todo, priority order

### 1. Exactness evidence (highest value)
- [x] Multi-realization rank/coverage test at lmax=64 — DONE and reconfirmed under the restored packing (`achievements.md`).
- [x] `docs/paper/main.tex` updated to the confirmed numbers (both the 2026-08-31 dof-shape fix and the 2026-09-06 packing restoration).
- [~] **Resolve the Block 4 ON / proper-prior `C_L^φφ` strict SBC rank** — re-measured under the restored packing (0.4245, KS_p 0.00395, job 11965813): the residual survives and is confirmed localised to `[10,30)`; the packing is excluded as its cause. Remaining candidate is Block 3 conditioning there. Decision needed (Next actions §2), not another tuning run.
- [ ] Optional strengthening: push to lmax≈128 only if a referee asks — **blocked**: job 11966631 confirms the low-L φ mode post-fix (NO-GO, lag-1 0.967 in `[2,10)`). Per "demonstrated beats asserted", a passing test at 64 outranks a partial one at 128.

### 2. Differentiator figures (what the paper is *for*)
- [x] Joint (C_ℓ^TT, C_L^φφ) posterior correlation figure — built 2026-09-01, result is a null at current sample size (`achievements.md`). Next step is more effective samples (a dedicated long run, or pooling job 11903182 + job 11912088's post-thin draws), not a new estimator — not yet done.
- [ ] Per-mode uncertainty-propagation figure: what joint sampling buys over marginal methods.
- [x] C_ℓ^TT bias reduction vs a lensing-blind (Commander-style) analysis — **DEMONSTRATED 2026-09-11: 93% bias reduction** (mean |fractional bias| 0.0226 → 0.0016 over the four reliable ℓ bins; lensing-aware consistent with unbiased everywhere, blind deficit growing to −5.4% at `[100,128)`). `scripts/compare_cl_bias_reduction.py`, figure `results/analysis/figures/cl_bias_reduction_lmax128.png`. `[2,10)` excluded (φ not equilibrated at lmax=128). Note the reference had to be the *expected posterior mean* `S_l/(k_l−4)`, not the realized power and not the fiducial — both wrong references reversed the conclusion (`achievements.md`). Both sides done under the restored packing: lensing-aware job 11966631, lensing-blind re-run job 11980570 (the 2026-08-12 baseline was old-2L-packed and unusable).
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
- [x] Read Modrák et al. (`2211.02383`) — **done 2026-09-12, and it falsified the premise of this item**: the paper is about test-quantity *sensitivity*, not about valid test quantities being non-uniform under a correct sampler. Per-ℓ-bin φ-power is not excused; the `[10,30)` residual stands as a real mixing deficiency (see Impact strategy §S2). Actionable consequence is §S3: adopt the joint likelihood as a test quantity.
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
- **An unbiased `C_l` posterior does not centre on the truth's realized power.** Under the flat improper prior Block 1's posterior *mean* is `S_l/(k_l−4)`, not `S_l/k_l` — +15% at ℓ~20. Difference any spectrum comparison against `S_l/(k_l−4)`; against the realized power (or worse, the fiducial, which adds cosmic variance) the artifact is larger than the signal and reverses conclusions (`achievements.md`).
- **A rank/coverage statistic needs its own simulated null before any flag is read as bias** — it can ranks the truth against its conditional's mode, which is non-uniform by construction even for a perfect sampler.
- **An intermittent test failure is a hypothesis, not a flake.**
- **Claims hygiene**: every "first" carries scope qualifiers and nearest-prior-work citations.
- **R-hat on C_ℓ alone is not convergence** — check the alm block and tail-ESS trends too.
- **No further φ-equilibration tuning without user sign-off** — this track has produced a negative or ambiguous result on almost every attempt (`achievements.md`); report and wait rather than launching the next idea unilaterally.
- **Cluster/storage operational rules** (job-submission caps, `/cosma8` quota, checkpoint placement, `$TMPDIR`): see the global `~/.claude/CLAUDE.md` COSMA entries and `achievements.md`'s "Engineering gotchas" — cluster-account facts, not project-plan facts.
