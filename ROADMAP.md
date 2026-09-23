# Research Roadmap: Differentiable Bayesian CMB Analysis

*Forward-looking plan only. What is done, retracted or closed is in `achievements.md`. Positioning and novelty argument: `literature.md`. Figure-by-figure caption requirements: `papers/7_DiffCMB/plots/STORY.md`.*

**The claim:** the first full-sky, curved-sky (HEALPix), differentiable joint Gibbs sampler over (a_ℓm unlensed, C_ℓ, φ, C_L^φφ), certified by simulation-based calibration. The title names the fourth block, so the evidence must be the **Block-4-ON** ensemble. Flat-sky joint sampling exists (CMBLensing.jl, including T+P in `1708.06753`); full-sky methods are point estimates or marginal (MUSE, QE). Nothing had appeared in the curved-sky joint cell as of 2026-09-22.

---

## Where the project stands (2026-09-23)

**Cautionary core:**
- **The legacy bias-reduction headline was an interpolation artefact.** The 93.7 % / 98.4 % came from the bilinear operator, not lensing (retracted; `achievements.md`).
- **Exactness is capped at lmax = 64.** The low-L φ mode fails at 128 and 192.
- **At lmax ≤ 300 a physical sky carries almost no lensing information in temperature.** The QE S/N on C_L^φφ is 0.002 at lmax = 64 and 0.13 at 300. The legacy figure 3 sat exactly there.

**Already positive:**
- **A four-block joint sampler that passed calibration on the legacy model**, which nobody else has. Certification on the exact-operator model is **not yet achieved**: it is the blocking item below (T0.1).
- **An exact, differentiable, geodesic curved-sky lensing operator**, with FD-checked gradients and physical sign conventions.
- **Exact-operator production ensembles** on identical skies, all complete (24/24 each: Block-4-ON, lensing-blind, Block-4-OFF).
- **Physics validated (figure 2):** on 24 same-sky pairs the blind fit lands on the lensing each sky received (−45.6 vs −46.0 % at ℓ ∈ [45,64)), while the joint fit stays within ~1–3 %.
- **A QE validated at that configuration**: noise 0.995, response 0.971.
- **A test suite that covers the scripts as well as the package.**

**⚠ Blocking as of the 2026-09-23 harvest:** the exact-operator ensembles do **not** pass calibration.
- Block-4-ON rejects on the joint-likelihood SBC (0.716, p = 0.001) and on φ/a_ℓm field ranks.
- Block-4-OFF passes the joint-likelihood and φ tests but fails a_ℓm `[30,60)`, the same bin that fails in the ON ensemble.
- Alm ESS is ~35 per 1200 sweeps.

Until this is diagnosed, **no exactness claim, no figure 1/3/maps result is citable** (`achievements.md` → First harvest).

**Handling the caution in the paper:**
- The retraction becomes one methods paragraph: validate an operator on the observable you report, not only its gradients.
- The lmax cap is reframed as *the regime where exact inference is the only certified option*.
- The amplified-lensing validation (A_φ = 3000, D4) is stated openly as a stress test.

## Positive routes (these organise T1)

1. **Exact operator, then measure the real thing — in progress.** The operator is done and the ensembles have landed. Do **not** rebuild the retracted headline on physical lensing: at these ℓ the physical C_ℓ^TT effect is +0.05–0.2 %. Figure 2 now shows a blind fit absorbing exactly the lensing each (amplified) sky received while the joint fit does not. Put the positive claim where exact inference is actually needed.
2. **Certify a learned posterior (T1.1): the highest-impact positive use.** Score a learned CMB-lensing posterior (`2603.04535`, or a trained NPE/diffusion model) against the exact one, including the correlation structure that marginals miss. Either it passes (the first certification of one) or it fails in a quantified way. In both cases the positive deliverable is **the benchmark itself**: the public lmax = 64 reference posteriors (T3.2).
3. **Science where exactness is certified: low-L lensing on Planck (T1.2b), conditional on scale.**
   - **The idea:** low-L C_L^φφ is where the QE has its largest reconstruction noise and mask/mean-field problems. Use that weakness as the claim: "an exact joint posterior on C_L^φφ at L < 30 from Planck temperature, where the QE is noise-dominated", on real data with the real mask.
   - **⚠ Feasibility check first.** Planck's low-L lensing information comes from temperature at ℓ ≳ 1000. At the certified lmax = 64 the physical S/N is 0.002, so the posterior would return the prior. The route needs the sampler at lmax ≳ 1000 (Parked: scaling), or a formulation where low-L φ is inferred from high-ℓ T.
   - **Scope this before building anything**, starting from what Planck's own low-L lensing analysis reports there.
4. **A_L post-mortem (T1.2)** as a real-data consistency test on Planck 2018 vs PR4, framed as a demonstration, not a competitive measurement.

## Decisions (all resolved)

- **D1:** one paper, all extensions included (2026-09-22).
- **D1b:** all figures final before any paper text (2026-09-22).
- **D2:** journal PRD (2026-09-22); it is the `paper_style.py` default.
- **D3:** exact lensing operator (2026-09-23).
- **D4:** amplified-lensing validation at A_φ = 3000, σ = 30 μK/pixel, lmax = 64 (2026-09-23, chosen by pilots; `achievements.md`).

---

## How this roadmap is ordered

**One paper, plots first, paper last.** No paper text (`main.tex`, drafts, stubs) until every figure for every section passes the checklist below.

| Tier | Purpose | Gate |
|---|---|---|
| **T0** | Core figures, from the exact-operator ensembles | every core figure passes the checklist |
| **T1** | Extension figures (positive routes), each validated to the core's standard on a branch/worktree | every extension figure passes |
| **T2** | Write the paper | draft complete and internally read |
| **T3** | Release, benchmark, submit | submitted |

**"Final form" checklist:**
1. built from the right, validated data, with the configuration checked (Block-4-ON for anything touching C_L^φφ; the exact operator; the QE validated at that configuration);
2. covers the claim it supports;
3. null or uncertainty drawn;
4. the demonstrated number in a stat box, and no other prose in the panel;
5. clear at printed size in PRD geometry/font;
6. reproducible by `make figures`;
7. caption requirements recorded in `plots/STORY.md`.

## T0 — Core figures from the exact-operator ensembles

*Done 2026-09-23 (recorded in `achievements.md`): the work was committed on branch `exact-operator-rerun` in both repos; all three ensembles harvested; `make figures` rebuilt all seven figures; the strict SBC, Block 4 PIT and joint-likelihood SBC ran on both ensembles (job 12040798).*

1. **⚠ BLOCKING — diagnose the calibration failure** before anything else touches the figures. The standing rules apply: a localised anomaly must survive more realizations *and* finer resolution *and* a calibrated test; and no φ-equilibration tuning without sign-off (this is a stationarity check first, not tuning). In order:
   - a. **Re-score at thin ≥ τ_int** (~35–40 for the alms), with `fig1_validation.py --thin 40` and `sbc_joint_likelihood.py --thin 40`, on both ensembles. Rank *means* are unbiased under autocorrelation *if the chain is stationary*, so a surviving mean offset indicts stationarity or the sampler, not the thinning.
   - b. **Stationarity:** first vs second half of each chain for a_ℓm power in `[30,60)`, φ power and log P. Does the offset shrink with sweep number (burn-in from the MAP start too short)?
   - c. If (b) shows drift: **longer chains** (e.g. 1000 burn-in + 2400 samples, ~11 h/task, 72 tasks packed) or discard more burn-in. If there is no drift: treat it as a sampler or statistic defect and check the a_ℓm power rank against `validate_coverage_rank_nulls.py`'s null-style reasoning for the field statistic.
   - d. If A_φ = 3000 cannot be made to mix, **pilot a lower A_φ** (e.g. 1000, QE S/N ≈ 1.8) and restate D4.
2. **Only after T0.1 passes: read the rebuilt figures against the checklist.**
   - **figure 1:** `p_bin` for both ensembles; the power curve at the final draws per chain.
   - **figure 2:** the aware residuals (`[2,10)` is −2.96 ± 1.05 % now); the A_L = 1 wording.
   - **figure 3:** the width ratio 0.61–0.74 must be re-derived from calibrated chains before it means anything.
   - **convergence:** re-derive `--thin` from τ_int.
   - **maps:** per-mode z sd (1.108 now, too narrow).
3. **Figure 1 and the Block-4-OFF ensemble:** decide whether the three-block certification gets its own panel or a caption number (the figure is currently built from Block-4-ON only).
4. **Power statement:** decide whether the paper needs a table as well as figure 1's curve; if so, generate it from the same script.
5. **Whole-set pass** at printed size: colours, ℓ-axis conventions, ensemble names.
6. **Keep `plots/STORY.md` and `results/analysis/dashboard.md` current** with every harvest number (both were updated 2026-09-23 with the first harvest).
7. **Merge `exact-operator-rerun` into `main` in both repos** once T0.1 is resolved, then **re-tag** the finalised core-figure state (the stale `methods-paper-v1` tag points at `d9979b7`).
8. **Small test gap:** freeze N_L's absolute normalisation at lmax = 16 in `tests/test_qe.py`.
9. **Limitations to carry into T2:**
    - exactness above lmax = 64 (low-L φ NO-GO at 128/192);
    - A_φ above ~3000 at lmax = 64 (Gibbs stalls);
    - physical-sky lensing information at lmax ≤ 300;
    - whatever T0.1 establishes about mixing at A_φ = 3000;
    - no further φ-equilibration tuning without sign-off.

## T1 — Extension figures (by impact per effort)

Each extension is validated to the core's standard (gate → production → calibrated check) on a branch or worktree. It delivers final-form figures plus caption requirements, not text. **Step one for each: which figures, supporting which claim.**

1. **Certify a learned CMB-lensing posterior against the exact one (route 2).**
   - Reuse the lmax = 64 Block-4-ON ensemble.
   - Train a conditional diffusion or NPE model on the same simulator (exact operator, A_φ = 3000, σ = 30), or use `2603.04535`'s if its code is available (read its body first for geometry).
   - Score it against the exact posteriors with SBC ranks, coverage and C2ST, including the (C_ℓ, C_L^φφ) and φ-mode correlations.
   - Likely figures: learned-vs-exact rank histograms with drawn nulls; per-mode width ratio; correlation-structure comparison.
2. **Real Planck data.**
   - **2a — A_L post-mortem (route 4):** Planck 2018 vs PR4. First the pipeline on a simulation carrying the real mask, `noise_map` and beam (note the beam path's prior fix, `achievements.md`), then data. Figures: real-data C_ℓ and C_L^φφ posteriors vs Planck spectra and the QE; the posterior-mean φ on the real mask.
   - **2b — low-L C_L^φφ (route 3):** only after its feasibility check; it needs lmax ≳ 1000 or a formulation using high-ℓ T.
3. **Polarization / LiteBIRD.**
   - Target the LiteBIRD lensing forecast (`2507.22618`).
   - Benchmark against `2511.21949` and `2608.06343`; the flat-sky T+P precedent is `1708.06753`.
   - **Scope the scale first:** the E-modes and φ that source low-ℓ B-modes sit at L of several hundred, beyond the certified scale, and the same physical-information check as route 3 applies.
   - Then:
     - [ ] ducc0 spin-2 lensing (the exact operator generalises);
     - [ ] TQU likelihood (C_ℓ^TE breaks conjugacy: 2×2 inverse-Wishart or HMC);
     - [ ] small-lmax TQU validation to the core's standard;
     - [ ] LiteBIRD-like simulation (delensing efficiency, r).
4. **ΛCDM parameters from the posterior C_ℓ.** One consistency figure (simulated and real-data chains vs Planck/ACT/SPT).

## T2 — Write the paper (only after the T0 and T1 gates)

1. Fix the structure from the finished figure set, then write `docs/paper/main.tex`:
   - a. Every exactness statement as a **bound** with its power, quoting the rebuilt figure 1's power curve.
   - b. Block-4-OFF and Block-4-ON certifications separated; Block-4-ON cited for the title's claim.
   - c. The amplified-lensing validation stated plainly: why (the physical S/N table), A_φ = 3000, σ = 30, and that the prior targets the same amplified spectrum.
   - d. The retraction as one methods paragraph, plus the lmax-cap reframing.
   - e. Positioning:
     - vs learned inference: `2603.04535`, concede `2606.12255`, cite `2606.10023`, then T1.1's result;
     - vs Flinch and Almanac (`2210.13260` for the masked sphere);
     - vs Darwish 2025;
     - the scale paragraph, where the comparison set is 10²–10⁴× larger and the answer is certified correctness, not scale.
   - f. Figure 3 near the front of the results if the rebuilt version supports a positive claim.
   - g. Limitations from T0.10.
2. Captions from `plots/STORY.md`.
3. Panels assembled into `figure*` environments, no rescaling past column width.
4. The literature actions below.

## T3 — Release and reach

1. **Tagged public code release** (GitHub `OscarHickman/diffcmb`) with a Zenodo DOI: README quick-start, `make test`, `make figures`.
2. **The benchmark:** the lmax = 64 exact reference posteriors (truth, data, configuration, thinned joint samples, T1.1's scoring script) on Zenodo. Check the quota rules before staging.
3. **Submit and post to arXiv together**, after a final named-author scan.
4. **Talks and outreach:** Durham/ICC, one external meeting, and announce the benchmark to the learned-inference groups it targets.

## Parked

- **lmax ≳ 1000 scaling (cuHPX `2510.01785`, cunuSHT `2406.14542`):** now the gate for routes 3 and T1.3, not just a platform item. The low-L φ mixing mode must be solved first; Millea et al.'s reparameterisation is the standing external lead.
- **Non-Gaussian extensions:** fNL, in-painting, learned priors, systematics.
- **Matrix-free HMC step-size re-tuning:** only if T1 chains show it matters.

## Literature actions (full annotations in `literature.md`)

- [ ] Re-check the full `main.tex` reference list for ID errors.
- [ ] Cite `2603.04535` as the lead competing-paradigm example, after reading its body for sky geometry; concede `2606.12255`.
- [ ] Cite Doeser & Jasche (`2606.10023`) in the introduction.
- [ ] Add `2210.13260` alongside `2305.16134`; correct the "all-sky, noiseless" characterisation of Almanac.
- [ ] Cite Modrák et al. (`2211.02383`) for the joint-likelihood test quantity.
- [ ] Rank-normalised R̂ (`1903.08008`) is adopted in the convergence figure: cite it, consider nested R̂ (`2110.13017`), cite `2408.13411` for ESS as an estimate, and cite the corrected (2017) Cook, Gelman & Rubin.
- [ ] Cite Millea, Anderes & Wandelt 2019 (`1708.06753`) alongside `2002.00965`; pin `\bibitem{Commander}` to the verified 2004 IDs.
- [ ] SFNO CMB delensing: still no arXiv ID; cite as OpenReview `I8k3wwwm9l` or drop.
- [ ] Before every milestone: named-author scan (Millea, Seljak, Bayer, Loureiro, Bonici; check Seljak and Bayer by hand, since the API author query missed them) and the reference-ID grep. **Monthly** while the paper is unposted.

## Standing discipline

- **One paper; plots first, paper last.** Unattended compute starts early; extensions are validated before they are written up, on a branch or worktree.
- **Never lead with the differentiable machinery** (Flinch). The novelty is the joint (C_ℓ, C_L^φφ) posterior as a capability; the evidence is validation. Don't sell a null as a detection or a validation as a discovery.
- **Name the configuration, not just the result:** blocks sampled, operator, fiducial, A_φ, noise. Every chain now records them and every replay reads them.
- **Validate a forward operator on the observable you report**, against an exact reference, not only its gradients against finite differences. The legacy operator's gradients were correct and it was still the wrong model.
- **Estimate the physical information content before interpreting a posterior.** A posterior that learns more than the data can contain is learning from the model's defects.
- **Pin sign and convention choices against an independent reference.** Two opposite sign errors (deflection and QE) cancelled and passed every internal consistency test.
- **Test the scripts, not only the package.** Metadata saved into the wrong call, replays through the wrong operator, and invalid pooled statistics all lived in `scripts/`. Mutation-check new tests by reintroducing the bug they target.
- **Watch authors, not only keywords** (Millea, Seljak, Bayer, Loureiro, Bonici).
- **Demonstrated beats asserted.** Exactness is certified at lmax = 64 only; never conflate scales in one sentence.
- **Read the script's own verdict and `.err`, not the SLURM state** (`COMPLETED 0:0` has hidden NO-GO verdicts and post-run crashes).
- **Precision:** fp64 end-to-end, including prior terms.
- **Dense or exact reference first:** validate every new sampler or operator against an exact small-scale reference before production.
- **A gate must run the production script's own initialisation path.**
- **Check calibration, not just mixing:** compare recovered quantities to their truth scale.
- **A PIT against the sampler's own conditional validates the draw, not the derivation;** a derivation needs an independent generative draw (proper prior).
- **Derive constants from the structure**, and grep every script that mirrors a derivation when fixing one.
- **A control that passes makes its check vacuous.**
- **Check array widths against `packed_length(lmax)`** before combining any pre-2026-09-06 product.
- **Spectrum comparisons reference S_ℓ/(k_ℓ−4)** under the flat prior, never S_ℓ/k_ℓ or the fiducial.
- **A localised anomaly is noise until it survives more realizations, finer rank resolution and a calibrated test.**
- **Calibrate a test at the granularity you use it,** and never pool correlated units as if independent (ℓ-bins of one chain). Report the power a pass carries.
- **A rank or coverage statistic needs its own simulated null** before a flag is read as bias.
- **An intermittent test failure is a hypothesis, not a flake.**
- **Figures:**
  - no prose inside a panel;
  - a figure must cover the object the title claims;
  - draw the null, don't just compute it;
  - a colour scale is an analysis choice;
  - a validated curve is validated at a configuration.
- **State what each side of a cross-check conditions on; compare like with like** (QE⊕prior, not raw N_L).
- **Claims hygiene:** every "first" carries scope qualifiers and nearest-prior-work citations.
- **R̂ on C_ℓ alone is not convergence;** check every block.
- **No further φ-equilibration tuning without user sign-off.**
- **Cluster operations** (job caps, quotas, `$TMPDIR`, packing TF tasks per node): see `~/.claude/CLAUDE.md` and `achievements.md` → Engineering gotchas.
