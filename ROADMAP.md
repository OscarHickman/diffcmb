# Research Roadmap: Differentiable Bayesian CMB Analysis

*Forward-looking plan only. What is done, retracted or closed is in `achievements.md`. Positioning and novelty argument: `literature.md`. Figure-by-figure caption requirements: `papers/7_DiffCMB/plots/STORY.md`.*

**The claim:** the first full-sky, curved-sky (HEALPix), differentiable joint Gibbs sampler over (a_ℓm unlensed, C_ℓ, φ, C_L^φφ), certified by simulation-based calibration. The title names the fourth block, so the evidence must be the **Block-4-ON** ensemble. Flat-sky joint sampling exists (CMBLensing.jl, including T+P in `1708.06753`); full-sky methods are point estimates or marginal (MUSE, QE). Nothing had appeared in the curved-sky joint cell as of 2026-09-22.

---

## Where the project stands (2026-10-03)

**Cautionary core:**
- **The legacy bias-reduction headline was an interpolation artefact.** The 93.7 % / 98.4 % came from the bilinear operator, not lensing (retracted; `achievements.md`).
- **Exactness is capped at lmax = 64.** The low-L φ mode fails at 128 and 192.
- **At lmax ≤ 300 a physical sky carries almost no lensing information in temperature.** The QE S/N on C_L^φφ is 0.002 at lmax = 64 and 0.13 at 300.

**Already positive** (details in `achievements.md`):
- **An exact, differentiable, geodesic curved-sky lensing operator**, with FD-checked gradients and physical sign conventions.
- **The adopted Block-4-ON ensemble (ν = 30, 1000 + 3600 sweeps, D5), N = 48, passes the joint-likelihood SBC (0.528, KS_p 0.45), the strict C_L^φφ SBC (pooled 0.495, KS_p 0.65), the Block 4 PIT (KS_p 0.22, controls rejected) and every φ field-rank bin (p_bin 0.386).**
- **Physics validated (figure 2, 48 same-sky pairs):** the blind fit lands on the lensing each sky received (−41.1 vs −41.5 % at ℓ ∈ [45,64)), while the joint fit stays within ~1 %.
- **A QE validated at that configuration**: noise 0.995, response 0.971.
- **A test suite that covers the scripts as well as the package.**

**⚠ Blocking: the a_ℓm `[60,64)` offset survived the fresh skies.** Posterior C_63 sits ~6 % below the φ-fixed exact reference. The offset appears at lmax = 64 only, in Block-4-ON and -OFF alike. The `--cl_init data` pilot (2026-10-03) ruled out the MAP start. The next test, a φ-fixed production run, needs sign-off (T0.1 a). Figure 1's a_ℓm row fails until then, so **no exactness claim is citable**.

**Handling the caution in the paper:**
- The retraction becomes one methods paragraph: validate an operator on the observable you report, not only its gradients.
- The lmax cap is reframed as *the regime where exact inference is the only certified option*.
- The amplified-lensing validation (A_φ = 3000, D4) is stated openly as a stress test.

## Positive routes (these organise T1)

1. **Measure where exact inference is needed.** Do **not** rebuild the retracted headline on physical lensing: at these ℓ the physical C_ℓ^TT effect is +0.05–0.2 %. Figure 2 already shows the blind fit absorbing the (amplified) lensing each sky received while the joint fit does not.
2. **Certify a learned posterior (T1.1): the highest-impact positive use.** Score a learned CMB-lensing posterior (`2603.04535`, or a trained NPE/diffusion model) against the exact one, including the correlation structure that marginals miss. It either passes (the first certification of one) or fails in a quantified way. In both cases the positive deliverable is **the benchmark itself**: the public lmax = 64 reference posteriors (T3.2).
3. **Science where exactness is certified: low-L lensing on Planck (T1.2b), conditional on scale.**
   - **The idea:** low-L C_L^φφ is where the QE has its largest reconstruction noise and mask/mean-field problems. The claim would be "an exact joint posterior on C_L^φφ at L < 30 from Planck temperature, where the QE is noise-dominated", on real data with the real mask.
   - **⚠ Feasibility check first.** Planck's low-L lensing information comes from temperature at ℓ ≳ 1000. At the certified lmax = 64 the physical S/N is 0.002, so the posterior would return the prior. The route needs the sampler at lmax ≳ 1000 (Parked: scaling), or a formulation where low-L φ is inferred from high-ℓ T.
   - **Scope this before building anything**, starting from what Planck's own low-L lensing analysis reports there.
4. **A_L post-mortem (T1.2)** as a real-data consistency test on Planck 2018 vs PR4, framed as a demonstration, not a competitive measurement.

Decisions D1–D5 (one paper, figures first, PRD, exact operator, A_φ = 3000 / σ = 30, ν = 30) are all taken; see `achievements.md` → Decisions taken.

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

*Done so far (details in `achievements.md`): the calibration diagnosis (T0.1 a–d: a_ℓm is the statistic, φ is funnel mixing, ν = 30 adopted as D5); the ν = 30 blind pair; figure 1 at thin 50; the figure 2 exact-sampler band; the figure 4 cell inspection; the N_L normalisation test; the N = 48 extension (r024–r047) with its harvest (job 12065737); the figure 1 power table; the band-edge diagnosis and localisation, with Block-4-OFF at N = 48 (2026-09-27/30); and the `--cl_init data` pilot (2026-10-03, negative). Sources: `logs/harvest_t01e_12065737.out` and `logs/alm_band_edge_*.out`; tables in `docs/dashboard.md`.*

1. **⚠ BLOCKING — the a_ℓm `[60,64)` offset: posterior C_63 sits ~6 % below the φ-fixed exact reference, at the band limit only.** Everything established so far is in `achievements.md` → "the a_ℓm `[60,64)` band-edge offset". In brief: it is ℓ = 63 alone. It is not pixelisation or the null, because the dense reference with φ = truth is clean. It occurs at lmax 64 only (lmax 32 is clean), and in Block-4-OFF too. **It is not the MAP start**: the `--cl_init data` pilot (2026-10-03) reaches the same late C_63 as production. A separate defect, C_ℓ collapse (r036's C_62, plus one Block-4-OFF sky), is seeded by the MAP's shrunk a_ℓm, and the data start does not prevent it.
   - **Next. Each step needs user sign-off for compute; step a also needs a small opt-in code change.**
     - a. **Separate φ marginalisation from the a_ℓm/C_ℓ blocks (recommended next).** Run the production Gibbs code with φ fixed at truth (Blocks 1 + 2 only), through a new opt-in flag in `coverage_ensemble_chain.py`, tested first, on the same skies as `exact_dense_alm_reference_l64_nu30` (r032–r039 at minimum).
       - If C_63 comes back at the dense value (mean 1.02 on r032–r039 excluding r036; 0.99 over 48), the a_ℓm/C_ℓ blocks are exact and the deficit comes from marginalising φ. Then either the exact φ-marginal posterior genuinely sits lower at the band edge, where lensing is largest (−41 % at `[45,64)`), so the φ-known null is the wrong reference; or a–φ mixing at the edge is too slow.
       - If it stays near 0.935, the defect is in the production a_ℓm block or the likelihood at ℓ = lmax − 1.
     - b. Why lmax 64 and not 32. The candidate is lensing strength at the band edge. Test it after a, with lmax 64 at a lower A_φ.
     - c. The W_ℓ²-corrected data start, C init from S_ℓ(a_MAP)/W_ℓ². The pilot says it will not fix C_63, but it should stop the collapse at the start. Use it only as a collapse guard.
     - d. Decide what to do with r036: drop it with a stated reason, or re-run it with a start that avoids the collapse. In the pilot it recovered by itself only in the last half.
   - Until then figure 1's a_ℓm row is not final and no exactness claim is citable. Quote per-mode sd(z) beside the rank: 1.03–1.08 at ν = 30, against 1.00 exact.
2. **Figure checklist read at N = 48** (rebuilt by job 12065737 from `_nu30_long`). Every number is in `achievements.md` → "N = 48 results" and `docs/dashboard.md`.
   - **figure 1:** φ and C_L^φφ pass; the a_ℓm row waits for T0.1.
   - **figure 2:** the exact-sampler band and the zoomed aware-only panel (c), `figure2/bias_aware_zoom.pdf`, are built. Panel (c) is drawn separately because an inset would be unreadable at 3.375 in. To do: the user looks at both; settle the A_L = 1 wording; re-read `[45,64)` after T0.1 (it is −2.97 against the effective expectation and probably shares the cause).
   - **figure 3:** the caption must say the φ prior is hierarchical at ν = 30.
   - **figure 4:** the caption reads "no correlation detected". Watch the one recurring cell, `C_ℓ[30,60)×C_L^φφ[10,30)`; it is not significant over 16 cells.
   - **convergence:** carry the R̂ and ESS numbers into the caption.
   - **maps:** do the checklist read.
   - Then write the caption numbers into `plots/STORY.md`. They are held back until T0.1 resolves.
3. **Whole-set pass** at printed size: colours, ℓ-axis conventions, ensemble names.
4. **Keep `plots/STORY.md` and `docs/dashboard.md` current** with every harvest number.
5. **Re-tag** the finalised core-figure state once T0.1 is resolved (`exact-operator-rerun` is already merged into `main`; check the paper repo too). The stale `methods-paper-v1` tag points at `d9979b7`.
6. **Housekeeping before reusing old GPU wrappers:** the 57 GPU wrappers must stay on dine2, which does not mount `/cosma5` (where `results/analysis` lives). They now fail at start, so relocate their outputs (e.g. to `/cosma7`) before running any of them.
7. **Limitations to carry into T2:**
    - exactness above lmax = 64 (low-L φ NO-GO at 128/192);
    - A_φ above ~3000 at lmax = 64 (Gibbs stalls);
    - physical-sky lensing information at lmax ≤ 300;
    - mixing at A_φ = 3000: low-L φ τ_int ≈ 280–430 sweeps; the centred (φ, C_L^φφ) hierarchy is not certifiable at ν = 6 in 3600 sweeps;
    - the ν = 30 prior carries 30 dof per L against the data's 2L+1, a third of the information at L = 30 (figure 3's QE⊕prior reference uses the ν = 30 prior);
    - whatever T0.1 concludes about a_ℓm at the band limit;
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
   - g. Limitations from T0.7.
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
