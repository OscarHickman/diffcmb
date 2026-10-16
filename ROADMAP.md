# Research Roadmap: Differentiable Bayesian CMB Analysis

*Forward-looking plan only. What is done, retracted or closed is in `achievements.md`. Positioning and novelty argument: `literature.md`. Figure-by-figure caption requirements: `papers/7_DiffCMB/plots/STORY.md`.*

**The claim:** the first full-sky, curved-sky (HEALPix), differentiable joint Gibbs sampler over (a_ℓm unlensed, C_ℓ, φ, C_L^φφ), certified by simulation-based calibration. The title names the fourth block, so the evidence must be the **Block-4-ON** ensemble. Flat-sky joint sampling exists (CMBLensing.jl, including T+P in `1708.06753`); full-sky methods are point estimates or marginal (MUSE, QE). Nothing had appeared in the curved-sky joint cell as of 2026-09-22.

---

## Where the project stands (2026-10-14)

**Cautionary core:**
- **The legacy bias-reduction headline was an interpolation artefact.** The 93.7 % / 98.4 % came from the bilinear operator, not lensing (retracted; `achievements.md`).
- **Exactness is capped at lmax = 64.** The low-L φ mode fails at 128 and 192.
- **At lmax ≤ 300 a physical sky carries almost no lensing information in temperature.** The QE S/N on C_L^φφ is 0.002 at lmax = 64 and 0.13 at 300.

**Already positive** (numbers in `achievements.md`):
- an exact, differentiable, geodesic curved-sky lensing operator, with FD-checked gradients and physical sign conventions;
- the adopted Block-4-ON ensemble (ν = 30, D5, N = 48) passes the joint-likelihood SBC, the strict C_L^φφ SBC, the Block 4 PIT and every φ field-rank bin;
- **figure 1 (D7) is the strict SBC of all four blocks**, with no simulated null: C_ℓ^TT, a_ℓm (band edge included) and C_L^φφ pass; φ fails in `[30,60)` (`p_bin` 0.014), which is stated as a limitation;
- figure 2's physics: the blind fit lands on the lensing each sky received, the joint fit stays within ~1 %;
- a learned (NPE) posterior that is calibrated and unbiased against the exact one, and 3–55 % wider (T1.1);
- a QE validated at the production configuration;
- a test suite that covers the scripts as well as the package.

**Nothing is blocking.** T0.1 is closed by D7 (2026-10-14; `achievements.md` → T0.1). The exactness claim is the strict SBC with its stated power: three blocks pass, and φ `[30,60)` is a stated limitation.

Decisions D1–D8 are taken (`achievements.md` → Decisions taken).

---

## How this roadmap is ordered

**One paper, plots first, paper last.** No paper text (`main.tex`, drafts, stubs) until every figure for every section passes the checklist below.

| Tier | Purpose | Gate |
|---|---|---|
| **T0** | Core figures, from the exact-operator ensembles | every core figure passes the checklist |
| **T1** | Extension figures, each validated to the core's standard on a branch/worktree | every extension figure passes |
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

*Done items are in `achievements.md`; live tables in `docs/dashboard.md`.*

1. **Figure checklist read at N = 48.** Figures 2–4, maps and convergence come from `_nu30_long` (job 12065737); figure 1 comes from `_nu30_strict` (D7, rebuilt 2026-10-14). Every number is in `achievements.md` and `docs/dashboard.md`.
   - **figure 1 (D7):** the user looks at the new layout (`figure1/rank_histograms.pdf`, `clpp_sbc.pdf`, `cltt_sbc.pdf`). The caption states: the strict design (truths drawn from the sampler's prior, no null); C_ℓ^TT 1.00, a_ℓm 0.178, C_L^φφ 0.185 pass, φ 0.014 fails in `[30,60)`; the 0.28σ power; the paired flat-prior numbers (`figure1_flatprior`); and the φ-fixed certification of the a_ℓm/C_ℓ blocks.
   - **figure 2:** the user looks at the exact-sampler band and the zoomed panel (c), `figure2/bias_aware_zoom.pdf`; settle the A_L = 1 wording; the caption says the `[45,64)` z of −2.97 is against an expectation that conditions on known φ (the T0.1 cause), and that r036 is kept (D8; dropping it moves the bins by ≤ 0.26 points).
   - **figure 3:** the caption must say the φ prior is hierarchical at ν = 30.
   - **figure 4:** the caption reads "no correlation detected". Watch the one recurring cell, `C_ℓ[30,60)×C_L^φφ[10,30)`; it is not significant over 16 cells.
   - **convergence:** carry the R̂ and ESS numbers into the caption.
   - **maps:** do the checklist read.
   - Then write the caption numbers into `plots/STORY.md`, now unblocked.
2. **Whole-set pass** at printed size: colours, ℓ-axis conventions, ensemble names.
3. **Keep `plots/STORY.md` and `docs/dashboard.md` current** with every harvest number.
4. **Re-tag** the finalised core-figure state on `main` after the checklist read; check the paper repo too. The stale `methods-paper-v1` tag points at `d9979b7`.
5. **Limitations to carry into T2:**
    - exactness above lmax = 64 (low-L φ NO-GO at 128/192);
    - A_φ above ~3000 at lmax = 64 (Gibbs stalls);
    - physical-sky lensing information at lmax ≤ 300;
    - mixing at A_φ = 3000: low-L φ τ_int ≈ 280–430 sweeps; the centred (φ, C_L^φφ) hierarchy is not certifiable at ν = 6 in 3600 sweeps;
    - the ν = 30 prior carries 30 dof per L against the data's 2L+1, a third of the information at L = 30 (figure 3's QE⊕prior reference uses the ν = 30 prior);
    - φ `[30,60)` fails the strict SBC at ν = 30 (mean_u 0.626, `p_bin` 0.014) and sits at 0.56 under the flat C_ℓ prior on the same φ truths: residual mixing of the centred (φ, C_L^φφ) hierarchy. Fixing it is φ-sampler work, which needs sign-off;
    - against a φ-known reference the a_ℓm band edge sits 6–9 % low at ℓ = 63. That deficit is stationary, and the strict SBC passes there;
    - no further φ-equilibration tuning without sign-off.

## T1 — Extension figures (by impact per effort)

Each extension is validated to the core's standard (gate → production → calibrated check) on a branch or worktree. It delivers final-form figures plus caption requirements, not text. **Step one for each: which figures, supporting which claim.**

1. **Certify a learned CMB-lensing posterior against the exact one: the highest-impact positive use.** Either outcome is publishable, and the deliverable is **the benchmark itself**: the public lmax = 64 reference posteriors (T3.2).
   - **Done (`achievements.md` → T1.1, D6):** the spectra-level NPE passes held-out SBC and, on the 48 strict skies, is unbiased against the exact posterior and wider: 3–24 % for C_ℓ^TT and 16–55 % for C_L^φφ, growing with ℓ. The correlation structure agrees to within 0.11.
   - **Done (2026-10-15, D9): all three width-gap routes failed to close the gap.** Seven variants all sit within ±0.03 of the baseline's ratios; the compressor is worse and fails SBC. The baseline stays the headline; the residual is the information cost of the summaries.
   - **Done (2026-10-16): learning curve flat** (50k/100k/200k give the same ratios), so the gap is summary-limited and more sims won't help. Only a field-level or richer-sufficient-statistic posterior could narrow it (D6 deferred the field-level model).
   - **Done (2026-10-16): C2ST** (job 12173707, `scripts/npe_c2st.py`; `achievements.md` → T1.1). Learned vs exact is separable from width and shift (raw accuracy 0.765), and near chance once both are removed (0.47 against a standardised null 0.44). The φ-mode correlations cannot be scored: the NPE is spectra-level and has no φ field; the bin-level correlation structure agrees to 0.12 (above). A full-space C2ST is out of reach (ESS 115–3500 per bin).
   - Likely figures: per-mode width ratio, learned against exact (the headline); rank histograms with drawn nulls; correlation-structure comparison. The strict SBC passes at ℓ ≥ 60 (D7), so the band edge can be pooled; report it separately as well.
2. **Real Planck data.**
   - **2a — A_L post-mortem:** a real-data consistency test on Planck 2018 vs PR4, framed as a demonstration, not a competitive measurement. First the pipeline on a simulation carrying the real mask, `noise_map` and beam (note the beam path's prior fix, `achievements.md`), then data. Figures: real-data C_ℓ and C_L^φφ posteriors vs Planck spectra and the QE; the posterior-mean φ on the real mask.
   - **2b — low-L C_L^φφ, conditional on scale.** The claim would be "an exact joint posterior on C_L^φφ at L < 30 from Planck temperature, where the QE is noise-dominated" (largest reconstruction noise, mask and mean-field problems), on real data with the real mask.
     - **⚠ Feasibility check first.** Planck's low-L lensing information comes from temperature at ℓ ≳ 1000. At the certified lmax = 64 the physical S/N is 0.002, so the posterior would return the prior. It needs the sampler at lmax ≳ 1000 (Parked: scaling), or a formulation that infers low-L φ from high-ℓ T.
     - Scope it before building anything, starting from what Planck's own low-L lensing analysis reports there.
3. **Polarization / LiteBIRD.**
   - Target the LiteBIRD lensing forecast (`2507.22618`).
   - Benchmark against `2511.21949` and `2608.06343`; the flat-sky T+P precedent is `1708.06753`.
   - **Scope the scale first:** the E-modes and φ that source low-ℓ B-modes sit at L of several hundred, beyond the certified scale, and the same physical-information check as 2b applies.
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
     - vs learned inference: `2603.04535`, concede `2606.12255`, keep `2606.10023` (already cited), then T1.1's result;
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

- **lmax ≳ 1000 scaling (cuHPX `2510.01785`, cunuSHT `2406.14542`):** now the gate for T1.2b and T1.3, not just a platform item. The low-L φ mixing mode must be solved first; Millea et al.'s reparameterisation is the standing external lead.
- **Non-Gaussian extensions:** fNL, in-painting, learned priors, systematics.
- **Matrix-free HMC step-size re-tuning:** only if T1 chains show it matters.

## Literature actions (full annotations in `literature.md`)

- [ ] Fix the bibitem metadata found by the 2026-10-07 reference check (`literature.md` → Citation corrections, items 10–17: wrong titles/years and the `Drouin:2023` placeholder; every arXiv ID is right).
- [ ] Cite `2603.04535` as the lead competing-paradigm example (body read 2026-10-07: flat 32×32 patches, Gaussian stand-in for lensing, no φ; `literature.md`); concede `2606.12255`; consider `2406.19367` (PUREPath, curved-sky learned CMB posterior without lensing).
- [ ] Add `2210.13260` alongside `2305.16134`; correct the "all-sky, noiseless" characterisation of Almanac.
- [ ] Cite Modrák et al. (`2211.02383`) for the joint-likelihood test quantity.
- [ ] Cite rank-normalised R̂ (`1903.08008`), which the convergence figure uses; consider nested R̂ (`2110.13017`), cite `2408.13411` for ESS as an estimate, and cite the corrected (2017) Cook, Gelman & Rubin.
- [ ] Cite Millea, Anderes & Wandelt 2019 (`1708.06753`) alongside `2002.00965`; pin `\bibitem{Commander}` to the verified 2004 IDs.
- [ ] SFNO CMB delensing: still no arXiv ID; cite as OpenReview `I8k3wwwm9l` or drop.
- [ ] Before every milestone: named-author scan (Millea, Seljak, Bayer, Loureiro, Bonici; use the arXiv author-search page for Seljak and Bayer, since the API author query misses them; last run 2026-10-07, nothing relevant, next due ~2026-11-07) and the reference-ID grep. **Monthly** while the paper is unposted.

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
- **Cluster operations** (job caps, quotas, `$TMPDIR`, packing TF tasks per node, which partitions mount `/cosma5`): see `~/.claude/CLAUDE.md` and `achievements.md` → Engineering gotchas. Before reusing an old GPU wrapper, move its output off `/cosma5`: dine2 does not mount it.


---

## Thesis-level priorities and real-data routes (added 2026-10-15)

*From `../PAPERS.md` §9 and the 2026-10-15 thesis review. Proposed, not authorised: nothing here changes the order of the work above. It records what this project contributes to the thesis and the concrete real-data steps that exist. Literature named here is cited as already recorded in this repo's `literature.md` or `../PAPERS.md`; anything new needs an `arxiv.org/abs` check first.*

- **Thesis role.** §2b Tier 2 #8, outside the clustering spine; its thesis value is as the "exact reference"
  leg of the validation barrier. Commit and re-tag first (last code commit `8f8a0f5`, 2026-10-07).
- **Real data: T1 item 2a is the realistic route.** The A_L consistency test on Planck 2018 vs PR4, framed
  as a demonstration. Steps already set: the pipeline on a simulation carrying the real mask, `noise_map`
  and beam, then the data. A quick check found no Planck maps in this repo (`data/` does not exist);
  confirm where the maps and masks will come from before scoping.
- **2b (low-L C_L^φφ) stays behind its feasibility check:** at lmax = 64 the posterior would return the
  prior.
