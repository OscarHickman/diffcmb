# Literature — DiffCMB (Paper 7)

*Annotated bibliography with the novelty/positioning argument folded in. IDs added or re-checked this pass carry `[v 2026-09-03]`; older entries were verified in earlier passes.*

**Last checked: 2026-09-03.** Method: ~14 targeted web searches across the eight axes below, plus arXiv advanced-search enumeration (abstract field, astro-ph + cross-lists) for `lensing` over 2026-08-01→2026-09-04 and `"field-level"` over 2026-06-01→2026-09-04, and the `astro-ph.CO` 2026-08 (377 entries) and 2026-09 (20 entries) listings. **The arXiv API (`export.arxiv.org`) returned HTTP 429 throughout and was not used** — the previous pass's exhaustive API enumeration back to 2026-03 could not be repeated, so coverage rests on the search UI and keyword queries and is narrower than 2026-08-05's. Two advanced-search queries also 429'd.

**No `.bib` to reconcile against yet.** `papers/7_DiffCMB/` has no manuscript content. The live draft is `diffcmb/docs/paper/main.tex`, using inline `\bibitem`s — **three referee-visible citation defects in it were found this pass** (below).

---

## Verdict

**The core claim survives: no curved-sky joint (a_ℓm, C_ℓ, φ, C_L^φφ) sampler exists, and no curved-sky MUSE exists. Scoop risk: LOW.** Supports:

- The `"field-level"` enumeration for 2026-06→2026-09 returns **no CMB entry at all** — every cosmology hit is LSS, galaxy weak lensing, or 21cm.
- The `lensing` enumeration for August 2026 returns exactly one CMB-lensing analysis, `2608.31136` (SPT-3G D1), and it is a **quadratic estimator**. The other two August CMB-lensing papers are a QE template paper and an instrument paper. September (through 09-02) has nothing on-topic.
- Named-author watch (Millea, Seljak, Bayer, Loureiro): **no φ or lensing extension of Flinch, Almanac or CMBLensing.jl**.

**Must engage, in order:**

1. **Flinch `2510.26691`** — unchanged as the most consequential entry. It occupies the *Commander cell* differentiably; no φ, no C_L^φφ. Differentiable curved-sky machinery is table stakes: **never lead with the SHT engineering.**
2. **`2603.04535` — a learned posterior sampler now exists for CMB delensing itself.** The competing-paradigm section previously had to reach to galaxy weak lensing (JADE) for its sharpest example. It no longer does.
3. **Doeser & Jasche `2606.10023`** — the external statement of why an exact reference posterior is needed. Still uncited in the introduction.

**On the demonstrated scale (lmax=64, not 128).** A referee will ask, and **nothing found this pass states a defensible minimum scale** — no threshold exists in the literature. What does exist is the comparison set, and it is unflattering on raw dimension: MUSE ~6×10⁶ latents (`2112.09354`), Almanac 1.68×10⁷ parameters (`2210.13260`), Bayer et al. ~2.6×10⁵ (`2307.09504`), practical weak-lensing FLI 8×10⁶ (`2606.12255`). DiffCMB's lmax=64 is ~4×10³ real dof per field. **The defence cannot be "this is large" — it must be "this is the first demonstration that the joint posterior is sampled *correctly*, at a scale where correctness can be certified."** State the scale plainly, state the SBC result, give the scaling route (`2510.01785` cuHPX, `2406.14542` cunuSHT) as future work. Never write lmax≈128.

---

## Citation corrections (this pass)

Found by grepping `main.tex` for arXiv IDs and resolving each on its abstract page.

1. **MUSE is cited as `2112.09091`, which is "Dualities in one-dimensional quantum lattice models"** (Lootens, Delcamp, Ortiz & Verstraete) — an unrelated quantum-physics paper. Correct: **`2112.09354`** (Millea & Seljak, PRD 105, 103531). `[v 2026-09-03]`
2. **CMBLensing/SPTpol is cited as `2012.00011`, which is "Mass-gap Mergers in Active Galactic Nuclei"** (Tagawa et al.). Correct: **`2012.01709`** (Millea et al. 2021, ApJ 922, 259). `[v 2026-09-03]`
3. **`2212.08549` is titled "Microcanonical *Hamiltonian* Monte Carlo", not "…Langevin…"** (Robnik, De Luca, Silverstein & Seljak; v1 2022-12, v3 2026-05). MCLMC appears inside it as a continuous variant. ID and authors are right; the title string is wrong. `[v 2026-09-03]`

Still standing from earlier passes: Carron & Lewis MAP lensing is **`1704.08230`**, not `1701.01712`; **`2209.10512`** is geometry-agnostic MUSE-via-implicit-differentiation, not a flat-sky MUSE follow-up.

---

## The 2×2: {flat-sky, curved-sky} × {marginal/point-estimate, joint φ sampling}

Cell (curved-sky, joint φ sampling) is empty. Everything below occupies one of the other three.

**Flat-sky, joint sampling — nearest prior work**

- **Millea, Anderes & Wandelt 2020**, `2002.00965`, PRD 102, 123542 — flat-sky joint (f, φ, r, A_φ) sampling; CMBLensing.jl. **The nearest prior work, full stop.** Their result that naive block alternation mixes catastrophically and the fix is reparameterisation, not compute, is the standing external reference for DiffCMB's Block 3 problem.
- **Millea et al. 2021**, `2012.01709`, ApJ 922, 259 — CMBLensing.jl on real SPTpol data; A_φ = 0.949 ± 0.122, 17% smaller errors than their own QE. **ID corrected this pass.**
- **Anderes, Wandelt & Lavaux 2015**, `1412.4079` — the Gibbs-over-(field, φ) ancestor. **`1708.06753`** — same lineage, flat-sky; *author list unverified*.

**Curved-sky, marginal or point-estimate**

- **MUSE**, `2112.09354`, PRD 105, 103531 — ~6×10⁶ latents; not a sampler, no joint posterior. Their "curved-sky HMC is slightly out of reach" defines this project's empty cell. **No curved-sky MUSE has appeared.** **`2209.10512`** — MUSE via implicit differentiation, geometry-agnostic. **`2411.06000`** — production MUSE on SPT-3G data.
- **Carron & Lewis 2017**, `1704.08230`, PRD 96, 063510 — iterative MAP lensing (LensIt); a point estimate.
- **Belkner, Carron et al. 2023**, `2310.06729` — `delensalot`, CMB-S4 iterative internal delensing, 92–93% B-lensing power removed. **The curved-sky state of the art that is not a sampler.**
- **Darwish 2025**, `2503.03682` — optimal *joint* MAP over multiple line-of-sight distortion fields (lensing + birefringence + patchy screening), extending `delensalot`. Joint over *distortion fields*, a point estimate, no C_ℓ or C_L^φφ block. Draw the distinction explicitly — a referee may raise the word "joint" as prior art.
- **Iterative/QE frontier**: `2407.00228` (non-Gaussian deflections, PRD 110, 103520); `2506.20667` (noise-bias-minimising iterative estimator); `2605.18659` (control variates, ~5× cheaper RD bias).
- **SPT-3G D1 QE lensing**, `2608.31136` (Omori et al., 2026-08-31) — **new.** Lensing amplitude to 2% of ΛCDM, Σm_ν < 0.072 eV (95%), σ₈Ω_m^0.25 = 0.6046 ± 0.0096. No field-level, MAP or sampling content. **This is the scoop check's headline negative: the flagship 2026 lensing analysis is still a quadratic estimator.** `[v 2026-09-03]`

**Curved-sky, joint (map, C_ℓ) sampling but lensing-blind**

- **The Commander line** — Eriksen et al. 2004 / Jewell, Levin & Anderson 2004 / Wandelt, Larson & Lakshminarayanan 2004: full-sky Gibbs over (a_ℓm, C_ℓ), conjugate-only, lensing-blind; the structural ancestor of Blocks 1–2. *IDs not pinned.* **`0905.3823`** — the pedagogical guide.
- **Racine, Jewell, Eriksen & Wehus 2016**, `1512.06619` — the joint-move step fixing low-S/N signal–spectrum degeneracy; relevant to Block 1/2 mixing, not just related work. **`2111.07664`** — the successor line; *author list unverified*.
- **BeyondPlanck** `2303.04819`; **Cosmoglobe** `2306.15511` — polarization Gibbs blocks and end-to-end Bayesian analysis without likelihood approximations; the Phase 3 TQU reference.
- **Almanac**, `2305.16134` (Sellentin, Loureiro, Whiteway, Lafaurie, Balan, Olamaie, Jaffe & Heavens 2023, OJA) — *"MCMC-based signal extraction of power spectra and maps on the sphere"*: all-sky HMC over maps and auto/cross-spectra, millions of parameters, spin-2 E/B/EB without EB-leakage. **Actual title recorded for the first time this pass.** `[v 2026-09-03]`
- **Almanac companion**, `2210.13260` (Loureiro, Whiteway, Sellentin, Lafaurie, Jaffe & Heavens, OJA 6, 2023) — **new, and it matters**: *"Weak Lensing power spectra and map inference on the masked sphere"*, HMC over 1.68×10⁷ parameters on the **curved and masked** sky. The file previously implied Almanac was full-sky/noiseless only. Masked curved-sky HMC over (map, C_ℓ) already exists; acknowledge it. `[v 2026-09-03]`
- **Flinch**, `2510.26691` (Crespi, Bonici, Loureiro, Ruiz-Zapatero, Sladoljev, Li, Bayer, Millea & Seljak, 2025-10-30) — differentiable curved-sky field-level inference from masked CMB temperature maps to cosmological parameters; MCLMC "orders-of-magnitude" over HMC; up to 40% tighter than pseudo-C_ℓ. **No φ, no C_L^φφ.** Must be cited, must be distinguished, and its author list must inform the schedule. `[re-verified 2026-09-03]`
- **Taylor, Ashdown & Hobson**, `0708.2989` — HMC-instead-of-Gibbs ancestor; correlation lengths comparable to Gibbs except at the highest S/N. *Author list unverified.*

---

## The competing paradigm: diffusion / learned posteriors

**This axis moved this pass, in the direction that matters.**

- **Sotoudeh, Lemos & Perreault-Levasseur 2026**, `2603.04535` (2026-03-04) — *A Fast Generative Framework for High-dimensional Posterior Sampling: Application to CMB Delensing.* Hierarchical Probabilistic U-Net / VAE posterior sampler, an order of magnitude faster than a diffusion baseline, recovering the unlensed CMB power spectrum with uncertainty estimates and reported robust to cosmological-parameter variation. **The first learned posterior sampler applied to the CMB delensing problem itself.** Sky geometry is not stated in the abstract and it does not infer C_L^φφ — read the body before asserting either in print. `[v 2026-09-03]`
- **`2405.05598`** (MNRAS 533, 423) — denoising-diffusion reconstruction of the lensing convergence, pitched explicitly as an HMC alternative. **JADE `2606.31988`** — joint diffusion posterior over convergence map *and* cosmology, amortised ~0.2 s/sample, needing neither a differentiable forward model nor inference-time MCMC; galaxy weak lensing, not CMB. **`2606.00803`**, **`2512.22683`** (low-ℓ B-modes by reverse diffusion; treats lensing as contaminant), **`2511.04792`** (strong lensing — off-topic, recorded so it is not re-litigated).
- **Spherical Fourier Neural Operators for CMB delensing** — SFNO on HEALPix with differentiable SHTs in JAX; OpenReview `I8k3wwwm9l`, circa Nov 2025. **[UNVERIFIED — no arXiv ID located, page not fetchable. Do not cite until a versioned copy is found.]**

**Four answers to have on file** (unchanged, now aimed at a CMB-domain opponent): exactness with an asymptotic guarantee; the *scope* of the posterior (C_ℓ and C_L^φφ and their correlations with φ, which no learned route produces); no training set required; and someone has to be the reference standard.

---

## "Exact sampler as reference standard"

- **Doeser & Jasche 2026**, `2606.10023` — matching posterior means, marginals or cross-correlations does **not** imply correct uncertainty structure, established by checking Stochastic Interpolants and GLOW flows against **HMC reference posteriors**. The DiffCMB argument, made by someone else, in a neighbouring regime. **Cite prominently in the introduction.**
- **Omori, Zeghal, Chang, Lanusse & Perreault-Levasseur 2026**, `2606.12255` (2026-06-10) — *Towards Practical Field-Level Inference for Weak Lensing.* Implicit (SBI) and explicit field-level inference compared head-to-head on 8×10⁶-parameter forward models; posteriors closely consistent, implicit analyses coverage-tested. **The counterpoint to cite honestly**: there the learned route *did* agree. DiffCMB's claim is that agreement must be demonstrated per problem, not assumed — which is what a reference standard provides. `[v 2026-09-03]`
- **Pietroni & Schmidt 2026**, `2604.25385` (2026-04-28) — *On the Relation Between Field-Level Posteriors, Correlators, and their Likelihoods*: makes explicit which terms power-spectrum and bispectrum likelihoods capture and which are lost under compression. The principled version of "why a field-level posterior, not a summary." `[v 2026-09-03]`
- **Mishra 2026**, `2606.16248` — exact MCMC as gold standard, GP/SBI measured against it (0.3σ→1.5σ drift). Low-dimensional and late-time: a precedent for the protocol, not a competitor.
- **ANVIL / KARMA** (in-house) — the calibrated-but-not-accurate verdict and the C2ST instrument. Own the general claim by citing them, not re-deriving.

---

## Differentiable CMB infrastructure

- **jax-cosmo**, `2302.05163`, OJAp 6, 15 — the canonical "why differentiability" citation and Flinch's substrate.
- **Reinecke, Belkner & Carron 2023**, `2304.10431`, A&A 678, A165 — the accuracy standard for the curved-sky lensing operator and its adjoint (`lenspyx`/`ducc`). **The correct citation for DiffCMB's lensing operator.**
- **Price & McEwen — s2fft**, `2311.14670`, J. Comput. Phys. 510, 113109 (2024) — differentiable, accelerated spherical harmonic and Wigner transforms in JAX/PyTorch; the custom-vjp pattern DiffCMB's `tf.custom_gradient` wrapper follows. **ID confirmed — open item closed.** `[v 2026-09-03]`
- **cuHPX**, `2510.01785` (Cheng, Subramaniam, Wu & Brenowitz, 2025-10-02) — GPU-accelerated differentiable SHTs on **HEALPix**, >20× over existing libraries; better matched to DiffCMB's pixelisation than s2fft. **Abstract confirmed — open item closed.** `[v 2026-09-03]`
- **cunuSHT**, `2406.14542`, RASTI 3, 711 — GPU non-uniform SHTs; the route if Phase 4 (lmax ≥ 1000) is unparked.
- **Furax**, `2603.19600` (Chanial et al., 2026-03-20) — **new.** Modular JAX framework of composable linear operators for CMB map-making, instrument modelling and component separation. Adjacent infrastructure, not a competitor: no lensing operator, no field-level lensing posterior. `[v 2026-09-03]`
- **Negative result, recorded so it is not re-checked:** `2606.28175` (HIcosmo) is a differentiable JAX cosmology framework but is **background-only** — no Boltzmann solver, no CMB, no spherical harmonics. Irrelevant. `[v 2026-09-03]`
- **Elsner & Wandelt 2013**, `1210.4931` — messenger field, the abandoned Phase 0c route. **Papež, Grigori & Stompor 2018**, **Huffenberger & Næss 2018** — messenger-preconditioned CG, the named fallback family; *IDs still not pinned*.

---

## Samplers and the Block 3 mixing problem

Block 3 (φ|a_ℓm, C_ℓ) is where the remaining defect lives: Option 2's strict C_L^φφ SBC rank is 0.3802, moving to 0.4196 under doubled trajectories, with the residual concentrated in ℓ∈[10,30).

- **Neal 2011 / Duane et al. 1987** — HMC; **TFP `DualAveragingStepSizeAdaptation`** — the step-size scheme `run_chain_hmc` uses.
- **Robnik, De Luca, Silverstein & Seljak**, `2212.08549` — *Microcanonical **Hamiltonian** Monte Carlo* (MCHMC), introducing MCLMC as a continuous variant. **Title corrected this pass.** `[v 2026-09-03]`
- **Bayer, Seljak & Modi 2023**, `2307.09504` — MCLMC >1 order of magnitude over HMC at ~2.6×10⁵ dimensions, gap widening with dimension; corroborated by Flinch on curved-sky CMB maps. **In-house status: MCLMC was ported and tested; it fails the stationarity gate and production is plain HMC.** Cite as prior art — and keep the draft from describing Block 3 as MCLMC (that error was found and fixed 2026-09-01).
- **Millea, Anderes & Wandelt**, `2002.00965` — the reparameterisation lesson; still the leading external suspect for a slowly-mixing φ block.
- **Nothing found this pass bears on the ℓ∈[10,30) localisation.** No paper reports a multipole-localised mixing failure in a joint lensing sampler. The internal evidence — strong, asymmetric cross-L Hessian coupling; the Nyström `block` mass matrix falsified — has no literature counterpart. State that plainly rather than manufacturing one.

---

## Validation, convergence and coverage

The lmax=64 rank/coverage test is the paper's headline evidence, so its methodology must be cited exactly.

- **Talts, Betancourt, Simpson, Vehtari & Gelman 2018 — SBC**, `1804.06788` — the protocol `scripts/aggregate_coverage_ranks.py` implements, and the one the paper should name.
- **Cook, Gelman & Rubin 2006**, JCGS 15, 675 — the original posterior-quantile scheme. **Cite the 2017 published correction**, not the 2006 distributional statement.
- **Modrák et al.**, `2211.02383` — SBC sensitivity depends entirely on the test quantity. Directly load-bearing: DiffCMB ranks per-ℓ-bin φ-power summaries, and the residual is now localised to one bin. **Still unread; still on the critical path.**
- **Vehtari, Gelman, Simpson, Carpenter & Bürkner 2021**, `1903.08008`, Bayesian Analysis 16, 667 — rank-normalised, folded, split R̂ plus quantile-local ESS, and rank plots rather than trace plots. Use *this* R̂.
- **Margossian, Hoffman, Sountsov, Riou-Durand, Vehtari & Gelman**, `2110.13017`, Bayesian Analysis (2024) — **nested R̂**, for convergence when running **many short chains** rather than a few long ones. **New, and directly applicable**: the coverage ensemble is 12 short chains per realization, exactly the regime classical split-R̂ is weakest in. `[v 2026-09-03]`
- **Seiffert & Pereira 2024**, `2408.13411` (stat.ME) — ESS/IACT estimators **may not be statistically consistent**; their variance grows linearly in chain length, and two estimators on the same chain disagreed on the *order of magnitude* of the ESS. **New, and the literature home for the hard-won internal lesson that the lag-1 equilibration gate was a measurement artefact** conflating slow-but-stationary with unequilibrated. Cite it when reporting τ_int (Geyer's estimator truncated early in all 48 chain×bin combinations, so those are bounds) — it turns reporting a bound into a virtue rather than a weakness. `[v 2026-09-03]`
- **The SBC-scope finding still has no literature counterexample.** Strict SBC applies to the fields (a_ℓm, φ) but not to spectra whose blocks carry flat/improper implied priors — there is no θ_true ~ p(θ) to rank. Consistent with Talts et al.'s prior-sampling requirement. **Label the spectrum result "interval coverage against realized power," never "calibration."**

---

## Science targets and real-data framing (Phase 3 / LiteBIRD)

- **LiteBIRD lensing forecast**, `2507.22618` — Planck+LiteBIRD full-sky QE; internal delensing improves σ(r) by ~6%. The target experiment's pipeline is still QE. **LiteBIRD multitracer delensing**, `2312.05194`, JCAP 06 (2024) 010 — external tracers improve σ(r) ~20%; Phase 3 must say why *internal* sampling-based delensing is complementary.
- **Hertig et al. (ACT) 2025**, `2511.21949` (2025-11-26, submitted PRD) — *B-mode delensing with DR6 data and external tracers*: ACT DR6 internal lensing + unWISE + Planck CIB, removing ~39% of lensing power at 100≤ℓ≤1500 and ~47% at 30≤ℓ≤300, "the highest delensing efficiency to date" on data. **New — the current real-data benchmark.** `[v 2026-09-03]`
- **Nakato et al. (SPT-3G) 2026**, `2608.06343` (2026-08-06) — *Foreground-Robust Lensing Templates for Primordial Gravitational Wave Searches*: profile-hardened GMV quadratic estimator + CIB, **A_lens^res ≈ 0.48 over 20≤ℓ≤200**, claimed highest-efficiency template to date, validated against non-Gaussian foreground sims. **New — the number any Phase 3 delensing claim will be measured against.** `[v 2026-09-03]`
- **SPT-3G+**, `2608.20236` (2026-08-20) — next-generation SPT receiver for deep lensing maps and foreground-B-mode delensing. Motivation citation only. `[verified via arXiv search listing 2026-09-03; abstract page not fetched]`
- **CMB-S4 iterative internal delensing**, `2310.06729` — the 92–93% simulation benchmark.
- **A_L anomaly is not a live hook.** Planck PR4/NPIPE (CamSpec, HiLLiPoP) weaken it; ACT DR6 lensing (`2503.14454`) shows no excess; `2310.03127` anatomises it. The honest real-data pitch is post-mortem / internal consistency.
- **Foreground robustness**: `2406.15351` (polarized extragalactic foregrounds on Bayesian CMB lensing, PRD 111, 023503) and `2502.20801` (non-Gaussian foreground bias in optimal *joint* lensing + temperature-spectrum reconstruction — the closer analogue). The honest limitations citations.

---

## Counterclaims / must-engage

1. **"Flinch already does curved-sky differentiable field-level CMB inference."** True; cite it. No φ block, no C_L^φφ block — it is the Commander cell done differentiably.
2. **"Almanac already does curved-sky HMC over maps and spectra, on a masked sphere."** True (`2305.16134`, `2210.13260`), at 1.68×10⁷ parameters, and lensing-blind. Do not overstate the novelty of curved-sky sampling itself.
3. **"Darwish 2025 is already a *joint* optimal reconstruction."** Joint over distortion fields, MAP point estimate, no spectrum blocks. One sentence settles it.
4. **"Learned posteriors are faster and now exist for CMB delensing" (`2603.04535`).** Answer with exactness, posterior scope, no-training-set, and the reference-standard argument — and concede honestly that `2606.12255` found implicit and explicit field-level inference agreeing in a neighbouring problem.
5. **"lmax=64 is a toy."** The weakest point in the paper; no literature sets a threshold and the comparison set is 10²–10⁴× larger. Answer on certified correctness, not scale, and give the scaling route.
6. **"Your τ_int and ESS numbers are unreliable."** They are lower bounds. `2408.13411` makes reporting them as bounds defensible; `2110.13017` (nested R̂) is the right diagnostic for a many-short-chains ensemble.

---

## Standing claims-hygiene rule

Re-scan before every submission milestone for **four** things — search the *problem*, not only the *method family*:

1. curved-sky MUSE / curved-sky field-level lensing samplers;
2. the generative/diffusion/VAE route for **CMB** (not just galaxy) lensing posteriors — now an occupied cell (`2603.04535`), not a hypothetical;
3. any φ or lensing extension of **Flinch, Almanac or CMBLensing.jl** — watch Millea, Seljak, Bayer, Loureiro **by name**;
4. **delensing-efficiency records** (`2511.21949`, `2608.06343`) — Phase 3's headline number is benchmarked against a target that moved twice in ten months.

---

## Open items

- [ ] **Fix `docs/paper/main.tex`: `2112.09091`→`2112.09354` (MUSE); `2012.00011`→`2012.01709` (CMBLensing/SPTpol); retitle `2212.08549` to "Microcanonical *Hamiltonian* Monte Carlo".** Highest priority — all three are instantly checkable by a referee.
- [ ] **Cite `2603.04535`** in the competing-paradigm section, replacing JADE as the lead example; read its body first to establish sky geometry.
- [ ] **Cite Doeser & Jasche (`2606.10023`) in the introduction**; and **cite `2606.12255` honestly as the counterpoint** rather than leaving it for a referee.
- [ ] **Add `2210.13260` (masked-sphere Almanac)** alongside `2305.16134`, and correct the draft's "all-sky, noiseless" characterisation of Almanac.
- [ ] **Read Modrák et al. (`2211.02383`) before finalising the coverage-test design** — per-ℓ-bin φ-power is a non-innocent test quantity and the residual is now localised to one bin.
- [ ] **Adopt nested R̂ (`2110.13017`)** alongside rank-normalised split-R̂ (`1903.08008`) and rank plots; **report τ_int as a lower bound citing `2408.13411`**; **cite the corrected form of Cook, Gelman & Rubin 2006**.
- [ ] **State the demonstrated scale as lmax=64 everywhere**, with the comparison set and the cuHPX/cunuSHT scaling route. Never lmax≈128.
- [ ] **Add the "what the deficit is not" paragraph** (not N0/N1, not mean-field, not non-Gaussian deflection, not foregrounds).
- [ ] **Benchmark Phase 3 delensing against `2511.21949` (~47% at 30≤ℓ≤300) and `2608.06343` (A_lens^res ≈ 0.48)**, not only the CMB-S4 forecast.
- [x] **s2fft ID** — `2311.14670`, JCP 510, 113109 (2024). Closed 2026-09-03.
- [x] **cuHPX abstract** — `2510.01785`, GPU differentiable SHTs on HEALPix. Closed 2026-09-03.
- [ ] Locate an arXiv/proceedings version of the SFNO CMB-delensing paper (OpenReview `I8k3wwwm9l`) or drop it — currently **[UNVERIFIED]**.
- [ ] Still unverified before citing: author lists for `1708.06753`, `2111.07664`, `0708.2989`; Papež et al. 2018 and Huffenberger & Næss 2018 IDs; the Eriksen/Jewell/Wandelt 2004 Commander trio IDs.
