# Literature — DiffCMB (Paper 7)

*Annotated bibliography with the novelty/positioning argument folded in. Action items live in `ROADMAP.md`'s "Literature actions" section — this file is reference only.*

**Last checked: 2026-09-22.** Method: the arXiv API worked this time (HTTPS endpoint; the HTTP one only redirects). Date-sorted queries covering 2026-08-28→09-22 for `abs:"CMB" AND abs:lensing`, `"field-level"`, `delensing`, `"lensing potential"`, `Bayesian AND "CMB lensing"`, `diffusion AND CMB`, `Gibbs AND CMB`, `MUSE`, `"Hamiltonian Monte Carlo"` (astro-ph.CO), `microcanonical`, `"simulation-based calibration"`, `differentiable AND "spherical harmonic"`. Named authors: Millea, Loureiro, Carron. `id_list` lookups closed every outstanding `[unverified]` ID below. Also ~7 web searches. Previous pass: 2026-09-03 (web search plus arXiv listings; the API returned 429 that time).

**No `.bib` to reconcile against yet.** `papers/7_DiffCMB/` has no manuscript content; the live draft is `diffcmb/docs/paper/main.tex` with inline `\bibitem`s.

---

## Verdict

**The core claim survives: no curved-sky joint (a_ℓm, C_ℓ, φ, C_L^φφ) sampler exists, and no curved-sky MUSE exists. Scoop risk: LOW.** The 2026-09-22 rescan (08-28→09-22) found no CMB field-level, MAP or sampling lensing paper at all. The only new CMB-lensing reconstruction since 09-03 is a matter-power *deconvolution* of published bandpowers (`2609.08457`), which says nothing about the map-level posterior. The earlier pass also missed a curved-sky QE paper, SPT-3G Summer (`2607.05784`), which is added below. The `"field-level"` enumeration for 2026-06→09 returns no CMB entry at all; August's only CMB-lensing analysis (`2608.31136`, SPT-3G D1) is a quadratic estimator; named-author watch (Millea, Seljak, Bayer, Loureiro) found no φ or lensing extension of Flinch, Almanac or CMBLensing.jl.

**Must engage, in order:** (1) **Flinch `2510.26691`** — occupies the Commander cell differentiably, no φ/C_L^φφ; differentiable machinery is table stakes, never lead with it. (2) **`2603.04535`** — a learned posterior sampler now exists for CMB delensing itself, replacing galaxy weak lensing (JADE) as the sharpest competing-paradigm example. (3) **Doeser & Jasche `2606.10023`** — external statement of why an exact reference posterior is needed; still uncited.

**On the demonstrated scale (lmax=64, not 128).** No threshold exists in the literature for a defensible minimum scale — the comparison set is 10²-10⁴× larger on raw dimension (MUSE ~6×10⁶ latents, Almanac 1.68×10⁷, Bayer et al. ~2.6×10⁵, weak-lensing FLI 8×10⁶ vs DiffCMB's ~4×10³ dof/field). **The defence must be certified correctness at a scale where correctness can be certified, not scale itself.** State the scale plainly, give the scaling route (cuHPX/cunuSHT) as future work. Never write lmax≈128.

**A defence resting on certified correctness cannot afford a parameterisation defect — why the missing `Im(a_{ℓ,1})` dof mattered here, not only internally.** Under the pre-2026-09-06 packing the field count above would have read 4030, and a referee finding a missing mode per multipole alongside the MUSE-scale comparison would have had a second, cheaper objection. Closed (`achievements.md`).

---

## Citation corrections (found and fixed 2026-09-03)

1. MUSE was cited as `2112.09091` ("Dualities in one-dimensional quantum lattice models" — unrelated). Correct: **`2112.09354`** (Millea & Seljak, PRD 105, 103531).
2. CMBLensing/SPTpol was cited as `2012.00011` ("Mass-gap Mergers in AGN" — unrelated). Correct: **`2012.01709`** (Millea et al. 2021, ApJ 922, 259).
3. `2212.08549` was titled "Microcanonical *Langevin* Monte Carlo" — the paper is "Microcanonical **Hamiltonian** Monte Carlo" (Robnik, De Luca, Silverstein & Seljak). ID and authors were right, title string was wrong.

Still standing from earlier passes: Carron & Lewis MAP lensing is `1704.08230`, not `1701.01712` (`1701.01712` is Carron, Lewis & Challinor, *Internal delensing of Planck CMB temperature and polarization*, a different paper); `2209.10512` is geometry-agnostic MUSE-via-implicit-differentiation, not a flat-sky MUSE follow-up.

**IDs verified 2026-09-22 against the arXiv API** (these were all marked unverified before):

4. `1708.06753` is **Millea, Anderes & Wandelt 2019, *Bayesian delensing of CMB temperature and polarization*, PRD 100, 023509**. It is not a minor entry in the same lineage: it is the flat-sky joint (f, φ) sampler for **T+P** and the predecessor of `2002.00965`. It is promoted below.
5. `2111.07664` is **Ducrocq, Chopin, Errard & Stompor, *Improved Gibbs samplers for CMB power spectrum estimation***. It is not by the Racine/Eriksen group. It is its own line of work on better (a_ℓm, C_ℓ) Gibbs moves.
6. `0708.2989` — Taylor, Ashdown & Hobson, *Fast optimal CMB power spectrum estimation with Hamiltonian sampling*. Confirmed.
7. Commander trio pinned: **Jewell, Levin & Anderson** `astro-ph/0209560` (ApJ 609, 1, 2004); **Wandelt, Larson & Lakshminarayanan** `astro-ph/0310080` (PRD 70, 083511, 2004); **Eriksen et al.** `astro-ph/0407028` (ApJS 155, 227, 2004).
8. Messenger family pinned: **Papež, Grigori & Stompor** `1803.03462`; **Huffenberger & Næss** `1705.01893` (arXiv 2017, published 2018).
9. SFNO CMB delensing: the OpenReview page (`I8k3wwwm9l`, 2025-11-21) is confirmed to exist. **Still no arXiv ID.** Cite it as an OpenReview workshop paper or leave it out.

---

## The 2×2: {flat-sky, curved-sky} × {marginal/point-estimate, joint φ sampling}

Cell (curved-sky, joint φ sampling) is empty. Everything below occupies one of the other three.

**Flat-sky, joint sampling — nearest prior work**

- **Millea, Anderes & Wandelt 2020**, `2002.00965`, PRD 102, 123542 — flat-sky joint (f, φ, r, A_φ) sampling; CMBLensing.jl. **The nearest prior work, full stop.** Their result that naive block alternation mixes catastrophically and the fix is reparameterisation, not compute, is the standing external reference for DiffCMB's Block 3 problem.
- **Millea et al. 2021**, `2012.01709`, ApJ 922, 259 — CMBLensing.jl on real SPTpol data; A_φ = 0.949 ± 0.122, 17% smaller errors than their own QE.
- **Millea, Anderes & Wandelt 2019**, `1708.06753`, PRD 100, 023509 — *Bayesian delensing of CMB temperature and polarization*. Flat-sky joint sampling of the unlensed T/E/B fields and φ. The polarization precedent to cite whenever Phase 3 TQU is discussed. `main.tex` does not cite it yet.
- **Anderes, Wandelt & Lavaux 2015**, `1412.4079` — the Gibbs-over-(field, φ) ancestor.

**Curved-sky, marginal or point-estimate**

- **MUSE**, `2112.09354`, PRD 105, 103531 — ~6×10⁶ latents; not a sampler, no joint posterior. Their "curved-sky HMC is slightly out of reach" defines this project's empty cell. `2209.10512` — MUSE via implicit differentiation, geometry-agnostic. `2411.06000` — production MUSE on SPT-3G data.
- **Carron & Lewis 2017**, `1704.08230`, PRD 96, 063510 — iterative MAP lensing (LensIt); a point estimate.
- **Belkner, Carron et al. 2023**, `2310.06729` — `delensalot`, CMB-S4 iterative internal delensing, 92-93% B-lensing power removed. The curved-sky state of the art that is not a sampler.
- **Darwish 2025**, `2503.03682` — optimal *joint* MAP over multiple line-of-sight distortion fields (lensing + birefringence + patchy screening). Joint over distortion fields, a point estimate, no C_ℓ/C_L^φφ block — draw the distinction explicitly if "joint" is raised as prior art.
- **Iterative/QE frontier**: `2407.00228` (non-Gaussian deflections), `2506.20667` (noise-bias-minimising iterative estimator), `2605.18659` (control variates).
- **SPT-3G Summer QE lensing**, `2607.05784` (Levy et al., 2026-07-07) — curved-sky QE on ~2640 deg², A = 1.015 ± 0.053 over 50<L<2000. Anderes is a co-author. Another large 2026 curved-sky analysis that still uses a quadratic estimator.
- **SPT-3G D1 QE lensing**, `2608.31136` (Omori et al., 2026-08-31) — lensing amplitude to 2% of ΛCDM, Σm_ν < 0.072 eV (95%). No field-level/MAP/sampling content. The scoop check's headline negative: the flagship 2026 lensing analysis is still a quadratic estimator.

**Curved-sky, joint (map, C_ℓ) sampling but lensing-blind**

- **The Commander line** — Jewell, Levin & Anderson `astro-ph/0209560` / Wandelt, Larson & Lakshminarayanan `astro-ph/0310080` / Eriksen et al. `astro-ph/0407028`: full-sky Gibbs over (a_ℓm, C_ℓ), conjugate-only and lensing-blind. The structural ancestor of Blocks 1-2. `0905.3823` — the pedagogical guide.
- **Racine, Jewell, Eriksen & Wehus 2016**, `1512.06619` — the joint-move step fixing low-S/N signal-spectrum degeneracy; relevant to Block 1/2 mixing. **Ducrocq, Chopin, Errard & Stompor**, `2111.07664` — improved Gibbs samplers for (a_ℓm, C_ℓ). A separate line of work from Racine et al., also aimed at the low-S/N signal-spectrum mixing problem.
- **BeyondPlanck** `2303.04819`; **Cosmoglobe** `2306.15511` — polarization Gibbs blocks and end-to-end Bayesian analysis without likelihood approximations; the Phase 3 TQU reference. **Cosmoglobe/LiteBIRD feasibility**, `2507.05324` (Aurvik et al.) — about 3000 CPU-hours per Gibbs sample from TOD to cosmological parameters, still lensing-blind. Useful as a cost reference if Phase 3 talks about LiteBIRD.
- **Almanac**, `2305.16134` (Sellentin et al. 2023, OJA) — all-sky HMC over maps and auto/cross-spectra, millions of parameters, spin-2 E/B/EB without EB-leakage.
- **Almanac companion**, `2210.13260` (Loureiro et al., OJA 6, 2023) — masked-sphere Almanac: HMC over 1.68×10⁷ parameters on the curved **and masked** sky. Corrects an earlier "all-sky/noiseless only" characterisation — acknowledge it.
- **Flinch**, `2510.26691` (Crespi, Bonici, Loureiro, Ruiz-Zapatero, Sladoljev, Li, Bayer, Millea & Seljak, 2025-10-30) — differentiable curved-sky field-level inference from masked CMB temperature maps to cosmological parameters; MCLMC "orders of magnitude" over HMC; up to 40% tighter than pseudo-C_ℓ. No φ, no C_L^φφ. Must be cited and distinguished; its author list informs the schedule.
- **Taylor, Ashdown & Hobson**, `0708.2989` — the ancestor of using HMC instead of Gibbs for C_ℓ estimation.

---

## The competing paradigm: diffusion / learned posteriors

- **Sotoudeh, Lemos & Perreault-Levasseur 2026**, `2603.04535` — *A Fast Generative Framework for High-dimensional Posterior Sampling: Application to CMB Delensing.* Hierarchical Probabilistic U-Net/VAE sampler, an order of magnitude faster than diffusion, recovering the unlensed CMB spectrum with uncertainty estimates. The first learned posterior sampler applied to CMB delensing itself — read the body before asserting sky geometry or C_L^φφ inference either way.
- **`2405.05598`** (MNRAS 533, 423) — denoising-diffusion lensing convergence reconstruction, pitched as an HMC alternative. **JADE `2606.31988`** — joint diffusion posterior over convergence map and cosmology, amortised ~0.2s/sample, no differentiable forward model or MCMC needed; galaxy weak lensing, not CMB. `2606.00803`, `2512.22683` (low-ℓ B-modes by reverse diffusion), `2511.04792` (strong lensing, off-topic).
- **Guzman & Meyers 2025**, `2512.19577` — *Deep Learning for Primordial B-mode Extraction*. A learned route for removing lensing (and other secondary) B-modes. It targets r directly and has no φ posterior. Missed by the 09-03 pass.
- **SFNO CMB delensing** — HEALPix SFNO with differentiable JAX SHTs; OpenReview `I8k3wwwm9l`, 2025-11-21 (the page is confirmed to exist). Still no arXiv ID after the 2026-09-22 search.
- **White, Chandrashekaran, Avestruz, Regier & LSST DESC 2026**, `2609.07833` — amortized NPE from images to tomographic shear/convergence fields, reported as "well-calibrated" on DC2. This is galaxy weak lensing, not CMB. It is another learned field-level posterior whose calibration was checked against truth, not against an exact reference posterior, so it is the kind of result the Doeser & Jasche point applies to.

**Four answers on file:** exactness with an asymptotic guarantee; the *scope* of the posterior (C_ℓ, C_L^φφ, and their correlations with φ — no learned route produces this); no training set required; someone has to be the reference standard.

---

## "Exact sampler as reference standard"

- **Doeser & Jasche 2026**, `2606.10023` — matching posterior means/marginals/cross-correlations does **not** imply correct uncertainty structure (checked against HMC reference posteriors, Stochastic Interpolants and GLOW flows). The DiffCMB argument, made by someone else, in a neighbouring regime. Cite prominently in the introduction.
- **Omori, Zeghal, Chang, Lanusse & Perreault-Levasseur 2026**, `2606.12255` — *Towards Practical Field-Level Inference for Weak Lensing.* Implicit (SBI) and explicit field-level inference compared head-to-head at 8×10⁶ parameters; posteriors closely consistent. The counterpoint to cite honestly — there the learned route did agree; DiffCMB's claim is that agreement must be demonstrated per-problem, which is what a reference standard provides.
- **Pietroni & Schmidt 2026**, `2604.25385` — makes explicit which terms power-spectrum/bispectrum likelihoods capture and which are lost under compression. The principled version of "why a field-level posterior, not a summary."
- **Stadler 2026**, `2609.16719` — measures the field-level vs power+bispectrum gain for DESI-LRG-like redshift-space mocks: about 2× on the primordial amplitude, up to 3× in extended models. A galaxy-clustering number that goes with Pietroni & Schmidt when motivating field-level analysis.
- **Mishra 2026**, `2606.16248` — exact MCMC as gold standard, GP/SBI measured against it. Low-dimensional/late-time precedent for the protocol, not a competitor.
- **ANVIL / KARMA** (in-house) — the calibrated-but-not-accurate verdict and the C2ST instrument.

---

## Differentiable CMB infrastructure

- **jax-cosmo**, `2302.05163`, OJAp 6, 15 — the canonical "why differentiability" citation and Flinch's substrate.
- **Reinecke, Belkner & Carron 2023**, `2304.10431`, A&A 678, A165 — accuracy standard for the curved-sky lensing operator and its adjoint (`lenspyx`/`ducc`). The correct citation for DiffCMB's lensing operator.
- **Price & McEwen — s2fft**, `2311.14670`, J. Comput. Phys. 510, 113109 — differentiable JAX/PyTorch SHTs and Wigner transforms; the custom-vjp pattern DiffCMB's `tf.custom_gradient` wrapper follows.
- **cuHPX**, `2510.01785` — GPU differentiable HEALPix SHTs, >20× over existing libraries; better matched to DiffCMB's pixelisation than s2fft.
- **cunuSHT**, `2406.14542`, RASTI 3, 711 — GPU non-uniform SHTs; the route if Phase 4 (lmax≥1000) is unparked.
- **Furax**, `2603.19600` — modular JAX linear-operator framework for CMB map-making/component separation. Adjacent, not a competitor: no lensing operator, no field-level lensing posterior.
- **Elsner & Wandelt 2013**, `1210.4931` — messenger field, the abandoned Phase 0c route. Papež, Grigori & Stompor 2018; Huffenberger & Næss 2018 — messenger-preconditioned CG fallback family; IDs not pinned.
- **Blast.jl extension**, `2609.01855` (Chiarenza, Bonici et al., 2026-09-01) — differentiable non-Limber angular C_ℓ, including CMB-lensing cross-spectra, with custom AD rules, demonstrated with gradient-based samplers on a 35-parameter model. This is theory-spectrum infrastructure, not field-level. Bonici is also a Flinch author, so it is the Julia/Flinch group's stack and worth watching.
- Checked and irrelevant: `2606.28175` (HIcosmo) is differentiable JAX cosmology but background-only, no CMB. `2609.08457` (Dawn et al.) is Richardson-Lucy deconvolution of the Planck+ACT+SPT lensing *bandpowers* into P_lin(k), with no map-level inference. `2609.16128` (CMB-HD foregrounds) is off-topic apart from its remark that polarization-only lensing estimators are barely affected by extragalactic foregrounds.

---

## Samplers and the Block 3 mixing problem

Block 3 (φ|a_ℓm, C_ℓ) is where the remaining defect lives (`achievements.md`'s "Open sampling question" section): strict C_L^φφ SBC rank concentrated in ℓ∈[10,30).

- **Neal 2011 / Duane et al. 1987** — HMC; **TFP `DualAveragingStepSizeAdaptation`** — the step-size scheme `run_chain_hmc` uses.
- **Robnik, De Luca, Silverstein & Seljak**, `2212.08549` — *Microcanonical Hamiltonian Monte Carlo* (MCHMC), introducing MCLMC as a continuous variant.
- **Bayer, Seljak & Modi 2023**, `2307.09504` — MCLMC >1 order of magnitude over HMC at ~2.6×10⁵ dimensions, gap widening with dimension; corroborated by Flinch on curved-sky CMB maps. In-house: MCLMC ported and tested, fails the stationarity gate, production is plain HMC — cite as prior art without describing Block 3 as MCLMC.
- **Millea, Anderes & Wandelt**, `2002.00965` — the reparameterisation lesson; still the leading external suspect for a slowly-mixing φ block.
- **von Campe & Schäfer 2026**, `2609.07620` — a thermodynamic analysis of MCHMC. It argues canonical MCMC is more natural than the microcanonical form on thermodynamic and information-theoretic grounds, and gives a variant for low-dimensional problems. Useful only as a secondary citation if a referee asks why production uses plain HMC rather than MCLMC. The real reason is the failed in-house stationarity gate.
- **Nothing found bears on the ℓ∈[10,30) localisation specifically.** No paper reports a multipole-localised mixing failure in a joint lensing sampler; the internal evidence (asymmetric cross-L Hessian coupling, the falsified Nystrom `block` mass matrix) has no literature counterpart.

---

## Validation, convergence and coverage

The lmax=64 rank/coverage test is the paper's headline evidence, so its methodology must be cited exactly.

- **Talts, Betancourt, Simpson, Vehtari & Gelman 2018 — SBC**, `1804.06788` — the protocol `scripts/aggregate_coverage_ranks.py` implements.
- **Cook, Gelman & Rubin 2006**, JCGS 15, 675 — the original posterior-quantile scheme. Cite the 2017 published correction, not the 2006 distributional statement.
- **Modrák et al.**, `2211.02383` — SBC sensitivity depends entirely on the test quantity; directly load-bearing since DiffCMB ranks per-ℓ-bin φ-power and the residual is localised to one bin.
- **Vehtari, Gelman, Simpson, Carpenter & Bürkner 2021**, `1903.08008`, Bayesian Analysis 16, 667 — rank-normalised, folded, split R̂ plus quantile-local ESS. Use this R̂.
- **Margossian et al.**, `2110.13017`, Bayesian Analysis (2024) — nested R̂, for many-short-chains convergence — directly applicable, the coverage ensemble is 12 short chains per realization.
- **Seiffert & Pereira 2024**, `2408.13411` (stat.ME) — ESS/IACT estimators may not be statistically consistent; two estimators on the same chain disagreed on the ESS's order of magnitude. The literature home for the hard-won internal lesson that the lag-1 equilibration gate was a measurement artefact.
- **The SBC-scope finding still has no literature counterexample.** Strict SBC applies to the fields (a_ℓm, φ) but not to spectra whose blocks carry flat/improper implied priors — no θ_true~p(θ) to rank. Label the spectrum result "interval coverage against realized power," never "calibration."

---

## Science targets and real-data framing (Phase 3 / LiteBIRD)

- **LiteBIRD lensing forecast**, `2507.22618` — Planck+LiteBIRD full-sky QE; internal delensing improves σ(r) ~6%. Target pipeline is still QE. **LiteBIRD multitracer delensing**, `2312.05194`, JCAP 06(2024)010 — external tracers improve σ(r) ~20%; Phase 3 must say why internal sampling-based delensing is complementary.
- **Hertig et al. (ACT) 2025**, `2511.21949` — ACT DR6 internal lensing + unWISE + Planck CIB, ~39% removed at 100≤ℓ≤1500, ~47% at 30≤ℓ≤300 — highest efficiency to date on data. The current real-data benchmark.
- **Nakato et al. (SPT-3G) 2026**, `2608.06343` — profile-hardened GMV QE + CIB, A_lens^res≈0.48 over 20≤ℓ≤200. The number any Phase 3 delensing claim is measured against.
- **SPT-3G+**, `2608.20236` — next-generation SPT receiver for deep lensing maps and foreground-B-mode delensing. Motivation citation only.
- **CMB-S4 iterative internal delensing**, `2310.06729` — the 92-93% simulation benchmark.
- **A_L anomaly is not a live hook.** Planck PR4/NPIPE weaken it; ACT DR6 lensing (`2503.14454`) shows no excess; `2310.03127` anatomises it. The honest real-data pitch is post-mortem/internal consistency.
- **Foreground robustness**: `2406.15351` (polarized extragalactic foregrounds on Bayesian CMB lensing) and `2502.20801` (non-Gaussian foreground bias in joint lensing+temperature-spectrum reconstruction — the closer analogue).

---

## Counterclaims / must-engage

1. **"Flinch already does curved-sky differentiable field-level CMB inference."** True; cite it. No φ/C_L^φφ block — it's the Commander cell done differentiably.
2. **"Almanac already does curved-sky HMC over maps and spectra, on a masked sphere."** True (`2305.16134`, `2210.13260`), 1.68×10⁷ parameters, lensing-blind. Don't overstate the novelty of curved-sky sampling itself.
3. **"Darwish 2025 is already a joint optimal reconstruction."** Joint over distortion fields, MAP point estimate, no spectrum blocks. One sentence settles it.
4. **"Learned posteriors are faster and now exist for CMB delensing" (`2603.04535`).** Answer with exactness, posterior scope, no-training-set, reference-standard — and concede `2606.12255` found implicit/explicit field-level inference agreeing in a neighbouring problem.
5. **"lmax=64 is a toy."** The weakest point; no literature threshold, comparison set 10²-10⁴× larger. Answer on certified correctness, not scale.
6. **"Your τ_int and ESS numbers are unreliable."** They are lower bounds — `2408.13411` makes reporting them as bounds defensible; `2110.13017` is the right diagnostic for a many-short-chains ensemble.

---

## Standing claims-hygiene rule

Re-scan before every submission milestone for four things — search the *problem*, not only the *method family*: (1) curved-sky MUSE/curved-sky field-level lensing samplers; (2) the generative/diffusion/VAE route for CMB (not just galaxy) lensing posteriors — now an occupied cell (`2603.04535`); (3) any φ/lensing extension of Flinch, Almanac or CMBLensing.jl — watch Millea, Seljak, Bayer, Loureiro, Bonici by name (2026-09-22: the only new item from this group is LiFT `2609.14862`, a learned BAO reconstruction on DESI galaxies, not CMB; also Blast.jl above); (4) delensing-efficiency records (`2511.21949`, `2608.06343`) — Phase 3's headline number is benchmarked against a target that moved twice in ten months.
