# T1.1 scoping: certify a learned CMB-lensing posterior against the exact one

*2026-10-07. Step one of `ROADMAP.md` T1.1: which figures, supporting which claim, and what has to be decided before anything is built. Nothing here needs compute yet.*

## The claim

A learned posterior trained on our simulator (exact operator, lmax = nside = 64, A_φ = 3000, σ = 30) either reproduces the exact joint posterior within stated bounds, or departs from it in a quantified way. Either way it is the first such certification for CMB lensing, and the exact posteriors are released as the benchmark (T3.2). Scope qualifiers that every sentence must carry: lmax = 64, amplified lensing, temperature only, full sky.

## Can we reuse `2603.04535`? No

Read 2026-10-07 (arXiv HTML; Section 3.2, Appendix A.2, B.4):

- **Geometry:** flat-sky patches, "each covering a 160′×160′ region with 32×32 resolution" (1024 pixels). Ours is full sky, nside 64 (49 152 pixels).
- **Forward model:** "We begin by calculating the lensed and unlensed TT power spectra using CAMB … We then apply the same noise realization to these power spectra to generate lensed and unlensed maps in Fourier space." No deflection field is applied, and no φ appears anywhere.
- **Inferred:** the unlensed map only (the network predicts lensed − unlensed). No φ, no C_L^φφ, and no C_ℓ posterior; spectra are computed afterwards from map samples.
- **Noise:** σ_n = 20 on the lensed maps, no beam stated. Temperature only.
- **Training:** 2¹³ training and 2¹¹ test maps; a 27.8 M-parameter mean U-Net plus a 3.9 M-parameter HPU-Net for the dispersion.
- **Validation:** TARP and spectrum overlays ("slightly conservative"). No exact or MCMC reference posterior.
- **Code:** no repository or data statement. A web search found none.

So T1.1 trains its own model on our simulator. For T2 positioning: the first learned CMB-delensing sampler uses a Gaussian stand-in for lensing on 32×32 flat patches, validated against truth, not against a reference posterior.

## What the exact reference can support

- **Ensemble:** Block-4-ON, ν = 30, N = 48 skies (`ens_exact_l64_A3000_n30_nu30_long`), 3600 post-burn-in sweeps per sky.
- **Effective samples per sky:** median ESS per 3600 sweeps is 252 / 74 / 181 / 342 (Blocks 1–4: C_ℓ, a_ℓm, φ, C_L^φφ; `achievements.md`), and lower for the low-L φ modes (τ_int ≈ 280–430 sweeps). That is enough for per-sky means and widths, but too few for a classifier two-sample test (C2ST) in the ~8300-dimensional joint space. A spectra-level C2ST (124 dimensions) is possible but weak, so its power has to be stated.
- **⚠ The C_ℓ^TT prior is flat and improper** in that ensemble, with the truth held at the fiducial. An amortised learned posterior is trained on draws from a proper prior, so it cannot target that posterior as it stands. **Decision 1** below.
- **⚠ The a_ℓm band edge `[60,64)`** waits on T0.1 b. Leave ℓ ≥ 60 out of any learned-vs-exact a_ℓm/C_ℓ comparison until the right reference is known.

## Candidate figures (each with its drawn null)

1. **Learned-posterior SBC:** rank histograms per block, by C_ℓ and C_L^φφ bins, on ~10³ fresh simulations. An amortised model makes this cheap, and it needs no exact chains, so it carries the most statistical power.
2. **Learned vs exact, sky by sky (N = 48):** mean shift in exact-sd units and the width ratio per ℓ/L bin, with a Monte Carlo band from the exact chains' ESS.
3. **Correlation structure:** the (C_ℓ, C_L^φφ) cross-correlation matrix, learned vs exact, the figure-4 analogue. This is the Doeser & Jasche point (`2606.10023`): matching marginals does not imply matching correlations.
4. *(optional)* a spectra-level C2ST, with its power stated.

## Decisions (taken 2026-10-07: 1a strict ensemble, 2a spectra-level NPE, 3 TFP)

1. **The prior the comparison uses.**
   - **(a) Recommended:** a strict exact ensemble (`--cl_prior_nu`, proper prior on every block). This is new compute, a 48-sky campaign at ~19 h per task. The same choice is a candidate for T0.1 c's collapse guard, so one campaign can serve both.
   - **(b)** Keep the flat-prior ensemble: train the learned model with a proper prior and importance-reweight its samples to the flat prior. This needs no new exact compute, but the weights may degenerate, which has to be checked first.
2. **What the learned model infers.**
   - **(a) Recommended first:** spectra-level NPE, posterior over (C_ℓ, C_L^φφ) (124 dimensions) from summaries of the map (the data power spectrum plus the QE validated at this configuration). It is cheap and runs on CPU, and it targets the title's object directly. Any shortfall against the exact posterior is information lost by the summaries, a result in its own right (compare figure 3's QE⊕prior).
   - **(b)** Field-level conditional generative model over (a_ℓm, φ) on HEALPix nside 64. It needs sphere-aware architecture and a GPU, so it is the stretch goal.
3. **Framework.** The `.venv` has TF + TFP, with no torch, jax, sbi or sklearn. Either build flows in TFP (no new dependencies), or create a separate torch environment (zuko/sbi). For GPU training, dine2 has 8 GPU nodes but does not mount `/cosma5`, so outputs and inputs go to `/cosma7` or `/cosma8`.

## Related learned posteriors (for T2 positioning; `literature.md`)

- **PUREPath `2406.19367`** (Sudevan & Chen 2024): learned CMB temperature posterior on HEALPix nside 32 for component separation. Curved sky, but no lensing and no exact reference.
- **ABCMB `2407.10013`** (Yi et al. 2024): generative delensing of polarization feeding LFI on r. No φ posterior, and no exact reference stated in the abstract.
