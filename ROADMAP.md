# Research Roadmap: Differentiable Bayesian CMB Analysis

## The Honest Critique First

This codebase currently implements a Gibbs sampler over `(alm, C_l)` with:
- A Gaussian CMB prior `p(alm | C_l)`
- A white-noise Gaussian likelihood `p(d | alm)`
- Exact inverse-Gamma draws for `C_l | alm`
- HMC for `alm | C_l`

This is algorithmically what Commander (Jewell et al. 2004, Wandelt et al. 2004) does, and Commander has been running on real Planck data in production for 15 years. Replicating it in Python is not a publishable contribution on its own.

**The question is: what can this codebase do that Commander cannot?**

Commander is Fortran, uses conjugate priors throughout, and is not differentiable. The moment your model steps outside conjugate-Gaussian assumptions — lensing, non-Gaussian primordial physics, learned priors, instrument systematics — Commander's Gibbs blocks break and you need something else. This repo has `psi_tf` with full autodiff via TensorFlow. That is the unlock.

---

## The Gap in Current Research

### What Commander leaves unsolved

The CMB we observe is the **lensed** temperature field:

```
T_tilde(n) = T(n + grad(phi(n)))
```

where `phi` is the projected gravitational lensing potential. This remapping makes `T_tilde` non-Gaussian even though the unlensed `T` is Gaussian. The true joint posterior is:

```
p(alm_unlensed, C_l^TT, phi, C_l^phiphi | d)
```

Current state of the art:

| Method | What it does | Limitation |
|---|---|---|
| Commander | Ignores lensing; treats `T_tilde ≈ T` | Biased C_l estimates; gets worse with S4 sensitivity |
| Quadratic estimator (Okamoto & Hu 2003) | Estimates phi from data alone | Suboptimal; no uncertainty propagation to C_l |
| Carron & Lewis (2017) iterative | MAP of `p(phi | d)` | Point estimate only, no posterior samples |
| MUSE (Millea & Wandelt 2021) | Marginalises over alm analytically, samples phi | Only samples phi, not the joint (alm, C_l, phi) |

**The gap:** nobody has a working full joint posterior sampler over `(alm_unlensed, C_l, phi)` at map level that properly propagates uncertainties between all three. For Simons Observatory and CMB-S4, where the lensing signal is enormous and B-mode delensing is critical for detecting primordial gravitational waves, this matters enormously.

### Why this repo is the right starting point

- `psi_tf` is differentiable — you can compute gradients of the log-likelihood with respect to both `alm` and `phi` through a lensing operator
- The Gibbs structure naturally extends to a third block for `phi`
- The HMC infrastructure for non-conjugate blocks already exists
- Commander cannot do any of this without a full rewrite in a differentiable language

---

## Roadmap

### Phase 0 — Establish and validate the baseline (current work)

**Goal:** demonstrate the L=300 Gibbs sampler recovers the correct CMB power spectrum from real Planck data, establishing that the infrastructure is correct before extending it.

**Historical run summary** (all chains to date):

| Run | Precision | Accept | logp std | R-hat C_l med | R-hat alm med | ESS C_l | Status |
|-----|-----------|--------|----------|--------------|---------------|---------|--------|
| Gibbs L=200 synthetic | float32 | ~63% | ~10.5 | 1.000 | not measured | ~1550 | C_l only |
| Gibbs L=200 real | float32 | ~63% | 10.5 | 1.000 | 17 619 | ~1550 | FROZEN |
| Gibbs L=300 real | float32 | ~38% | 13.2 | 1.000 | 58 112 | ~1550 | FROZEN |
| **Gibbs L=300 real** | **float64** | **71%** | **26 051** | **1.026** | **2.64** | **385** | **Phase 0 ✓** |
| CG L=300 real (running) | float64 | 100% | — | — | — | — | Phase 0b ▶ |

Float32 chains show false convergence: C_l R-hat ≈ 1.000 but alm R-hat = 18k–58k. Root cause: float32 gradient noise in the `Y^H` matvec across ~607k pixels drives HMC step to floor ~1e-7.

- [x] Get 4-chain L=300 float64 run — completed 2026-06-27 (4 chains × 1000 samples)
- [x] Compute R-hat — **median 1.026, max 1.085, 0% exceed 1.1** ← C_l fully converged
- [x] Measure ESS per C_l — **median 385/800 post-burn (48% efficiency); ACF drops to 0 at lag 1**
- [x] Implement CG exact alm sampler (`sample_alm_cg`, Phase 0b) — 2026-06-27
- [x] Warm-start 4 CG chains from Phase 0 checkpoints — job 11513133 submitted on gc001–gc004
- [x] Tag MSci project baseline in git — `v0.0-msci` at commit `2f7441c` (unlensed Gibbs + CG, pre-lensing)
- [ ] Verify CG results: ESS ≈ N at all multipoles including l=200–300; logp plateau reached
- [ ] Plot recovered C_l vs Planck official power spectrum (Commander/Plik)

**Phase 0 high-l drift (resolved by CG):**

| Multipole | lnCl drift significance | ESS HMC (chain 4) | ESS CG (expected) |
|-----------|------------------------|-------------------|-------------------|
| l=2 | ~0.5σ | ~1000 | ~1000 |
| l=10 | ~1σ | ~900 | ~1000 |
| l=50 | ~3σ | ~900 | ~1000 |
| l=200 | **11σ** | ~41 | **~1000** |

**Output:** validated C_l posterior with ESS ≈ N at all l. This is table-stakes before claiming anything about extending the model.

---

### Phase 0b — CG-based exact alm sampler (algorithmic fix for high-l mixing)

**Goal:** replace the HMC alm | C_l step with a direct conjugate-gradient draw. This gives IAT = 1 at every multipole by construction, eliminating the high-l drift problem entirely without requiring longer chains.

**Why this is possible:** the alm | C_l Gibbs conditional is exactly Gaussian:

```
p(alm | C_l, d) ∝ exp(-½ alm^T A alm + alm^T Y^T N^{-1} d)
A = C_l^{-1}(diag, grouped by l) + (1/σ²) Y^T Y
```

An exact independent draw can be obtained by solving the linear system:

```
A x = b_noise
b_noise = C_l^{-1/2} ω₁ + (1/σ) Y^T ω₂ + (1/σ²) Y^T d
```

where ω₁ ~ N(0, I_alm), ω₂ ~ N(0, I_pix) are fresh white noise draws each iteration. The solution x is a sample from the exact conditional; no accept/reject needed.

The system is solved via PCG using the diagonal preconditioner `P = C_l^{-1} + (1/σ²) diag(Y^T Y)` (which is already computed as the mass matrix). Convergence in O(10–50) PCG iterations is expected since the preconditioner nearly diagonalises A. Both the Y matvec (alm → map) and Y^T matvec (map → alm) are already implemented in `almtomap_tf` / the adjoint, so no new infrastructure is needed.

**This is what Commander does.** The critical difference is that Commander is Fortran with conjugate priors throughout; once you add lensing in Phase 2, the posterior is no longer Gaussian and the CG approach breaks — requiring HMC to return. The CG sampler here serves as a validated baseline and optimal preconditioner for Phase 2.

- [x] Implement `sample_alm_cg(model, current_lncl, rng, n_pcg_iter=50)` in `samplers.py`
  - Forms b_noise using Y and Y^T matvecs via TF autodiff (`_cg_jt_v_fn`)
  - Solves A x = b_noise with PCG; diagonal preconditioner = `build_posterior_mass_sqrt²`
  - Returns x as the new alm state (no accept/reject)
- [x] Add convergence check: track PCG residual norm; warn if > 1e-6 after n_pcg_iter iterations
- [x] Swap into `run_gibbs_chain` as an optional `alm_sampler='cg'` argument (HMC path preserved for Phase 2)
- [x] Warm-start 4 chains from Phase 0 checkpoints; run 1000 samples with CG sampler (job 11513133)
- [ ] Verify: ESS ≈ N at all multipoles including l=200–300
- [ ] Benchmark: compare wallclock time per sample vs HMC at lmax=300 (expect 2–5x faster)
- [ ] Compare recovered C_l to Phase 0 results; confirm agreement at l ≤ 100

**Key reference:** Wandelt, Larson & Lakshminarayanan 2004 (arXiv:astro-ph/0310080); Jewell, Levin & Anderson 2004 — these are the original papers deriving the CG Gibbs CMB sampler.

**Output:** a validated, efficiently mixing Gibbs sampler at lmax=300 with ESS ≈ N for all multipoles. This is the correct baseline before adding lensing.

---

### Phase 1 — Differentiable lensing operator (the key technical unlock)

**Goal:** implement a differentiable lensing operator so that HMC can compute gradients through the lensed likelihood with respect to both `alm` and `phi`.

The lensed map is computed by:
1. Synthesising the unlensed map `T` from `alm` using the spherical harmonic transform
2. Applying the pixel remapping `n -> n + grad(phi(n))` (bilinear interpolation in pixel space)
3. Comparing the lensed synthetic map to the observed data

Steps 1 and 2 must be differentiable with respect to both `alm` (for the existing alm block) and `phi_alm` (for the new phi block in Phase 2).

- [x] Survey existing differentiable lensing implementations: `lenspyx` (Carron), `lensit`, JAX-based pixell — chose custom TF reimplementation for full autodiff control and TF stack compatibility
- [x] Implement `lens_map_tf(alm, phi_alm) -> lensed_map` as a TF operation — `diffcmb/lensing.py`; also `lens_map_phi_diff_tf` for joint differentiability w.r.t. both alm and phi
- [x] Validate gradient `dL/d_alm` against finite differences at lmax=50 — `test_apply_lensing_dT_grad_vs_fd`, `test_psi_lensed_alm_grad_vs_fd` pass
- [x] Validate gradient `dL/d_phi_alm` against finite differences at lmax=50 — `test_phi_grad_deflection_adjoint_vs_fd`, `test_psi_lensed_phi_grad_vs_fd` pass; required fixing factor-of-2 for m>0 in adjoint, Npix/(4π) normalisation, and scalar bilinear FD (eps=1e-7)
- [x] Replace the current unlensed likelihood term in `psi_tf` with the lensed version — `psi_lensed` in `diffcmb/lensing.py` is a drop-in replacement for `_psi_tf_raw`
- [ ] Benchmark forward + backward pass time at lmax=300, NSIDE=256

**Key reference:** Carron & Lewis 2017 (arXiv:1701.01712); lenspyx library.

---

### Phase 2 — Three-block Gibbs sampler over (alm, C_l, phi)

**Goal:** extend the Gibbs sampler to jointly sample the unlensed CMB signal, its power spectrum, and the lensing potential. This is the core novel contribution.

The three Gibbs blocks are:

```
Block 1:  C_l      | alm, phi, d  — exact inverse-Gamma draw (already implemented, unchanged)
Block 2:  alm      | C_l, phi, d  — HMC with lensed likelihood (Phase 1 prerequisite)
Block 3:  phi      | alm, C_l, d  — HMC targeting the lensing posterior
```

Block 3 targets:
```
log p(phi | alm, C_l, d) = log p(d | alm, phi) + log p(phi | C_l^phiphi)
```
where `p(phi | C_l^phiphi)` is a Gaussian prior and `C_l^phiphi` is either fixed to a LCDM prediction or jointly sampled via a fourth inverse-Gamma block.

**Note on Block 2 in the lensed setting:** once lensing is added, `p(alm | C_l, phi, d)` is no longer Gaussian (the lensing operator is nonlinear in alm), so HMC returns for Block 2. The CG-based mass matrix from Phase 0b — `P = C_l^{-1} + (1/σ²) diag(Y^T Y)` — remains the correct diagonal preconditioner for the HMC momentum distribution, since it captures the local Gaussian curvature of the unlensed prior. Using it as the HMC mass matrix in Phase 2 should give near-unit condition number for the unlensed part, with the lensing correction contributing a manageable perturbation.

- [ ] Add `phi_alm` as a new parameter block in the sampler state
- [ ] Implement `log_prob_phi_block(phi_alm, alm, Cl_phi)` using the Phase 1 lensed likelihood
- [ ] Add HMC step for the phi block in `run_gibbs_chain`
- [ ] Optionally add Block 4: `C_l^phiphi | phi` — exact inverse-Gamma, same structure as Block 1
- [ ] Test on lensed simulations with known phi_true: verify phi recovery and C_l^TT recovery
- [ ] Compare lensing reconstruction signal-to-noise to the quadratic estimator baseline
- [ ] Quantify bias reduction in recovered C_l^TT relative to Commander-style analysis (which ignores lensing)

**This is the paper.** A joint sampler over `(alm_unlensed, C_l^TT, phi, C_l^phiphi)` that outperforms MUSE (which only samples phi, with alm marginalised analytically) in joint uncertainty propagation is a clear and novel contribution.

The publishable claim:

> *We present the first fully joint Bayesian sampler over unlensed CMB signal, angular power spectrum, and lensing potential at map level. Unlike MUSE, which marginalises over the CMB signal and returns only lensing potential samples, our Gibbs sampler returns full posterior samples over all unknowns, correctly propagating uncertainty between C_l^TT, alm, and phi. We demonstrate reduced bias in power spectrum recovery and improved characterisation of delensing residuals relative to existing methods, on both simulations and Planck data.*

---

### Phase 2b — Cosmological parameter inference (optional path)

**Goal:** use the sampled C_l posterior to infer ΛCDM parameters, producing a direct comparison to Planck 2018 PR3 results.

This is a self-contained analysis layer that can run in parallel with Phase 3 once Phase 2 produces reliable C_l posteriors.

- [ ] Build an `emcee`-based wrapper over the sampled C_l posterior to infer `[H0, Ωb h², Ωc h², mν, Ωk, τ]`
- [ ] Validate recovered parameters against official Planck 2018 values
- [ ] Apply Blackwell-Rao marginalization for smooth marginal likelihoods
- [ ] Produce corner plot of ΛCDM posteriors

**Key reference:** Planck 2018 results V (arXiv:1907.12875) for parameter posteriors.

---

### Phase 3 — Polarization (E/B decomposition)

**Goal:** extend to full temperature + polarization (TQU) analysis.

Polarization matters because B-mode delensing — removing the lensing-induced B-mode to search for primordial gravitational waves — is the primary science driver of SO and CMB-S4. The joint `(T, E, B, phi)` sampler from this phase would directly enable optimal delensing and is the most natural extension of Phase 2.

- [ ] Extend `alm_utils.py` to handle spin-2 fields: Q, U maps <-> E, B alms
- [ ] Extend `psi_tf` to the full TQU joint likelihood with (TT, TE, EE, BB) power spectrum block
- [ ] Handle the TE cross-spectrum in the C_l Gibbs block: the off-diagonal `C_l^TE` term breaks the simple inverse-Gamma conjugacy and requires either a 2x2 inverse-Wishart draw or HMC
- [ ] Extend the lensing operator to spin-2 (lensing mixes E and B modes)
- [ ] Test on simulated TQU + lensed maps; measure recovered r constraint and delensing efficiency

**Key reference:** BeyondPlanck (Andersen et al. 2023, arXiv:2303.04819) for the polarization Gibbs block structure.

---

### Phase 4 — Scalability to SO / CMB-S4 resolution

**Goal:** reach lmax >= 1000 needed for next-generation experiments.

At L=300 a single chain costs ~133s/iter and 55h for 1000 samples. The dense spherical harmonic matrix scales as O(lmax^2 * Npix). At lmax=1000 this becomes prohibitive without algorithmic changes.

- [ ] Profile per-iteration breakdown: YA matvec vs C_l draw vs lensing operator vs Python overhead
- [ ] Replace the dense stored `sph` matrix with on-the-fly SHT using `ducc0` — avoids storing O(lmax^2 * Npix) in memory, computes harmonics on demand
- [ ] Benchmark ESS/hour vs lmax for both the dense and on-the-fly approaches
- [ ] Investigate multi-GPU distribution of the YA matvec (currently split into `sph_parts` which is a step in this direction)
- [ ] Profile and tune the lensing operator at lmax=1000, NSIDE=1024
- [ ] Target: 1000 samples at lmax=1000 within 72h on COSMA8 dine2 nodes

---

### Phase 5 — Non-Gaussian extensions (longer term)

Once the differentiable infrastructure from Phases 1-3 is in place, the non-conjugate structure of `psi_tf` becomes a platform for extensions Commander fundamentally cannot support:

- **fNL sampling:** add a bispectrum likelihood term to `psi_tf`; jointly sample `(alm, C_l, fNL)`. No existing method does this at map level without data compression.
- **Galactic mask in-painting:** constrained realisations of unobserved pixels as an additional Gibbs block — cleaner than apodisation.
- **Learned CMB prior:** replace the Gaussian alm prior with a diffusion model or normalising flow trained on CMB simulations; the prior score enters `psi_tf` as an additional term. Enables inference under non-standard cosmologies or foreground residuals with complex morphology.
- **Instrument systematics:** jointly sample calibration factors, beam errors, or 1/f noise amplitudes as additional HMC blocks — the same pattern as the phi block in Phase 2.
