# Project Roadmap: Advancing CMB Sampling to Publication

This document tracks the progression from the MSci thesis (2026) to a journal-worthy publication in CMB cosmology.

## Phase 1: Sampler Stabilization (COMPLETE)

**Goal**: Resolve the high-dimensional instabilities identified in the MSci thesis.

- [x] **Root Cause Analysis**: Diagnosed step-size collapse and control-loop lag in high-D space ($d \approx 40{,}000$).
- [x] **Algorithm Implementation**: Replaced sliding-window adaptation with Robbins-Monro stochastic approximation.
- [x] **Baseline Results**: Produced comparative dashboard vs. HMC and NUTS on synthetic L=200 data.
- [x] **Verification (partial)**: Achieved $R$-hat = 1.000 on $\ln C_\ell$ at L=200 (synthetic data).

  > **Correction (2026-06-14):** The "perfect convergence" milestone only verified $\ln C_\ell$ mixing. Alm R-hat was not measured at that stage. When real Planck data was applied and alm R-hat was measured, a separate blocking issue was discovered — see Phase 2.

---

## Phase 2: Scientific Scaling (IN PROGRESS)

**Goal**: Reach resolutions capable of capturing higher-order acoustic oscillations.

### The Alm Mixing Failure (identified 2026-06-14)

Running with real Planck data exposed a critical bug where the alm chains are effectively frozen despite nominal acceptance rates. Summary of all runs:

| Run | Data | Precision | Final step | Accept | logp std | R-hat alm median | Status |
|-----|------|-----------|-----------|--------|----------|-----------------|--------|
| Gibbs L=200 | Synthetic | float32 | ~1e-7 | ~63% | ~10.5 | *not measured* | Cl OK only |
| Gibbs L=200 | Real Planck | float32 | ~1e-7 | ~63% | 10.5 | 12 963 | BROKEN |
| Gibbs L=300 | Real Planck | float32 | 1e-7 | ~38% | 13.2 | 46 458 | BROKEN |
| Gibbs L=300 (50-sample test) | Real Planck | float64 | — | **84%** | **55.5** | pending | PROMISING |

**Root cause — float32 gradient noise in the spherical harmonic matvec:**

The matrix $Y$ (shape: $N_\text{pix} \times N_\text{alm}$) is stored as `complex64`. The HMC gradient

$$\nabla_a \psi \;=\; \text{Re}\!\left(Y^\dagger\, N^{-1}(Ya - d)\right)$$

is accumulated in float32 across $N_\text{pix} \approx 607{,}000$ unmasked Planck pixels. This introduces $\mathcal{O}(10^{-3})$ absolute noise per gradient component. In the HMC leapfrog this appears as spurious energy injection, causing excessive rejections in early burn-in. The Robbins-Monro rule drives the step size to its floor ($10^{-7}$), after which each accepted step moves $\lesssim 2 \times 10^{-6}$ in whitened alm space — effectively zero. The chains produce 35–65% nominal acceptance but are frozen.

**Why synthetic data escaped this:** The synthetic L=200/NSIDE=128 test case has $\sim 3\times$ fewer unmasked pixels than the real Planck sky at NSIDE=256, so accumulated gradient noise stays below the HMC rejection threshold. The $\ln C_\ell$ block always converges because those parameters are sampled analytically (inverse-Gamma), not by HMC.

**The fix — double precision (`complex128`):** A 50-sample test on L=300 real data immediately showed `accept_rate = 0.84` and `logp_std = 55.5` vs. 10.5 for the stuck float32 chains. The fix is confirmed; the full 4-chain run is queued.

**Memory implication:** `complex128` doubles the $Y$ matrix from ~218 GB to ~437 GB. Runs must use the high-memory `dine2` nodes (1.5–2 TB RAM).

---

### Milestones

- [x] **L=300 infrastructure** — memory chunking solved (21-part dynamic split); initial float32 chains completed but alm broken.
- [ ] **L=300 double-precision chains** — Job 11383627 (`cmb_gibbs_L300_dp`, array 1–4), queued on `dine2`, scheduled start **2026-06-16 18:00** (waiting for `gc005` reservation).
- [ ] **L=200 double-precision baseline** — run a clean L=200 real-data Gibbs chain in `complex128` to establish a verified baseline with both $C_\ell$ and alm R-hat < 1.1 before scaling further.
- [ ] **Milestone L=800**: Reach the 3rd acoustic peak.
    - [ ] Optimize Rust-extension pre-computation for 27B+ harmonic values.
    - [ ] Implement sample thinning to manage multi-TB chain output.
    - [ ] Confirm memory budget for `complex128` at NSIDE=512.

---

## Phase 3: Cosmological Parameter Inference

**Goal**: Move from power spectrum recovery to physical constant estimation.

- [ ] **Inference Engine**: Build a wrapper to use `emcee` over the sampled $C_\ell$ posterior.
- [ ] **Lambda-CDM Constraints**: Produce posterior distributions for $[H_0, \Omega_b h^2, \Omega_c h^2, m_\nu, \Omega_k, \tau]$.
- [ ] **Scientific Validation**: Compare recovered parameters to official Planck 2018 PR3 results.

---

## Phase 4: Publication & Documentation

**Goal**: Prepare results for submission to *Astronomy & Computing* or *JCAP*.

- [ ] **Performance Benchmarking**: Document the speedup over standard TFP implementations.
- [ ] **Error Analysis**: Apply Blackwell-Rao marginalization to provide smooth marginal likelihoods.
- [ ] **Final Graphics**: Produce "Figure 1" (Full-sky recovered map) and "Figure 2" (Parameter corner plot).

---

*Last Updated: 2026-06-14*
