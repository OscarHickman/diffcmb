# Research Roadmap: Differentiable Bayesian CMB Analysis

*Last substantive revision: 2026-07-04 (Phase 0c Step 6 started: run_gibbs_chain/run_sampler.py now actually wire use_block_correction/m_group_size through to the messenger sampler -- previously silently ignored; isolated-cost lmax=300 benchmark done, full end-to-end per-sweep timing benchmark running -- see Phase 0c Step 6).*

## The Honest Critique First

This codebase currently implements a Gibbs sampler over `(alm, C_l)` with:
- A Gaussian CMB prior `p(alm | C_l)`
- A white-noise Gaussian likelihood `p(d | alm)`
- Exact inverse-Gamma draws for `C_l | alm`
- HMC (Phase 0) and exact CG draws (Phase 0b) for `alm | C_l`

This is algorithmically what Commander (Jewell et al. 2004, Wandelt et al. 2004) does, and Commander has been running on real Planck data in production for 15 years. Replicating it in Python is not a publishable contribution on its own.

**And the differentiable-lensing extension is not new either.** CMBLensing.jl (Millea; Julia, GPU, autodiff via Zygote) already maximises and *samples* the joint lensing posterior `p(f, phi, theta | d)`: Millea, Anderes & Wandelt 2020 (arXiv:2002.00965) ran a joint Gibbs/HMC sampler over the unlensed polarization field, lensing potential, and `r` on 650 deg² flat-sky simulations, and Millea et al. 2021 (arXiv:2012.01709) applied the joint MAP + MUSE machinery to real SPTpol data. Any version of this roadmap that claims "first joint sampler over (alm, C_l, phi)" without qualification is factually wrong and would be caught at referee stage.

**The question is therefore: what can this codebase do that neither Commander nor CMBLensing.jl can?**

## Novelty: the one-paragraph answer

> CMBLensing.jl is a **flat-sky** code: its lensing operators (LenseFlow) and its joint HMC sampler operate on FFT-based periodic patches of a few hundred deg², appropriate for SPT/S4-style deep small-area surveys at l ≳ 500. Its authors state explicitly (MUSE paper, arXiv:2112.09354; package docs) that *"HMC sampling on the curved sky is at present slightly out of reach"* — which is why the production SPT-3G analysis (arXiv:2411.06000, 2024) uses MUSE, a marginal score-expansion approximation, rather than joint sampling. Commander is full-sky but non-differentiable and conjugate-only. **DiffCMB sits in the empty cell of that 2×2: a full-sky (curved-sky, HEALPix) differentiable joint Gibbs sampler over `(alm_unlensed, C_l, phi)`, with exact Commander-style CG draws for the Gaussian block and HMC for the non-conjugate blocks.** The science this uniquely enables is *full-sky, low-to-mid-l* field-level lensing inference — exactly the regime of LiteBIRD delensing and Planck/LiteBIRD large-scale reanalysis, where flat-sky patches are not an option (see arXiv:2507.22618 for the LiteBIRD full-sky lensing forecast, which still relies on quadratic/iterative estimators, not sampling).

Corollaries that shape everything below:

1. **The paper claim must be scoped to "full-sky / curved-sky".** The flat-sky joint-sampling problem is solved; do not compete there.
2. **The clock is ticking.** The MUSE authors say curved-sky MUSE is coming "in the near future". MUSE is marginal (no joint samples, no per-mode uncertainty propagation), so a curved-sky *sampler* remains distinct — but the window for "first field-level curved-sky lensing inference" is finite. Phases 1.5–2 are the critical path; everything else is subordinate.
3. **CMBLensing.jl is the mandatory benchmark.** Phase 2 validation must include a comparison against it (MUSE and/or joint HMC) on a matched simulation, or referees will ask why not.
4. **Temperature-only at lmax=300 is a demonstration, not a science result.** The science case (delensing, LiteBIRD) lives in polarization (Phase 3). Phase 2 is the methods paper; Phase 3 is the science paper.

---

## The Gap in Current Research

The CMB we observe is the **lensed** field:

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
| Commander | Full-sky Gibbs over (alm, C_l); ignores lensing | Biased C_l; non-differentiable, conjugate-only |
| Quadratic estimator (Okamoto & Hu 2003) | Estimates phi from data alone | Suboptimal below QE noise floor; no joint uncertainty propagation |
| Carron & Lewis 2017 / delensalot (Belkner et al. 2023) | Iterative MAP of `p(phi | d)`, curved sky | Point estimate only, no posterior samples |
| CMBLensing.jl joint sampler (Millea, Anderes & Wandelt 2020) | Joint Gibbs/HMC over (f, phi, r) | **Flat sky only** (~650 deg² patches); curved-sky HMC "out of reach" per authors |
| MUSE (Millea & Wandelt 2021; SPT-3G 2024, arXiv:2411.06000) | Approximate marginal inference of phi and CMB bandpowers | No joint samples; Gaussianised marginal; curved-sky version announced but not delivered |
| LiteBIRD lensing forecast (arXiv:2507.22618) | Full-sky QE + iterative reconstruction | Not Bayesian sampling; no C_l–phi joint posterior |

**The gap (scoped honestly):** nobody has a working **full-sky, curved-sky** joint posterior sampler over `(alm_unlensed, C_l, phi)` at map level. For LiteBIRD — whose entire science case is full-sky, low-l B-modes and whose delensing must operate on the curved sky — and for full-sky Planck reanalysis, this is the missing tool. Flat-sky codes structurally cannot address it.

### Why this repo is the right starting point

- `psi_tf` is differentiable — gradients of the log-likelihood flow through the lensing operator w.r.t. both `alm` and `phi` (validated, Phase 1)
- The whole stack is HEALPix/full-sky from day one — the hard part CMBLensing.jl lacks
- The Gibbs structure (exact inverse-Gamma C_l draws, exact CG alm draws in the unlensed limit) gives per-block validation Commander-style, rather than one monolithic HMC over everything
- The HMC infrastructure for non-conjugate blocks already exists and is wired in (Block 3 committed 2026-07-01)

### The counter-risks (named, so they get managed)

| Risk | Impact | Mitigation |
|---|---|---|
| Curved-sky MUSE ships first | Loses "first curved-sky field-level lensing inference" framing | MUSE is marginal, not a sampler; keep the joint-samples/uncertainty-propagation claim, and move fast on Phases 1.5–2 |
| Dense `sph` matrix hardware wall | Phase 2 HMC at lmax=300 infeasible on available GPUs (measured: 0.11 leapfrog steps/s) | **Phase 1.5 below — this is now a hard gate, with a named plan** |
| One-person bandwidth | Phases stall; stale claims accumulate | Roadmap enforces a single critical path (0b → 1.5 → 2); Phases 2b/5 explicitly parked |
| lmax=300 too low for lensing S/N in TT | Weak detection in the methods paper | Acceptable for a methods demonstration on simulations (phi_true known); the S/N story is Phase 3 polarization |

---

## Roadmap

**Critical path: Phase 0b (running) → Phase 1.5 (hardware gate) → Phase 2 (methods paper) → Phase 3 (science paper).** Phases 2b, 4-beyond-1.5, and 5 are parked and must not consume GPU-hours or working time until Phase 2 is validated.

### Phase 0 — Establish and validate the baseline ✓ (complete)

**Goal:** demonstrate the L=300 Gibbs sampler recovers the correct CMB power spectrum from real Planck data, establishing that the infrastructure is correct before extending it.

**Historical run summary** (all chains to date):

| Run | Precision | Accept | logp std | R-hat C_l med | R-hat alm med | ESS C_l | Status |
|-----|-----------|--------|----------|--------------|---------------|---------|--------|
| Gibbs L=200 synthetic | float32 | ~63% | ~10.5 | 1.000 | not measured | ~1550 | C_l only |
| Gibbs L=200 real | float32 | ~63% | 10.5 | 1.000 | 17 619 | ~1550 | FROZEN |
| Gibbs L=300 real | float32 | ~38% | 13.2 | 1.000 | 58 112 | ~1550 | FROZEN |
| **Gibbs L=300 real** | **float64** | **71%** | **26 051** | **1.026** | **2.64** | **385** | **Phase 0 ✓** |
| CG L=300 real | float64 | 100% | — | — | — | — | Phase 0b ▶ running |

Float32 chains show false convergence: C_l R-hat ≈ 1.000 but alm R-hat = 18k–58k. Root cause: float32 gradient noise in the `Y^H` matvec across ~607k pixels drives HMC step to floor ~1e-7. **This finding constrains every precision decision below: naive fp32 is disqualified; any mixed-precision scheme must keep fp64 accumulation and be validated against the fp64 chains.**

- [x] Get 4-chain L=300 float64 run — completed 2026-06-27 (4 chains × 1000 samples)
- [x] Compute R-hat — **median 1.026, max 1.085, 0% exceed 1.1** ← C_l fully converged
- [x] Measure ESS per C_l — **median 385/800 post-burn (48% efficiency); ACF drops to 0 at lag 1**
- [x] Tag MSci project baseline in git — `v0.0-msci` at commit `2f7441c` (unlensed Gibbs + CG, pre-lensing)
- [x] Plot recovered C_l vs Planck official power spectrum (Commander/Plik) — `results/analysis/planck_comparison.png` (2026-06-27, HMC chains). Confirms known high-l excess (5-10x CAMB/Planck above l~100): this is the HMC mixing failure Phase 0b's CG sampler targets, not a new issue. Re-plot against CG chains once job 11552530 completes.

**Phase 0 high-l drift (to be resolved by CG):**

| Multipole | lnCl drift significance | ESS HMC (chain 4) | ESS CG (expected) |
|-----------|------------------------|-------------------|-------------------|
| l=2 | ~0.5σ | ~1000 | ~1000 |
| l=10 | ~1σ | ~900 | ~1000 |
| l=50 | ~3σ | ~900 | ~1000 |
| l=200 | **11σ** | ~41 | **~1000** |

**Output:** validated C_l posterior with ESS ≈ N at all l. This is table-stakes before claiming anything about extending the model.

---

### Phase 0b — CG-based exact alm sampler ▶ (chains running)

**Goal:** replace the HMC alm | C_l step with a direct conjugate-gradient draw. This gives IAT = 1 at every multipole by construction, eliminating the high-l drift problem entirely without requiring longer chains.

**Why this is possible:** the alm | C_l Gibbs conditional is exactly Gaussian:

```
p(alm | C_l, d) ∝ exp(-½ alm^T A alm + alm^T Y^T N^{-1} d)
A = C_l^{-1}(diag, grouped by l) + (1/σ²) Y^T Y
```

An exact independent draw is obtained by solving `A x = b_noise` with `b_noise = C_l^{-1/2} ω₁ + (1/σ) Y^T ω₂ + (1/σ²) Y^T d`, ω₁ ~ N(0, I_alm), ω₂ ~ N(0, I_pix) fresh each iteration; x is a sample from the exact conditional, no accept/reject. Solved via PCG with the diagonal preconditioner `P = C_l^{-1} + (1/σ²) diag(Y^T Y)` (already computed as the mass matrix).

**This is what Commander does.** The critical difference is that Commander is Fortran with conjugate priors throughout; once lensing is added in Phase 2, the alm conditional is no longer Gaussian and CG alone breaks — HMC returns, with the CG machinery surviving as the preconditioner/mass matrix.

- [x] Implement `sample_alm_cg(model, current_lncl, rng, n_pcg_iter=50)` in `samplers.py` (Y/Y^T matvecs via TF autodiff, PCG residual warning, `alm_sampler='cg'` option in `run_gibbs_chain`)
- [x] First production attempt (job 11513133) — **timed out after 3 days, PCG residual stuck flat at ~5.3e4**
- [x] Root-caused and fixed: `matvec_on_device`'s `tf.custom_gradient` (`model.py`) left each `sph_part`'s gradient contribution on its own device before returning. With `sph_parts` split across >1 GPU (production layout: 41 parts across 2 GPUs + CPU), TF silently corrupts cross-device accumulation of the shared `alm` gradient — the CG operator `A p := ∇ψ(p) − ∇ψ(0)` was measurably non-linear (‖A(2p)‖/‖A(p)‖ ≈ 59000) and non-symmetric. Fix: move `grad_x` to `/CPU:0` before returning from `grad()`. Regression check: `scripts/verify_cg_matvec.py`, `tests/test_cg_matvec.py`
- [x] Validated the fix in isolation (job 11552527, 2026-07-01): A symmetric (1.8e-12 rel. error), positive-definite, linear (3.6e-13), PCG residual monotone decreasing
- [ ] Production CG chains: job 11552530 (array 1-4, dine2 gc001–gc007) **running since 2026-07-01, ~1 day elapsed**. Checkpoints truncated to the 1000 clean Phase 0 HMC samples; the broken-operator CG samples from 11513133 discarded (backed up in `buggy_cg_backup/`)
- **⚠ Chain 1 flat-residual anomaly (2026-07-02) — operator cleared, cause still open (superseded below):** chain 1's log (`logs/cmb_cg_L300_11552531_1.out`) shows the PCG residual flat at ~5.34e4 across all 228 logged solves since iteration 0, the same magnitude as the pre-fix `11513133` symptom. Ran `scripts/debug_cg.py` under the exact production device layout (`CUDA_VISIBLE_DEVICES=0`, real data, lmax=300, 41 parts, 1 GPU + 40 CPU — job `11555267`, with a job-private `$TMPDIR` to rule out node-local autograph cache staleness, see below): `A` is symmetric (1.9e-12), positive-definite, linear (3.6e-13), and **the PCG residual decreases properly from 2.49e7 to 2.02e5 over 10 iterations** — the matvec operator itself is sound at production scale, so the 66f169c fix does generalize to the GPU+CPU layout. The diagnostic ran without a checkpoint (fresh prior-cls `lncl`), so it doesn't 1:1 replicate chain 1's exact resumed state.
- **Found + fixed: node-local autograph cache collision (2026-07-02).** Two earlier attempts at this same diagnostic (jobs `11555241`, `11555248`) crashed with `TypeError: tf___ensure_tf_tensors() takes 1 positional argument but 2 were given` — a nonsensical error thrown from inside an autograph-traced call to an unrelated method. Every smaller-scale repro (lmax=10/30, real data, same GPU node) ran cleanly, isolating the trigger to lmax=300's autograph trace specifically on a node (`gc004`) that had hosted earlier attempts. Autograph writes transpiled sources to `$TMPDIR/__autograph_generated_file*.py`, which is node-local and not job-scoped by default; setting a job-private `$TMPDIR` (`scripts/debug_cg_single_gpu.slurm`) fixed it on the first retry. This was a real bug worth fixing but turned out to be unrelated to the flat-residual symptom (see below).
- **✓ Root-caused (2026-07-03): the flat residual is NOT a bug in the matvec or the trace cache — it's the diagonal (Jacobi) preconditioner failing on a masked sky.** Job `11555835` (4 chains, resubmitted 2026-07-02 post-TMPDIR-fix, ran 22h) showed the exact same flat `|r|≈5.3e4` (vs tol=1e-6) on **every single one of the ~240 PCG calls per chain, for the entire run** — i.e. not an anomaly specific to one resumed checkpoint. Cross-checked against the two earlier runs (`11513133`, `11552530`, both pre-TMPDIR-fix): identical residual range (5.336e4–5.39e4) in all three runs, so the TMPDIR/matrix-free-SHT fixes changed nothing here (expected — neither touches the PCG solve path). Diagnosis, in order of investigation:
  1. **Confirmed real but secondary bug:** `_build_inv_cl_diag` (`samplers.py`) and `build_posterior_mass_sqrt` (`model.py`) built the alm precision diagonal as `1/C_l` uniformly for every m, but the forward likelihood (`_psi_tf_raw`'s `l_weights`/`alm_weights` factor of 2.0 for m>0) implies the true precision is `2/C_l` for m>0 real/imag dof and `1/C_l` only for m=0 (standard convention: a complex a_lm for m>0 splits variance `C_l` evenly across Re/Im, each with variance `C_l/2`). **Fixed** (commit pending this update): both functions now apply the correct m-dependent factor. This is a genuine correctness fix — `_build_inv_cl_diag` feeds the sampled RHS directly, so pre-fix, real production samples for m>0 alm had prior-noise variance off by √2 low, independent of CG convergence. It does **not** explain the stall: PCG convergence is invariant to a uniform rescale of the preconditioner, and only the ~lmax-2 m=0 modes (out of ~89k) were relatively mis-scaled.
  2. **Confirmed dominant cause via `scripts/debug_cg_masksky.py`** (small lmax=20/NSIDE=16 synthetic sky, run on CPU in a few seconds): the same diagonal preconditioner reduces the residual by a factor **1.3e-7 over 200 iterations full-sky (f_sky=1.0)**, but only **2.5e-5 over 200 iterations masked (f_sky=0.74)** — 4 orders of magnitude worse from masking alone, at a problem 200x smaller than production. `build_posterior_mass_sqrt`'s docstring claim ("nearly diagonalises the posterior") is only true near `f_sky≈1`; a sharp sky cut induces substantial off-diagonal mode-coupling in `J^T N^{-1} J` that no diagonal preconditioner captures — a known failure mode in the CMB Gibbs literature (part of the original motivation for messenger fields, Elsner & Wandelt 2013).
  3. **"Just run more PCG iterations" is not viable.** From the 22h `11555835` log: 241 PCG calls / 80,762s ⇒ 335s/call ⇒ **6.7s/iteration** (matrix-free-SHT gradient over ~89k masked pixels). Extrapolating the observed conditioning degradation to lmax=300 (residual is *completely flat* at 50 iterations, not slowly decreasing), plausibly 10⁴–10⁵ iterations are needed — 19–190 hours *per single Gibbs step*, incompatible with a chain needing hundreds to thousands of Gibbs steps and COSMA's 22h job ceiling.
  4. **Action taken:** job `11555835` (all 4 chains) **cancelled 2026-07-03** — its samples were being generated from PCG solves that never approached the target tolerance and cannot be trusted as valid Gibbs draws (compounded by the pre-fix preconditioner bug above). See Phase 0c below for the fix.
- [ ] On completion of Phase 0c: verify ESS ≈ N at all multipoles including l=200–300; logp plateau; C_l agreement with Phase 0 at l ≤ 100; re-plot Planck comparison
- [ ] Benchmark wallclock per sample vs HMC at lmax=300 (expect 2–5x faster)

**Key reference:** Wandelt, Larson & Lakshminarayanan 2004 (arXiv:astro-ph/0310080); Jewell, Levin & Anderson 2004.

**Output:** a validated, efficiently mixing Gibbs sampler at lmax=300 with ESS ≈ N for all multipoles. This is the correct baseline before adding lensing. **Blocked on Phase 0c** (plain diagonal-preconditioned CG cannot converge on the real, masked-sky (f_sky=0.772) problem within any tractable iteration budget).

---

### Phase 0c — Messenger field for masked-sky alm sampling ▶ (started 2026-07-03)

**Goal:** replace/augment `sample_alm_cg`'s diagonal-preconditioned PCG — which Phase 0b showed cannot converge on the real masked sky within any tractable iteration budget (§ above) — with the messenger-field method (Elsner & Wandelt 2013, "A novel approach to Gaussian constrained sampling with messenger fields"; used in Commander and similar production Gibbs codes for exactly this problem). This is now the critical-path blocker for Phase 0b's completion and therefore for everything downstream.

**Why it fixes the conditioning problem:** the current operator `A = diag(1/C_l) + J^T N^{-1} J` is ill-conditioned because `J^T N^{-1} J` (the masked/inhomogeneous-noise term) is far from diagonal in harmonic space. The messenger method never inverts that operator. It introduces a latent field `t`, defined on the same *full, unmasked* pixelization as the map, via the generative reparametrisation `t | s ~ N(A s, T)`, `d | t ~ N(t, N-T)` with `T = τ²·I` chosen so `τ² ≤ min(N_ii)` over observed pixels (masked pixels formally have `N_ii = ∞`, i.e. no constraint from `t|s,d` there; marginalising `t` recovers the original `d | s ~ N(As, N)`). Gibbs then alternates two closed-form conjugate-Gaussian updates:

```
t | s, d  ~  N( (T⁻¹+(N-T)⁻¹)⁻¹ (T⁻¹ A s + (N-T)⁻¹ d),  (T⁻¹+(N-T)⁻¹)⁻¹ )   — pointwise per pixel
s | t     ~  N( (S⁻¹+T⁻¹)⁻¹ T⁻¹ Aᵀ t,                    (S⁻¹+T⁻¹)⁻¹ )      — pointwise per (l,m),
                                                            since T=τ²I isotropic + AᵀA=I ⇒ diagonal
```

where `s` is the full-sky signal (harmonic coefficients) and `S = diag(C_l)` is the harmonic-space prior. **Neither step requires a linear solve, let alone CG** — `t|s,d` is a per-pixel Gaussian draw, and `s|t` is a per-`(l,m)` Gaussian draw after one full-sky forward SHT (`t → Aᵀt`). The mask only ever enters through `(N-T)⁻¹` in the first step; the harmonic-space step is always perfectly conditioned regardless of `f_sky`. Cost: no longer 1 iteration = 1 autodiff gradient over masked pixels (6.7s at production scale); instead each messenger sub-iteration is 1 full-sky forward SHT + 1 backward SHT (map↔alm), both O(lmax²·Npix) — comparable to or cheaper than the current matrix-free SHT gradient, and this method's known convergence behaviour (linear in `τ²`, typically tens to low hundreds of sub-iterations for `τ²` near the true noise floor) is what makes it produce a usable sample per Gibbs step where the current method does not.

**Implementation plan:**

- [x] **Step 1 — standalone correctness validation of the two closed-form conditionals**, decoupled from the TF/production model. `diffcmb/diffcmb/messenger.py` (`sample_t_given_s`, `sample_s_given_t_orthonormal`, `run_messenger_gibbs`) plus `tests/test_messenger.py`: a dense, brute-force Bayesian linear-Gaussian toy problem (random orthonormal `A` with `AᵀA=I` — the same property a full-sky SHT synthesis has — ~30% of pixels given a huge noise variance to emulate masking) where the *true* posterior mean/covariance is computable by direct dense linear algebra (`Λ = S⁻¹ + AᵀN⁻¹A`, generally non-diagonal because of the mask, even though `AᵀA=I`). Runs the messenger Gibbs sampler for thousands of sweeps and checks the empirical posterior mean and *full covariance* (not just the diagonal) match the dense reference to within Monte Carlo error. **Caught a real derivation bug in the first implementation attempt**: the initial `t | s, d` draw used the naive-looking `N⁻¹`/`T⁻¹` combination (`precision = N⁻¹+T⁻¹`) rather than the correct conjugate update using the *reduced* noise `(N-T)`, which the `t|s~N(As,T)`, `d|t~N(t,N-T)` generative construction actually requires; the naive version is a plausible-looking formula that is subtly wrong and converged to a visibly biased posterior mean (max error 0.22 against a dense reference, ~20x larger than Monte Carlo noise at 20k samples) — caught precisely because Step 1 checks against an exact dense reference before any of this touches production code. Fixed by computing `Ninv_red = Ninv/(1-τ²·Ninv)` (the `(N-T)⁻¹` term, written to stay finite as `Ninv→0` for masked pixels); after the fix, empirical mean matches the dense posterior to 0.026 (consistent with MC error) and the full covariance matches to <3% relative error including all off-diagonal mask-induced coupling terms. This isolates "is the algorithm's math right" from "does it wire correctly into `_psi_tf_raw`/ducc0", which is Step 2.
- [x] **Step 2 — wire into the real model** (2026-07-03). `sample_alm_messenger(model, lncl_np, rng, n_messenger_iter, tau2, s0)` in `samplers.py`: a lazily-built full-sky `HealpixSHT` (`model._sht_full`, `unmasked_idx=None`) supplies `A_action`/`At_action`; `_packed_to_alm_ho`/`_alm_ho_to_packed` convert between the packed real+imag layout (`sample_alm_cg`'s convention) and complex healpy-ordered alm. **Caught two real bugs, both by probing `A_action` against a dense reference (mirroring Step 1's discipline) rather than trusting the analytic derivation:**
  1. `AᵀA` is **not** `norm_const·I` for a scalar `norm_const` — it needs the same m=0-vs-m>0 factor-of-2 weight as `_build_inv_cl_diag` (`_build_full_sky_norm_diag`): probing unit alm vectors showed m>0 modes carry exactly 2x the map-space norm of m=0 modes (real spherical-harmonic basis convention), which a flat scalar guess missed by up to 100% for individual modes even though it happened to be right on average.
  2. Even the corrected *per-mode* weight is only an analytic continuum-limit approximation — a real HEALPix SHT's quadrature is not exact, and empirical probing (`scripts/debug_messenger_masksky.py`) showed the m=0 (zonal) modes' true diagonal is systematically ~1-2% below the analytic guess, enough to bias the messenger sampler's posterior mean by 40-80 posterior standard errors for exactly those modes (found via a per-mode z-score diagnostic, not just an aggregate metric — the aggregate mean|z| looked passable at ~3-10 while specific modes were wildly off). Fixed by `_calibrate_full_sky_norm_diag`, which probes the *exact* diagonal empirically (O(n_alm) full-sky synthesis calls, cached once per model) instead of using the analytic `NPIX/(4π)` constant.
- [~] **Step 3 — validate at small-lmax masked-sky scale — BLOCKED, not passing.** `scripts/debug_messenger_masksky.py` (lmax=10, NSIDE=8, f_sky≈0.69, dense reference built by probing the real `A_action`) surfaced a **third, more serious problem, distinct from the bias above**: using the *exact* empirically-calibrated diagonal with no safety margin, the messenger Gibbs chain **diverges** under real masking (`Ninv=0` at masked pixels) — `|alm|` grows geometrically without bound (confirmed via direct iteration, not just the aggregate validation script). Root cause: `AᵀA`'s off-diagonal terms are small (~1-1.7% of the diagonal, confirmed by direct probing) but masking makes the pixel-space damping factor in the `t|s,d` step vary by an order of magnitude between observed and masked pixels, which is enough for that ~1% off-diagonal perturbation to push the effective single-Gibbs-step transition operator's spectral radius above 1 — an "incompatible conditionals" instability, not a bias problem. This does not appear at all on a full, unmasked sky (tested up to ~500k total sub-iterations with no divergence), only under masking.
  - Mitigation attempted: inflate the calibrated diagonal by a safety margin (`norm_diag_safety_margin`, Gershgorin-style majorisation — over-stating the precision guarantees contraction). Empirically, margins below ~1.002 still diverge (slowly); ≥1.005 is stable at lmax=10/NSIDE=8. **But stability and accuracy trade off directly**: margin=1.02 (chosen as the current default) gives posterior mean errors of ~26 SE (mean) / 89 SE (max) against the dense masked-sky reference; margin=1.1 gives ~54 SE / 203 SE. Both are far outside any acceptable tolerance (contrast Step 1's toy problem, which matched to <8 SE with an *exact* `AᵀA=I` operator).
  - **Conclusion: `alm_sampler='messenger'` is wired and runs, but is NOT statistically validated and must not be used for production inference yet.** The likely clean fix is an exactly-orthonormal full-sky SHT for the messenger's internal operator (e.g. Gauss-Legendre or Driscoll-Healy quadrature, which ducc0 also supports, instead of HEALPix's approximate quadrature) so `AᵀA` is exactly diagonal and no margin/trade-off is needed at all — untried this session. A cheaper partial fix worth trying first: retain the measured off-diagonal structure (a sparse or low-rank correction to the diagonal preconditioner) rather than discarding it entirely.
- [x] **Step 4 — wire `alm_sampler='messenger'` into `run_gibbs_chain`** (2026-07-03), mirroring the `'cg'` branch (warm-started via `s0=current_alm_np`, new `n_messenger_iter` argument). Regression test `tests/test_samplers.py::test_gibbs_chain_messenger_moves_and_stays_bounded` is deliberately a boundedness/smoke test (finite, non-exploding output), **not** a statistical-accuracy test, given Step 3's open issue — accuracy testing should be promoted once Step 3 is resolved.
- [~] **Step 5 — resolve the Step 3 stability/bias trade-off ▶ (root cause confirmed 2026-07-03, scalable fix still open).** Before building anything scalable, validated the hypothesis cheaply: does capturing the *exact* off-diagonal `AᵀA` (not just a better diagonal) actually fix both divergence and bias, or is something else going on? Added `sample_s_given_t_dense`/`run_messenger_gibbs(..., AtA=...)` (`messenger.py`) — an exact conjugate-Gaussian `s|t` update using the full `AᵀA` matrix via Cholesky, only tractable while `n_alm` is small (O(n_alm³) per draw). Two checks, both confirming the off-diagonal coupling is the actual root cause:
  - Toy-problem unit test (`tests/test_messenger.py::test_dense_AtA_correction_matches_reference_where_diagonal_approx_is_biased`): a synthetic near-orthonormal `A` with deliberately injected off-diagonal `AᵀA` structure. The diagonal approximation is measurably more biased than the dense correction; the dense correction matches the true masked posterior to within Monte Carlo error.
  - Real ducc0 HEALPix SHT at `scripts/debug_messenger_masksky.py`'s validation scale (lmax=10, NSIDE=8, f_sky=0.688): diagonal approximation (current production default, margin=1.02) gives mean error 136.67 SE (max) / 33.51 SE (mean) against the dense masked-sky reference — consistent with the ~40-80σ bias already on record. The exact dense `AᵀA` correction brings this down to 7.14 SE (max) / 2.25 SE (mean), a ~20x reduction, **with no divergence** (`max|alm|` over the chain: 14.6, same order as the bounded-but-biased diagonal chain's 13.8). This rules out "wrong problem" — the off-diagonal `AᵀA` term this diagonal approximation discards is both necessary and sufficient to fix the pathology; the remaining ~7 SE residual is plausibly finite-sample MC noise (only 1000×2 thinned samples) rather than a further systematic error, not yet separately confirmed.
  - **Consequence at the time:** the exact dense update is not production-viable on its own (`n_alm ≈ 45000` at lmax=300 → `O(n_alm³)` Cholesky per Gibbs step is infeasible), but it de-risked investing in a scalable approximation of the same off-diagonal structure.
  - **Structure characterisation (2026-07-04, `scripts/analyze_AtA_structure.py`, lmax=16/NSIDE=16 full-sky probe):** off-diagonal `AᵀA` is *not* diffuse/dense-random. 99.7% of the total off-diagonal energy is between same-`m` pairs (differing `L` by an even number only — odd `ΔL` terms are zero to floating-point precision, a parity selection rule); the residual ~0.3% is cross-`m`, decaying roughly an order of magnitude per `Δm=2`. Critically, *within* a fixed `m`, the coupling magnitude does **not** decay with `ΔL` (roughly flat ~1.5e-5×mean(diag) across the whole `L` range tested) — ruling out a banded-in-`L` correction, but consistent with each `(m, parity)` block being close to low-rank (SVD: rank 2 captures 50% of total off-diagonal energy, rank 20 captures 90%). This picked candidate (a) (structured/low-rank correction) over (b) (Gauss-Legendre grid switch, which would have required resampling the real Planck data/mask onto a non-HEALPix grid — a methodological change to the actual likelihood, not just an implementation detail) — no need to touch the data's pixelization at all.
  - **Scalable fix, built and validated (2026-07-04):** `messenger.sample_s_given_t_block`/`build_block_cholesky` (exact per-block conjugate-Gaussian update, block-diagonal by `m`) plus `samplers._calibrate_block_AtA(..., m_group_size=k)`, which groups `k` consecutive `m` values into one block (probed empirically, same `O(n_alm)` synthesis-call cost as the existing diagonal calibration) — capturing the dominant same-`m` coupling exactly within each block, with `m_group_size` trading a bit more cost for capturing more of the residual cross-`m` term. Wired into `sample_alm_messenger(..., use_block_correction=True, m_group_size=...)`. Unit test (`test_block_diagonal_correction_matches_dense_when_AtA_is_exactly_block_diagonal`) confirms the block solve exactly reproduces a dense solve when `AᵀA` truly is block-diagonal. At `debug_messenger_masksky.py`'s validation scale, a sweep over `m_group_size` shows monotonic improvement converging to (and slightly beating) the exact dense correction's accuracy: diagonal approx 93.11/23.83 SE (max/mean) → `m_group_size=1`: 34.13/6.88 → `m_group_size=3`: 26.97/6.26 → **`m_group_size=5`: 4.93/1.60 SE, matching the exact dense correction's 5.26/1.73 SE** — with no divergence throughout (`max|alm|` ~13.8 across all variants, same order as the bounded diagonal chain). This resolves the divergence-vs-bias trade-off at a fraction of the exact update's cost: block Cholesky factorisation is `O(Σ_blocks size³)` rather than `O(n_alm³)` (e.g. `m_group_size=5` blocks are `~5×2×lmax` wide near `m=0` and shrink with `m`, vs one `n_alm×n_alm` factorisation).
  - **Not yet done:** benchmarking `use_block_correction=True` at production lmax=300 (block-Cholesky cost per outer Gibbs sweep — it must be rebuilt every sweep since it depends on the current `C_l` draw, unlike the one-time diagonal calibration — vs the 6.7s/PCG-iteration CG baseline), tuning `m_group_size`/`n_messenger_iter`/any `τ²`-annealing schedule at that scale, and confirming the `m_group_size=5` sweet spot found at lmax=16 still holds at lmax=300 (larger blocks may behave differently as `lmax/m_group_size` grows). This is now squarely Step 6's job.
- [~] **Step 6 — production validation chain** at lmax=300, real Planck data ▶ (started 2026-07-04). Benchmark `use_block_correction=True` cost/`m_group_size` per the note above, then verify ESS ≈ N, C_l agreement with the Phase 0 HMC posterior, and no flat-residual-equivalent pathology.
  - **Wiring (2026-07-04):** `run_gibbs_chain`/`run_sampler.py` did not actually expose `use_block_correction`/`m_group_size` — `alm_sampler='messenger'` silently fell back to the plain diagonal approximation Step 5 showed is biased on a masked sky (and `run_sampler.py` had no `'messenger'` choice at all, nor a `--use_matrixfree_sht` flag it requires). Added `messenger_use_block_correction`/`messenger_m_group_size` params threading through `run_gibbs_chain` into `sample_alm_messenger`, and `--alm_sampler messenger`, `--n_messenger_iter`, `--messenger_use_block_correction`, `--messenger_m_group_size`, `--use_matrixfree_sht` CLI flags on `run_sampler.py` (auto-enables `use_matrixfree_sht` when `alm_sampler='messenger'`). `tests/test_samplers.py`/`tests/test_messenger.py` (15 tests) pass unchanged.
  - **Isolated-cost benchmark (`scripts/benchmark_messenger_block_lmax300.py`, job 11562977, lmax=300/NSIDE=256, real Planck data, n_alm=89698):** one-time `_calibrate_block_AtA` calibration (independent of `m_group_size`, cached per model) ≈ 25.2 min, extrapolated from 200 timed probes at 16.9 ms/probe. Per-outer-sweep `build_block_cholesky` rebuild cost (isolated, synthetic-but-correctly-sized blocks) vs `m_group_size`: `1`→0.48s, `3`→1.48s, `5`→2.73s, `10`→5.92s, `20`→12.5s (block sizes up to 596/1776/2930/5710/11018 respectively) — `m_group_size≤10` all cheaper than the 6.7s/PCG-iteration CG reference point on this cost component alone.
  - **Caveat found while writing this up:** the isolated benchmark above only timed the Cholesky *rebuild*, not the full `sample_alm_messenger` call — which also pays `n_messenger_iter` (100 by default) forward/adjoint full-sky SHTs per outer sweep (~17ms each from the same probe timing, i.e. ~3.4s/sweep on its own) plus a per-block triangular solve every inner iteration (not just at rebuild time), so the true per-sweep cost is higher than 2.73s at `m_group_size=5`. `scripts/benchmark_messenger_fullcall_lmax300.py` (job 11562981, started 2026-07-04) times the actual end-to-end call for `m_group_size ∈ {1,3,5}` to get the real number before picking a production setting.
  - **Not yet done:** reading back job 11562981's results and picking `m_group_size`; then a real production Gibbs chain (`run_sampler.py --alm_sampler messenger --messenger_use_block_correction --messenger_m_group_size <chosen>`) at lmax=300 to verify ESS ≈ N, C_l agreement with Phase 0 HMC, no pathology.

**Key reference:** Elsner & Wandelt 2013 (arXiv:1210.4931), "A novel approach to Gaussian constrained sampling with messenger fields."

**Output:** an alm|C_l sampler that actually converges on the real masked-sky problem, unblocking Phase 0b and the rest of the critical path. Step 5's stability-vs-bias trade-off is now resolved at validation scale (block-diagonal-by-`m` correction, `m_group_size=5`, matches the exact dense fix's accuracy with no divergence); **Step 6 (production-scale lmax=300 validation) is the remaining gate** before this sampler can replace CG in production chains.

---

### Phase 1 — Differentiable lensing operator ✓ (validated at small lmax; performance gated by Phase 1.5)

**Goal:** implement a differentiable lensing operator so that HMC can compute gradients through the lensed likelihood with respect to both `alm` and `phi`.

- [x] Survey existing differentiable lensing implementations: `lenspyx` (Carron), `lensit`, JAX-based pixell — chose custom TF reimplementation for full autodiff control and TF stack compatibility
- [x] Implement `lens_map_tf(alm, phi_alm) -> lensed_map` — `diffcmb/lensing.py`; also `lens_map_phi_diff_tf` for joint differentiability w.r.t. both alm and phi
- [x] Validate `dL/d_alm` and `dL/d_phi_alm` against finite differences at lmax=50 — all four gradient tests pass (required factor-of-2 fix for m>0 in adjoint, Npix/(4π) normalisation)
- [x] `psi_lensed` as drop-in replacement for `_psi_tf_raw`
- [x] Benchmark forward + backward at lmax=300, NSIDE=256 — `scripts/benchmark_lensing.py` (job 11552544, 2026-07-01, 2x A30): forward=4.55s, forward+backward=9.38s. **Bottleneck: only 2 of 53 dense sph matrix parts fit in 22GB GPU memory; 51 fall back to CPU. Implied Phase 2 HMC rate ≈0.11 leapfrog steps/s — a 20-step trajectory ~3 min.** This measurement is what forces Phase 1.5.

**Key reference:** Carron & Lewis 2017 (arXiv:1701.01712); lenspyx library.

---

### Phase 1.5 — Kill the dense SHT matrix (hard gate for Phase 2) ★ NEW

**Goal:** make the lensed forward+backward pass fast enough that Phase 2 HMC at lmax=300 is practical on available hardware, and that lmax ≥ 1000 (Phase 4) stops being structurally impossible.

**The problem, quantified:** the dense `sph` matrix is O(lmax² × Npix). At lmax=300, NSIDE=256, float64 that is ~570 GB split into 53 parts, of which 2 fit on a 22GB A30 — so 96% of every matvec runs on CPU. This is not a "get more GPUs" problem: at lmax=1000 the dense matrix is ~60x larger. The dense representation must go. This was previously buried in Phase 4; it is in fact a **prerequisite for Phase 2 production runs** and is promoted accordingly.

**Primary plan — matrix-free SHT (do this):**
- [x] Wrap `ducc0.sht` synthesis (alm → map) and adjoint synthesis (map → alm) in a `tf.custom_gradient` op — `diffcmb/diffcmb/sht_ducc.py` (`HealpixSHT`, `masked_synthesis_tf`), 2026-07-02. Forward = `ducc0.sht.synthesis` (verified to match `healpy.alm2map` to 1e-13), backward = `ducc0.sht.adjoint_synthesis` via the derived Wirtinger-gradient identity `sum(Synthesis(a)·m) == sum_lm w_lm·Re(conj(a_lm)·AdjointSynthesis(m)_lm)`, `w_lm` = 2 for m>0 / 1 for m=0 (same convention as the existing `alm_weights`). No stored matrix; geometry comes from `ducc0.healpix.Healpix_Base.sht_info()`, which needed no manual ring-geometry derivation.
- [x] Validate the wrapped op: `tests/test_sht_ducc.py` — forward vs `healpy.alm2map` (1e-13), adjoint identity (1e-16), and gradient vs finite differences (1e-4, cheap lmax=15 CPU-only check). All pass.
- [x] Benchmark at production scale (lmax=300, NSIDE=256, real Planck mask) — `scripts/benchmark_sht_ducc.py`, run on the login node (no GPU/dine2 needed): **forward+backward = 0.018s, ~500x under the <1s gate** (vs 9.38s dense GPU+CPU). Gate passed.
- [x] Wired `masked_synthesis_tf` into `model.py` (`_psi_tf_raw`, opt-in `use_matrixfree_sht=True` / `sht_nthreads` constructor args, dense path stays default) and `samplers.py` (`sample_alm_cg`'s `_cg_jt_v_fn`, `Ninv` lookup) — 2026-07-02. Dense path untouched/still default, per the "keep it behind a flag" plan below.
  - Caught two real bugs during wiring, both now covered by regression tests (`tests/test_sht_ducc_model_integration.py`): (1) **alm ordering mismatch** — `splittosingularalm_tf`'s output is this codebase's "author ordering" (row-major by (L,m), matching the dense `sph` matrix's column order), but ducc0/healpy expect "healpy ordering" (column-major by m); fixed with a precomputed `tf.gather` index (`model._alm_mo_to_ho_idx`, built from `alm_utils._ordering_indices`). (2) **gradient sign convention** — `masked_synthesis_tf`'s backward must return `w*g` unconjugated (not `conj(w*g)`, despite `matvec_on_device` using `conj(...)`), because this op fuses the real-part extraction internally rather than relying on a downstream `tf.math.real()` to set the sign convention; caught by an exact real-part match / flipped-sign imaginary-part mismatch against the dense path.
  - Also fixed: the op used bare `.numpy()` calls, which break when embedded in a `tf.function`-traced graph (as `_psi_tf_raw` always is, via `psi_tf`'s compiled wrapper and `samplers.py`'s `@tf.function`-decorated `_grad_fn`/`_jt_v_fn`) — switched to `tf.py_function`, verified by `test_psi_tf_raw_traceable_with_matrixfree_sht`.
- [x] Regression vs dense path: `tests/test_sht_ducc_model_integration.py` — `psi_tf` value matches dense to 1e-8 relative, alm gradient matches dense to 1e-6 (rtol/atol), both at lmax=12 synthetic data with identical underlying sky/mask/noise shared between the two model instances.
- [x] Re-run `tests/test_cg_matvec.py`'s symmetry/linearity/PD checks with `use_matrixfree_sht=True` specifically — added `test_cg_matvec_linear_symmetric_matrixfree_sht` (2026-07-03), same real-data lmax=30 A-operator checks as the dense-path test plus an explicit PD assertion (`dot(p, Ap) > 0`) added to both. Passes: linear to 1e-3, symmetric to 1e-6, PD confirmed.
- [ ] Switch production Phase 2 runs over to `use_matrixfree_sht=True` once the above is done — dense path stays the default/fallback for now
- Reference implementations to crib from: `s2fft` (JAX differentiable SHT — the same custom-vjp pattern), `lenspyx`/`delensalot` (ducc-backed curved-sky lensing)

**Secondary options (only if the primary path underdelivers):**
- Mixed precision: fp32 storage / fp64 accumulation for the matvec. Permitted *only* with fp64 accumulation and *only* if validated against fp64 chains — the Phase 0 float32 false-convergence result is the standing warning
- Gradient checkpointing through the lensing op to trade compute for memory
- COSMA Grace Hopper node (large unified memory) as a brute-force fallback for one-off validation runs — not a scaling strategy

**Also in this phase (removes the second Phase 2 slowdown):**
- [x] Make `lens_map_phi_diff_tf` graph-traceable — done 2026-07-03. Both escape hatches into numpy/healpy (the bilinear-geometry precompute, previously a bare `phi_packed_tf.numpy()` call outside any `tf.custom_gradient`, and the FD backward pass, previously calling `.numpy()` on the upstream/T_map tensors inside the `custom_gradient` closure) now go through `tf.py_function`, mirroring `sht_ducc.py`'s `masked_synthesis_tf`. No change to the interpolation math itself — same FD formulas, same `eps_angle=1e-7`, same `_bilinear_weight_grads`/`_deflection_adjoint` calls. New regression test `tests/test_lensing.py::test_lens_map_phi_diff_tf_traceable_in_tf_function` checks a `tf.function`-traced call reproduces the eager-mode forward value and both gradients (w.r.t. `T_map` and `phi`) exactly. Full test suite re-run clean apart from the two pre-existing phi-gradient-accuracy failures below (unaffected by this change, same failure mode before and after).

**Output:** lensed forward+backward at lmax=300 in ≲1s, no dense matrix, Phase 2 unblocked and Phase 4 de-risked. **Do not submit Phase 2 production chains until the gate benchmark passes.**

---

### Phase 2 — Three-block Gibbs sampler over (alm, C_l, phi) — the methods paper

**Goal:** extend the Gibbs sampler to jointly sample the unlensed CMB signal, its power spectrum, and the lensing potential, **on the full curved sky**. This is the core novel contribution — scoped as such.

```
Block 1:  C_l      | alm, phi, d  — exact inverse-Gamma draw (implemented, unchanged)
Block 2:  alm      | C_l, phi, d  — HMC with lensed likelihood (Phase 1); CG mass matrix from Phase 0b as preconditioner
Block 3:  phi      | alm, C_l, d  — HMC targeting log p(d | alm, phi) + log p(phi | C_l^phiphi)
```

**Note on Block 2 in the lensed setting — CORRECTED 2026-07-04 (the original claim here was
mathematically wrong):** this note used to say "`p(alm | C_l, phi, d)` is no longer Gaussian,
so HMC returns." That is not true. For **fixed** `phi`, lensing is a *linear* operator on the
unlensed field — the query points and hence the bilinear interpolation weights are fixed, so
`d = W_phi · Y · alm + n` with `W_phi` a constant matrix — which makes
`p(alm | C_l, phi, d)` **exactly Gaussian** with precision
`C_l^{-1} + (W_phi Y)^T N^{-1} (W_phi Y)`. Only the *joint* `(alm, phi)` posterior is
non-Gaussian; only Block 3 (`phi | alm`) genuinely requires HMC. Three consequences:
(1) **HMC for Block 2 is a cost/engineering choice, not a necessity** — the honest reason is
that the exact draw's operator changes with `phi` every sweep (so the Phase 0c messenger
`AᵀA` calibration, currently cached once per model, would need to be `phi`-dependent), not
non-Gaussianity; the paper text must say this or a referee who knows the algebra will.
(2) An **exact messenger/CG Block-2 draw in the lensed case may be viable**: weak lensing is
near-norm-preserving (`W_phi^T W_phi ≈ I` up to O(magnification)), so the block-diagonal-by-m
`AᵀA` structure plausibly survives as a good approximation with the *unlensed* calibration —
worth one cheap small-lmax check before defaulting to HMC, since an exact draw beats HMC on
both correctness bookkeeping and per-sweep cost. (3) **Free per-block validation either way**:
at small lmax the exact Gaussian conditional is computable densely, so Block-2 HMC (or the
lensed messenger draw) can be validated against an exact reference — the same
dense-reference discipline that caught three real bugs in Phase 0c Steps 1–2. The Phase 0b
preconditioner `P = C_l^{-1} + (1/σ²) diag(Y^T Y)` remains the correct HMC mass matrix if
HMC is retained — it captures the Gaussian curvature of the unlensed problem, with lensing a
manageable perturbation.

- [x] Add `phi_alm` as a parameter block — `run_gibbs_chain(..., cl_phiphi_full=...)`, 2026-07-01
- [x] Implement `log_prob_phi_block(phi_alm, alm, Cl_phi)` using the Phase 1 lensed likelihood
- [x] Wire Block 3 HMC into `run_gibbs_chain` (opt-in via `cl_phiphi_full`; Block 2 switches to `psi_lensed` when active). Smoke-tested at lmax=10 (`tests/test_samplers.py::test_gibbs_chain_with_phi_block_moves`); currently eager-mode (fixed by Phase 1.5 TF-native interpolation); not yet run at production lmax or validated statistically
- [x] **Phi-block gradient bug — fixed 2026-07-03.** Root cause was two-fold, both now resolved in `lensing.py`:
  - **The backward pass's own FD was replaced with an analytic formula.** `hp.get_interp_weights`' bilinear scheme is, for any single query, two nested linear interpolations (`v` in theta between two rings, `u1`/`u2` in phi within each ring) — verified to reproduce healpy's own weights to ~1e-14 away from the poles. Differentiating this closed form analytically (`_analytic_bilinear_weight_grads`), using only the single (neighbors, weights) already returned for the exact query point, never re-queries at a shifted angle, so it can't cross into a different interpolation cell (the mechanism behind the old bug: re-querying at theta'±eps could land in a different discrete neighbor set, producing wild step-size-dependent values that the spin-1 SHT adjoint then smeared across most phi_alm modes). One genuine exception remains: a thin polar annulus (~1.5% of sky per pole at NSIDE=16, scaling as ~1/(4·NSIDE)) where HEALPix's own scheme collapses the two rings into one and isn't bilinear; empirically confirmed FD *is* stable there (checked across eps spanning 1e-9–1e-4), so that annulus still uses a small-eps FD fallback, now correctly scoped to only those pixels instead of applied everywhere.
  - **The regression tests' own FD ground truth was also unstable at its original eps=1e-6.** A single phi_alm component perturbs every pixel's lensed position at once; since `hp.get_interp_weights` is C0 but not C1, eps=1e-6 was large enough that a handful of the ~600 unmasked pixels crossed a genuine interpolation-cell boundary within the perturbation — making *the FD reference itself* unstable (verified: swings by >100%, occasional sign flips, between eps=1e-6 and eps=1e-7 for affected components), not the analytic gradient. Cross-validated the fixed implementation against an independent, FD-free "gold standard" (exact linear deflection-field Jacobian via unit-impulse alm × analytic weight-gradient) — matches to ≤3.7% even including the polar-annulus fallback pixels, and to ~1e-6 relative or better for the majority. Reduced test eps to 1e-9 (safe: deflection_field is exactly linear in phi_alm, so no truncation/roundoff tradeoff forces eps larger; checked stable down to 3e-10). Both previously-failing tests (`test_phi_grad_deflection_adjoint_vs_fd`, `test_psi_lensed_phi_grad_vs_fd`) now pass; full suite (47 tests) green.
  - Block 3 HMC gradients can now be trusted for production inference (previously flagged as untrustworthy pending this fix).
- [ ] Optionally add Block 4: `C_l^phiphi | phi` — exact inverse-Gamma, same structure as Block 1
- [ ] **Simulation validation (the core result):** lensed full-sky simulations with known `phi_true` at lmax=300 — verify unbiased phi recovery, C_l^TT recovery, and coverage of the joint posterior (rank/coverage tests, not just point agreement)
- [ ] **Benchmark against CMBLensing.jl (mandatory):** on a matched flat-patch simulation within both codes' domains, compare phi posterior mean/uncertainty against its joint HMC sampler and MUSE; on the full sky, compare against the quadratic estimator (the only competitor that operates there). Referees will demand this comparison — do it before they ask
- [ ] Quantify bias reduction in recovered C_l^TT relative to a lensing-blind Commander-style analysis of the same lensed sims
- [ ] Quantify what joint sampling buys over marginal methods: per-mode uncertainty propagation between C_l^TT, alm, and phi that MUSE's Gaussianised marginal cannot provide (this is the differentiator — make it a figure, not a sentence)

**This is the methods paper.** The publishable claim, scoped to survive review:

> *We present the first fully joint Bayesian sampler over the unlensed CMB, its angular power spectrum, and the gravitational lensing potential **on the full curved sky**. Joint field-level lensing inference has previously been demonstrated only on flat-sky patches (Millea, Anderes & Wandelt 2020; CMBLensing.jl), while existing full-sky methods return point estimates (iterative MAP) or approximate marginals (MUSE, quadratic estimators). Our curved-sky Gibbs sampler returns full posterior samples over all unknowns, correctly propagating uncertainty between C_l^TT, alm, and phi, and reduces power-spectrum bias relative to lensing-blind Gibbs analyses. This provides the sampling infrastructure required for full-sky delensing with LiteBIRD-class experiments.*

If curved-sky MUSE appears before submission: the claim narrows to "first curved-sky joint *sampler*" (MUSE is not a sampler and returns no joint posterior) — still publishable, but move fast.

---

### Phase 2b — Cosmological parameter inference (PARKED)

`emcee` over the sampled C_l posterior → ΛCDM parameters, Blackwell-Rao, corner plots vs Planck 2018 (arXiv:1907.12875). Scientifically routine — it validates rather than extends, and every GPU-hour and working week it consumes comes off the Phase 1.5→2 critical path where the actual novelty lives. **Do not start until the Phase 2 methods paper is submitted**, then it becomes a cheap robustness section or short companion analysis.

---

### Phase 3 — Polarization and full-sky delensing — the science paper

**Goal:** extend to full temperature + polarization (TQU) analysis and target the science case that motivates full-sky sampling in the first place.

This is where the curved-sky positioning pays off: B-mode delensing for primordial gravitational waves is the primary science driver of LiteBIRD (full-sky by construction — flat-sky codes structurally cannot serve it; its current lensing pipeline is QE/iterative, arXiv:2507.22618) and of SO/CMB-S4 large-area surveys. A joint `(T, E, B, phi)` full-sky sampler enables sampling-based delensing with propagated uncertainties — a capability no existing code offers on the curved sky.

- [ ] Extend `alm_utils.py` to spin-2 fields: Q, U maps <-> E, B alms (matrix-free via ducc0 spin-2 transforms — Phase 1.5 infrastructure carries over directly)
- [ ] Extend `psi_tf` to the TQU joint likelihood with (TT, TE, EE, BB) power spectrum block
- [ ] Handle C_l^TE in the C_l Gibbs block: off-diagonal term breaks inverse-Gamma conjugacy — 2x2 inverse-Wishart draw (BeyondPlanck structure) or HMC
- [ ] Extend the lensing operator to spin-2 (lensing mixes E and B)
- [ ] Test on simulated lensed TQU maps at LiteBIRD-like noise: measure delensing efficiency (B-mode power removal vs QE/iterative baselines) and the recovered r constraint

**Key references:** BeyondPlanck (arXiv:2303.04819) for the polarization Gibbs blocks; LiteBIRD lensing forecast (arXiv:2507.22618) for the target experiment configuration.

---

### Phase 4 — Scalability to lmax ≥ 1000

With Phase 1.5 done (matrix-free SHT), the structural barrier is gone and this phase becomes tuning rather than rearchitecture:

- [ ] Profile per-iteration breakdown at lmax=1000, NSIDE=1024: SHT vs lensing interpolation vs C_l draw vs Python overhead
- [ ] Benchmark ESS/hour vs lmax
- [ ] Multi-GPU / multi-node distribution of chains (embarrassingly parallel) and, if needed, of the SHT itself
- [ ] Target: 1000 samples at lmax=1000 within 72h on COSMA8 dine2 nodes

---

### Phase 5 — Non-Gaussian extensions (longer term, PARKED)

Once Phases 1.5–3 are in place, the non-conjugate structure of `psi_tf` becomes a platform for extensions Commander fundamentally cannot support:

- **fNL sampling:** bispectrum likelihood term in `psi_tf`; jointly sample `(alm, C_l, fNL)` at map level
- **Galactic mask in-painting:** constrained realisations of unobserved pixels as an additional Gibbs block
- **Learned CMB prior:** diffusion model / normalising flow prior score as an additional `psi_tf` term (cf. arXiv:2405.05598 for diffusion-based phi reconstruction — a potential Block 3 alternative worth watching)
- **Instrument systematics:** calibration factors, beam errors, 1/f noise amplitudes as additional HMC blocks — same pattern as the phi block

These are listed to record the platform argument, not to be worked on. Each is a separate paper *after* Phases 2–3.

---

## Proposed additions — external strategy review 2026-07-04 (NOT YET DONE, none started)

Recorded from a portfolio-wide gap-analysis pass; none of these are commitments until the
critical path (Phase 0c Step 6) lands, but three of them are cheap and de-risk the two
things referees will actually attack.

1. **The joint (alm, phi) mixing problem — the single biggest un-planned research risk in
   Phase 2.** The roadmap plans naive Gibbs alternation between Block 2 (`alm | phi`) and
   Block 3 (`phi | alm`). But the flat-sky experience this project positions itself against
   is explicit that this is the hard part: Millea, Anderes & Wandelt 2020's central
   algorithmic contribution was not the per-block samplers but a **reparametrisation**
   (interpolating between lensed and unlensed parametrisations) introduced precisely because
   `f` and `phi` are so correlated in the joint posterior that naive block alternation mixes
   catastrophically slowly at high S/N. Nothing in Phase 2's checklist mentions
   block-correlation, joint IAT, or reparametrisation — with perfect per-block samplers the
   outer chain could still have autocorrelation times in the thousands, which is a
   show-stopper discovered only *after* production compute is spent. **Cheap early gate
   (do before any lmax=300 Phase 2 chain):** on a lensed simulation at lmax≤50, measure the
   joint (alm, phi) IAT of the alternating chain directly. If it is pathological, the known
   fixes are the Millea-style mixed parametrisation (curved-sky version = new methods
   content, arguably a *stronger* paper) or joint (alm, phi) HMC updates. Either way, knowing
   at lmax=50 costs hours; knowing at lmax=300 costs weeks.

2. **Forward-model realism gaps that block any real-data claim: beam, pixel window,
   anisotropic noise.** Verified against the code 2026-07-04 (grep, not memory): there is
   **no beam transfer function `B_l` and no HEALPix pixel window** anywhere in the forward
   model (`alm_utils.py`'s `hp.smoothalm(fwhm=0.0)` is a no-op), and the noise model is
   uniform white (`model.py`: `full_Ninv = 1/σ²`, zeroed under the mask) with no hit-count
   anisotropy. For simulation-only Phase 2 validation this is self-consistent and fine. It is
   **not** fine for the two real-data surfaces this project already touches: Phase 0's
   validation *fits real Planck data* (a 5′-class effective beam is a ~3–4% C_l suppression
   at l=300, NSIDE=256 pixel window another ~1–2% — both currently absorbed silently into the
   recovered C_l), and the A_L-anomaly interrogation PAPERS.md advertises as this project's
   tension hook is a real-Planck claim that would be referee-rejected without them. The fixes
   are cheap and localised: `B_l·p_l` is one diagonal multiply in harmonic space (works
   identically in the dense and matrix-free SHT paths), and per-pixel diagonal `N_ii` is
   natively supported by the messenger formalism (`τ² = min N_ii` over observed pixels) — but
   they must be *in the model* before any real-data figure is quoted. Add as an explicit
   pre-condition on the A_L work, and note the A_L hook itself in this roadmap (it currently
   exists only in PAPERS.md, so a reader of this file alone doesn't know Phase 2 has a
   real-data science headline waiting).

3. **Named fallback if pure messenger sweeps are too slow at lmax=300:** the
   messenger-as-preconditioner-for-CG hybrid (Huffenberger & Næss 2018) — known to converge
   faster than either pure method on masked-sky systems; it reuses everything built in
   Phase 0c (the messenger operator becomes the preconditioner, CG supplies the convergence
   guarantee the diagonal preconditioner couldn't). Also record the τ²-cooling-schedule
   option (Elsner & Wandelt's own λτ² annealing) as the first tuning knob. And one
   correctness note worth writing down so nobody "fixes" it wrongly later: alternating
   `t | s` / `s | t` sub-iterations is exact data augmentation — the augmented `(s, t)`
   chain targets the exact joint, so a finite `n_messenger_iter` per outer sweep affects
   *mixing only, not correctness*; tune `n_messenger_iter` by measuring augmented-chain IAT
   (e.g. 10 vs 100 sub-iterations may buy nearly identical mixing per wall-clock second),
   don't treat 100 as load-bearing.

4. **Block-2 exactness opportunity under lensing** — see the corrected "Note on Block 2"
   under Phase 2: the lensed `alm` conditional is still exactly Gaussian at fixed `phi`, so
   (a) the paper must not claim HMC is *required* for Block 2, and (b) one cheap small-lmax
   experiment (lensed messenger draw with the unlensed `AᵀA` calibration, checked against a
   dense exact reference) decides whether Phase 2 can keep exact draws for both Gaussian
   blocks and confine HMC to Block 3 only — a cleaner, faster, and more defensible sampler
   design if it works.

5. **The mandatory CMBLensing.jl benchmark implicitly assumes the sampler works at
   patch-scale f_sky — untested (added 2026-07-04, second pass).** The Phase 2 plan promises
   a comparison "on a matched flat-patch simulation within both codes' domains" — for
   CMBLensing.jl that means a ~650 deg² patch, i.e. **f_sky ≈ 0.016** on the HEALPix side.
   Every messenger-sampler validation to date ran at f_sky ≈ 0.69–0.77; the masked-sky
   pathology this whole Phase 0c campaign fought (off-diagonal `AᵀA` coupling from the mask)
   *grows* as the mask deepens, and nobody knows whether the block-diagonal-by-m correction
   (or the m_group_size=5 sweet spot) survives a 98%-masked sky. Two-hour check at small
   lmax (`debug_messenger_masksky.py` with a patch-scale mask) *before* the benchmark is
   promised in any paper text; if it fails there, the honest benchmark design inverts —
   compare full-sky DiffCMB against the quadratic estimator (its natural domain) as primary,
   with the flat-patch CMBLensing.jl comparison scoped to whatever f_sky the sampler
   demonstrably handles.

---

## Standing discipline

- **One critical path:** 0b (verify) → 1.5 (build+gate) → 2 (validate+write). Anything not on it waits.
- **No Phase 2 production submissions until the Phase 1.5 gate benchmark passes** (~1s lensed forward+backward at lmax=300).
- **Precision rule:** fp64 end-to-end unless a mixed scheme with fp64 accumulation is explicitly validated against fp64 chains (Phase 0 float32 false convergence is the standing counterexample).
- **Claims hygiene:** every "first" in a draft carries a scope qualifier ("full-sky", "curved-sky", "sampler") and a citation to the nearest prior work (Millea, Anderes & Wandelt 2020; MUSE; delensalot). Re-check the arXiv for curved-sky MUSE/field-level papers before each submission milestone.
- **Opt-in flags need an actually-active test.** The Step 6 wiring bug (2026-07-04: `use_block_correction`/`m_group_size` accepted but silently ignored by `run_gibbs_chain`, so `alm_sampler='messenger'` quietly ran the known-biased diagonal path) is the second silently-defaulted-flag bug in this portfolio in a week (cf. galform_imf's silent stub-import fallback). Any new opt-in code path gets a regression test asserting the non-default branch is actually exercised (sentinel, log line, or a result that *differs* from the default path) — "runs without error" does not count.
