# Research Roadmap: Differentiable Bayesian CMB Analysis

*Last substantive revision: 2026-07-02 (novelty repositioning vs CMBLensing.jl; Phase 2 hardware feasibility plan; LiteBIRD science target).*

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
- **⚠ Chain 1 flat-residual anomaly (2026-07-02) — operator cleared, cause still open:** chain 1's log (`logs/cmb_cg_L300_11552531_1.out`) shows the PCG residual flat at ~5.34e4 across all 228 logged solves since iteration 0, the same magnitude as the pre-fix `11513133` symptom. Ran `scripts/debug_cg.py` under the exact production device layout (`CUDA_VISIBLE_DEVICES=0`, real data, lmax=300, 41 parts, 1 GPU + 40 CPU — job `11555267`, with a job-private `$TMPDIR` to rule out node-local autograph cache staleness, see below): `A` is symmetric (1.9e-12), positive-definite, linear (3.6e-13), and **the PCG residual decreases properly from 2.49e7 to 2.02e5 over 10 iterations** — the matvec operator itself is sound at production scale, so the 66f169c fix does generalize to the GPU+CPU layout. The diagnostic ran without a checkpoint (fresh prior-cls `lncl`), so it doesn't 1:1 replicate chain 1's exact resumed state — the flat residual is therefore *not* explained by a broken operator, and is more likely either (a) that specific run's autograph trace hitting the same node-local `$TMPDIR` staleness bug found below (its trace happened before this was understood), or (b) a preconditioner/conditioning issue specific to the checkpoint's converged `lncl`. **Recommendation: kill and resubmit job 11552530 with the `$TMPDIR` isolation fix (see below) and confirm the residual decreases from a fresh trace before trusting any chain-1 CG samples.** Not done yet — resubmitting a multi-day production job is a call for the user, not made unilaterally here.
- **Found + fixed: node-local autograph cache collision (2026-07-02).** Two earlier attempts at this same diagnostic (jobs `11555241`, `11555248`) crashed with `TypeError: tf___ensure_tf_tensors() takes 1 positional argument but 2 were given` — a nonsensical error thrown from inside an autograph-traced call to an unrelated method. Every smaller-scale repro (lmax=10/30, real data, same GPU node) ran cleanly, isolating the trigger to lmax=300's autograph trace specifically on a node (`gc004`) that had hosted earlier attempts. Autograph writes transpiled sources to `$TMPDIR/__autograph_generated_file*.py`, which is node-local and not job-scoped by default; setting a job-private `$TMPDIR` (`scripts/debug_cg_single_gpu.slurm`) fixed it on the first retry. **Any script that traces `_psi_tf_raw` fresh (i.e. without a long-lived cached `_cg_grad_fn`) should set a job-private `$TMPDIR` on dine2/cosma8 nodes that may be reused across jobs** — this plausibly also explains the two "phantom" crashes above, unrelated to the operator-correctness question they were meant to test.
- [ ] On completion, verify: ESS ≈ N at all multipoles including l=200–300; logp plateau; C_l agreement with Phase 0 at l ≤ 100; re-plot Planck comparison
- [ ] Benchmark wallclock per sample vs HMC at lmax=300 (expect 2–5x faster)

**Key reference:** Wandelt, Larson & Lakshminarayanan 2004 (arXiv:astro-ph/0310080); Jewell, Levin & Anderson 2004.

**Output:** a validated, efficiently mixing Gibbs sampler at lmax=300 with ESS ≈ N for all multipoles. This is the correct baseline before adding lensing.

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

**Note on Block 2 in the lensed setting:** `p(alm | C_l, phi, d)` is no longer Gaussian, so HMC returns. The Phase 0b preconditioner `P = C_l^{-1} + (1/σ²) diag(Y^T Y)` remains the correct HMC mass matrix — it captures the Gaussian curvature of the unlensed problem, with lensing a manageable perturbation.

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

## Standing discipline

- **One critical path:** 0b (verify) → 1.5 (build+gate) → 2 (validate+write). Anything not on it waits.
- **No Phase 2 production submissions until the Phase 1.5 gate benchmark passes** (~1s lensed forward+backward at lmax=300).
- **Precision rule:** fp64 end-to-end unless a mixed scheme with fp64 accumulation is explicitly validated against fp64 chains (Phase 0 float32 false convergence is the standing counterexample).
- **Claims hygiene:** every "first" in a draft carries a scope qualifier ("full-sky", "curved-sky", "sampler") and a citation to the nearest prior work (Millea, Anderes & Wandelt 2020; MUSE; delensalot). Re-check the arXiv for curved-sky MUSE/field-level papers before each submission milestone.
