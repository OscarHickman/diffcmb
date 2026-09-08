# CMBLensing.jl benchmark design — research notes (2026-07-29)

*Reviewed 2026-09-06: the CMBLensing.jl facts below are unchanged. One diffcmb-side
detail referenced here has moved twice — the exact inverse-Gamma C_l block is still exact,
but its shape parameter was corrected on 2026-08-31 (it had assumed 2l+1 packed dof where
the vector then carried 2l), and on 2026-09-06 the missing `Im(a_{l,1})` dof was restored,
so the packing genuinely carries 2l+1 and the shape is `l - 0.5` again (see
`achievements.md`). The shape is derived from the packing, not hardcoded. Nothing in the
benchmark design depends on either move; the cost figures quoted below are per-sweep and
unaffected by the 62 extra coordinates at lmax=64.*

## 1. CMBLensing.jl technical facts (from Millea, Anderes & Wandelt 2020, arXiv:2002.00965, and repo)

### Field representation / sampler
- Flat-sky Fourier-space fields (`FlatMap`/`FlatFourier` types), lensing applied via LenseFlow
  (ODE-based lensing operator, exact gradients/no Taylor truncation).
- Joint posterior is P(f, phi, r, A_phi | d) — sampled directly (not the analytically-marginalized
  "marginal posterior" which needs an intractable det(Sigma_d)).
- Reparametrization to mixed variables (f', phi') via D(r), G(A_phi) (Eqns 13-16) is *load-bearing*:
  without it, sampling is reported as basically impossible (phi-only mixing chain autocorrelation
  ~25x worse without the G(A_phi) mixing matrix; the paper explicitly says without D(r) mixing "it
  would be impossible to even run a chain").
- Gibbs block structure (Algorithm 1):
  1. f' | phi', A_phi, r, d — **conjugate gradient** draw (Gaussian conditional, messenger-free CG,
     diagonal Jacobi preconditioner, Eqn 30).
  2. phi' | A_phi, r, f', d — **single HMC pass** (not NUTS): leapfrog, n_h=25 steps, eps_h=0.02,
     mass matrix = Fisher/Hessian approximation A_phi'(A_phi) = G(A_phi)^-2 [N_phi^-1 + C_phi(A_phi)^-1]
     (Eqn 26). Tuned per-configuration; HMC acceptance tuned to ~80%.
  3. A_phi | ... and r | ... — **slice sampling** (1D grid, 200 points), not HMC — split off because
     they're "global" params.
  - Initialization: 20 iterations of a cheap quasi-Newton "quasi-sample" warm-start (not full HMC)
    before the real chain starts, specifically to avoid mean-field bias in burn-in.
- This is architecturally close to diffcmb's setup (Gibbs over blocks with one non-Gaussian block on
  HMC) but CMBLensing uses CG for the Gaussian f-block (which diffcmb closed as "biased when phi
  block active" — see achievements.md) vs diffcmb's exact inverse-Gamma C_l block + full HMC on both
  alm and phi. Worth flagging directly: **diffcmb's own CG-shortcut route was tried and rejected
  (652% bias) — CMBLensing.jl's CG works because their f-block conditional is genuinely Gaussian
  given fixed phi in their formulation and phi is drawn in a separate HMC pass, i.e. same
  architecture we use, not the shortcut we rejected.** Confirm this distinction explicitly in any
  paper text — superficially "they use CG, we tried CG and it failed" reads as a contradiction that
  isn't actually one.

### MUSE
- MUSE (Millea & Seljak 2021, arXiv:2112.09354) is implemented as a separate package, MuseInference.jl,
  which CMBLensing.jl integrates with for "MUSE inference of bandpowers of phi and unlensed f."
  It is a marginal score-expansion estimator, not a sampler — point/quasi-Bayesian estimate + Fisher-
  approximated uncertainty, no joint posterior. (Consistent with achievements.md's positioning:
  "MUSE is marginal, not a sampler.")
- MUSE explicitly claims curved-sky HMC is "out of reach" in its own paper — this is diffcmb's
  stated novelty gap to exploit, not something to benchmark head-to-head on sampling quality (MUSE
  doesn't produce full posteriors to compare against ESS-wise; it's a different benchmark axis
  — accuracy/efficiency of point estimate + Fisher errors vs full joint posterior marginals).

### Documented example configurations — Table II of the paper (THE key matching target)
Three configs, all **polarization-only (Q/U)**, flat-sky, CMB-S4-like:

| | 2PARAM | MANY | BIG |
|---|---|---|---|
| Map size | 256×256 | 256×256 | 512×512 |
| Pixel width | 2 arcmin | 3 arcmin | 3 arcmin |
| Total area | 73 deg² | 160 deg² | **650 deg²** |
| Noise (P) | 1 µK-arcmin | 1 µK-arcmin | 1 µK-arcmin |
| (l_knee, alpha_knee) | (100,3) | (100,3) | (100,3) |
| Beam FWHM | 2 arcmin | 3 arcmin | 3 arcmin |
| Fourier mask (l range) | 2<l<5000 | 2<l<3500 | 2<l<3500 |
| Pixel mask border+apod | 0.4°+0.6° | 0.6°+0.9° | 1.2°+1.8° |
| Sampled params | r, A_phi | r | r |
| Wall-time (1 GPU) | 48h | 19h | 50h |

BIG's 650 deg² is described as "around the limit of what is currently computationally possible ...
about a third to a fifth of the planned CMB-S4 deep field." Effective *unmasked* area ~450 deg² after
apodization.

## 2. Patch-area match is already almost exact — a genuinely useful finding

diffcmb's patch-scale gate (job 11626450, `scripts/gate_patch_mixing_check.py`) used a polar-cap
HEALPix mask at **f_sky=0.016**. Converting to solid angle: 0.016 × 41253 deg² (full sky) ≈ **660
deg²** — essentially identical to CMBLensing.jl's BIG configuration (650 deg²), and CMBLensing.jl
itself frames BIG as near the practical ceiling for flat-sky codes. This wasn't by design (the gate
picked f_sky=0.016 for its own reasons) but it means the "matched patch area" leg of the benchmark is
already satisfied without redesigning the sanity-check gate — reuse its output, don't recompute.

**But resolution/lmax do NOT match**, and this is the header caveat for the whole benchmark:
- diffcmb's gate ran at lmax=300, nside=256 (HEALPix pixel ~13.7 arcmin, deliberately oversampled
  for the SHT, not resolution-limited by pixel size).
- CMBLensing.jl's BIG config uses 3 arcmin pixels and a Fourier mask up to l=3500 — over 10x diffcmb's
  probed lmax.
- This mismatch is real and referees will notice it if unaddressed. Two honest ways to frame it:
  (a) **State it as a scope-limited comparison**: same patch area, same cosmology, same rough noise
      regime, deliberately coarser lmax on the diffcmb side (matching Phase 2's currently-validated
      production scale, not artificially inflated for the benchmark) — argue the comparison is about
      matched *methodology and mixing behavior*, not matched *information content*, and say so
      explicitly.
  (b) **Push diffcmb to higher lmax** (~1000-3000) for the benchmark specifically, since the
      matrix-free ducc0 SHT path was built exactly to make lmax>>300 tractable — this is more work
      but produces a genuinely matched comparison. Given per-sweep cost at lmax=300 is already
      29-54s (achievements.md), lmax~1000-3000 chains at the sample counts CMBLensing.jl uses
      (5000-10000 iterations) may be very expensive — needs a cost estimate before committing.
  **Recommend (a) for a first submission-ready benchmark, flag (b) as future work** — matches the
  project's "validation before production" discipline (don't inflate the comparison; make the honest
  version defensible).

- diffcmb's gate used **temperature-only** data (`_noisesig=1.0` default units, TT power spectrum);
  CMBLensing.jl's examples are **polarization-only**. Another explicit caveat to state, not silently
  paper over — TT and EE/BB have different lensing sensitivity and different degeneracy structure
  with A_phi. If diffcmb has a working polarization path (need to check — CLAUDE.md describes T-only
  in the module docs, Phase 3 is "the reason curved-sky matters... full TQU"), a T-only vs P-only
  comparison is defensible as "matched patch/cosmology, different observable" but should say so.

## 3. Proposed benchmark design (concrete)

**Cosmology:** reuse `LCDM_PARAMS = [H0=67.74, ombh2=0.0486, omch2=0.2589, mnu=0.06, omk=0, tau=0.066]`
(already used throughout gate/validation scripts) — CAMB fiducial in both codes should be set to the
closest equivalent (CMBLensing.jl's own defaults, per the paper, are k*=0.002, r*=0.1, A_phi*=1,
omega_b=0.0224567, omega_c=0.118489, tau=0.055, logA=3.043, ns=0.968602, nt=-r*/8 — noticeably
different tau/omega_b/omega_c from diffcmb's defaults; must override CMBLensing.jl's example config
to diffcmb's LCDM_PARAMS for the comparison to be fair, not use its paper defaults verbatim).

**Patch:** reuse f_sky=0.016 polar-cap HEALPix patch (~660 deg²) already validated in job 11626450 —
directly comparable to CMBLensing.jl BIG (650 deg²). Direction of cut: **diffcmb full-sky sim → cut
flat patch → compare to CMBLensing.jl run on an equivalent flat patch of the same simulated sky**, not
the reverse (matches the task brief's instinct — diffcmb is the more general code, CMBLensing.jl only
knows flat patches).

**Noise/beam:** match CMBLensing.jl's BIG config (1 µK-arcmin white noise, 3 arcmin beam FWHM) — this
is also in the CMB-S4-relevant regime the project already targets.

**lmax:** diffcmb at its already-validated lmax=300 (state the mismatch vs CMBLensing.jl's l<3500
explicitly, per above); optionally push to a higher-lmax follow-up.

**What to compare (referee-convincing set):**
1. **C_l^TT (or C_l^EE if polarization) posterior recovery**: mean + credible interval per l-bin vs
   input/truth, both codes overlaid — the standard "does the posterior cover truth" plot.
2. **C_l^phiphi posterior recovery**: same, for the lensing potential power spectrum — this is the
   headline plot since phi is the shared object of interest.
3. **Per-mode (or per-l-bin) ESS and acceptance rate comparison** — table, not just a sentence, per
   the ROADMAP's own "as a figure, not a sentence" standard already stated for the joint-vs-marginal
   comparison item. diffcmb's own gate ESS numbers already exist (job 11626450 corrected numbers in
   achievements.md: **phi** ESS 4.7-11.7% after the `integrated_autocorr_time` tolerance fix; the alm
   block was the binding worst case in that gate) — use those directly rather
   than rerunning.
4. **Wall-clock-per-effective-sample** — CMBLensing.jl reports full wall-times per config (Table II
   above) and per-parameter autocorrelation lengths, so this is directly computable from published
   numbers without rerunning: wall-time / (chain_iterations / autocorr_length). diffcmb has per-sweep
   wall-clock times recorded (29-54s/sweep at lmax=300) and can compute the same ratio from its own
   ESS numbers.
5. **Qualitative mixing-behavior comparison**: diffcmb's gate verdict was "MARGINAL, not pathological"
   — worth directly juxtaposing against CMBLensing.jl's own reported chain behavior (their Fig 2 shows
   the *unmixed* parametrization is catastrophic and their mixed one is good; no raw ESS numbers are
   quoted in the excerpt above, but autocorrelation lengths are, in Table II: 22 (2PARAM), 5-33 (MANY),
   12 (BIG)). This is comparable in kind to diffcmb's autocorrelation-based ESS estimator.

## 4. Install-vs-cite tradeoff

**Julia/CMBLensing.jl install feasibility on COSMA:**
- No existing Julia environment anywhere in this repo or (as far as searched) on the cluster profile.
- CMBLensing.jl officially recommends/ships a Docker container (per its README/docs) with GPU
  support; COSMA likely uses Singularity/Apptainer, not raw Docker, for SLURM jobs — converting a
  Docker image to Singularity is routine but adds a dependency-conversion step.
- CMBLensing.jl's own benchmark runs (Table II) took 19-50 **hours on one GPU** per configuration —
  this is a real compute cost, and COSMA GPU allocation/queue specifics for dc-hick2 haven't been
  checked (this task didn't investigate GPU partition availability — flag as an open question).
- The existing 200-job SLURM cap on `durham`/`dine2` (dc-hick2's account) doesn't block a small number
  of CMBLensing.jl jobs by itself, but combine with unknown GPU-queue availability/limits — needs a
  separate check before committing to "run it ourselves."

**Recommendation: cite published numbers as the primary path, with local install as a stretch goal
only if reviewers demand it.**
Rationale:
- CMBLensing.jl's own paper already reports exactly the config table, autocorrelation lengths, and
  wall-times needed for the wall-clock-per-ESS comparison (#4 above) — no rerun needed for that axis.
- Installing and validating a new Julia/GPU toolchain from scratch is a multi-day side-project with
  real failure risk (unfamiliar codebase, GPU/Singularity conversion, possible version drift since
  2020 given Julia's ecosystem churn) that doesn't obviously improve the comparison's rigor over using
  the paper's own published, peer-reviewed numbers for the *same configuration* diffcmb is targeting
  (f_sky-matched, CMB-S4-like noise).
- The one thing published numbers can't give: a live rerun on the *exact same simulated sky
  realization* diffcmb uses, for a truly apples-to-apples posterior-recovery plot (item #1/#2 above).
  If a referee specifically demands same-realization overlays (not just same-configuration-class
  comparison), that's the trigger to actually install and run CMBLensing.jl — not before.
- This is a methods-paper-critical, referee-facing call, not a routine one — flagged per the task
  brief for human decision, not silently picked.

## 5. Open questions / blockers for a human to decide

1. **Install vs. cite** (above) — recommend cite-first, but this is the human's call given it affects
   how strong the paper's benchmark claim can be.
2. **T-only vs P-only mismatch** — does diffcmb have a validated polarization (QU) path at all? If not,
   the comparison is necessarily T-vs-P, which is a real disanalogy to flag in the paper, not hide.
3. **lmax mismatch (300 vs ~3500)** — accept as scope-limited (recommended default) or invest in a
   higher-lmax matrix-free run specifically for this benchmark? Needs a cost estimate first (per-sweep
   cost scaling with lmax is not yet measured beyond lmax=300 in this repo, per achievements.md).
4. **GPU availability for dc-hick2 on COSMA** — not checked in this pass; needed only if "install and
   run" is chosen.
5. **Cosmology defaults** — CMBLensing.jl's own paper example uses different fiducial tau/omega_b/
   omega_c than diffcmb's LCDM_PARAMS; if citing published numbers, this is a genuine (if small)
   confound — note it rather than silently treating the two "LCDM" runs as identical.
