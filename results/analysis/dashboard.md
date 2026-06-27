# MCMC Sampling Dashboard
*Last Updated: 2026-06-27*

---

## Summary Table — All Runs

| Run | Sampler | Precision | Chains×Samples | Accept | logp std | R-hat C_l (med/max) | R-hat alm (med) | ESS C_l | Status |
|-----|---------|-----------|---------------|--------|----------|---------------------|-----------------|---------|--------|
| lmax300 Gibbs | Gibbs | **float64** | 4×1000 | **71%** | **26 051** | **1.026 / 1.085** | 2.64 | **385** | ✅ C_l CONVERGED |
| lmax300 Gibbs | Gibbs | float32 | 4×2000 | 38% | 634 | 1.000 / 1.001 | 58 112 | 1 553 | ❌ alm FROZEN |
| lmax200 Gibbs | Gibbs | float32 | 4×2000 | 64% | 130 | 1.000 / 1.001 | 17 619 | 1 551 | ❌ alm FROZEN |
| lmax200 HMC prec. | HMC | float32 | 4×5000 | 64% | 3.4M | 2 985 / — | 36 608 | 5 | ❌ DIVERGED |
| lmax64 NUTS | NUTS | float32 | 4×2000 | 100% | 53 | 1.180 / — | 1.087 | 12.5 | ⚠️ needs samples |

---

## Float32 failure — confirmed root cause

Gradient noise in the spherical harmonic matmul accumulates across ~607k unmasked Planck
pixels, driving HMC step size to its floor (~1e-7). Nominal 38–65% acceptance, but chains
move <2e-6 in whitened alm space per step — effectively frozen.

Diagnostic trap: C_l R-hat appears perfect (1.000) because each chain is stuck at a
different frozen alm realisation and rapidly converges within its own stuck mode.

---

## Float64 result — headline (2026-06-27)

**lmax300, nside256, Gibbs, float64 — 4 chains × 1000 samples**

| Metric | Value |
|--------|-------|
| Accept rate | 60–82% (mean 71%) |
| logp std vs float32 | 26 051 vs 13 — genuinely exploring |
| C_l R-hat median | **1.026** |
| C_l R-hat max | **1.085** |
| C_l R-hat > 1.1 | **0%** |
| ESS (ln C_l, median) | **385** / 800 post-burn (48% efficiency) |
| ACF ln C_l | drops to 0 at lag 1 — near-independent draws |
| alm R-hat median | 2.64 (100% > 1.1) — expected at 89k dims |

**Open issues:**
- logp trace is monotonically decreasing over all 1000 samples — chains still transitioning
  from MAP to the typical set. Need ~3× more samples to reach plateau.
- alm inter-chain convergence not achieved — scientifically secondary to C_l, but needed
  for full joint posterior uncertainty.
- No comparison to official Planck 2018 TT spectrum yet.

---

## Next Steps (m1)

1. **Planck comparison** — overlay recovered C_l on CAMB ΛCDM best-fit + Planck 2018 data.
2. **Extend L=300 float64** — restart from checkpoints, 3000+ more samples, `--map_steps 0`.
3. **L=200 float64 baseline** — 39 996 params, faster convergence, clean reference.
4. **Phase 3 start** — use marginalised C_l to infer ΛCDM params via emcee.
