# MCMC Sampling Dashboard

## Synthetic Data Baseline — L=200, NSIDE=128

Results from initial sampler comparison (Phase 1). These used synthetic CMB data.

| Sampler | Chains | Avg Accept | R-hat median ($\ln C_\ell$) | Conv. fraction ($\ln C_\ell$, R-hat < 1.1) | Alm R-hat | Notes |
|:--------|:------:|:----------:|:---------------------------:|:-------------------------------------------:|:---------:|:------|
| Preconditioned HMC | 4 | 0.636 | 2062.6 | 0.0% | — | HMC without Gibbs; does not converge |
| Gibbs (Frozen nuisance) | 4 | 0.623 | 1.0188 | 84.8% | — | Partial convergence |
| Gibbs (Deep MAP stabilized) | 4 | 0.636 | **1.0000** | **100.0%** | *not measured* | Best synthetic result; alm mixing unverified |

> **Note (2026-06-14):** Alm R-hat was not computed for synthetic runs. When real Planck data was applied, alm R-hat was found to be catastrophically large, indicating alm chains were never mixing. The 100% convergence figure refers to $\ln C_\ell$ only.

---

## Real Planck Data — L=200, NSIDE=128 (float32)

Run date: 2026-06-12. Chains from `results/lmax200_nside128_gibbs_real`.

| Chain | Accept | logp mean | logp std |
|:-----:|:------:|:---------:|:--------:|
| 1 | 0.635 | 149 476 543 | 10.4 |
| 2 | 0.589 | 149 476 268 | 10.9 |
| 3 | 0.677 | 149 476 199 | 10.6 |
| 4 | 0.643 | 149 476 304 | 10.8 |

R-hat ($\ln C_\ell$): max = 1.001 | R-hat alm: **median = 12 963**

**Status: BROKEN.** HMC step collapsed to ~1e-7. `logp_std ≈ 10.5` on a posterior of ~1.5e8 — no exploration.

---

## Real Planck Data — L=300, NSIDE=256 (float32)

Run date: 2026-06-12–14. Chains from `results/lmax300_nside256_gibbs_real`. MAP: 2000 Adam steps (~1.9 h). Tensor load: ~346 s (21-part split).

| Chain | Accept | logp mean | logp std | Wall time |
|:-----:|:------:|:---------:|:--------:|:---------:|
| 1 | 0.363 | 471 924 265 | 13.2 | ~44 h |
| 2 | 0.399 | 471 925 727 | 13.5 | ~44 h |
| 3 | 0.352 | 471 924 291 | 13.4 | ~44 h |
| 4 | 0.388 | 471 924 238 | 12.9 | ~44 h |

R-hat ($\ln C_\ell$): max = 1.001, median = 1.000 | R-hat alm: **median = 46 458**, max = 1 874 037
Parameters with R-hat < 1.1: **0.3%**

**Status: BROKEN.** Same root cause as L=200. Larger matrix → more float32 gradient noise → worse alm R-hat.

---

## Real Planck Data — L=300 (float64 short test)

50 samples, 1 chain, `results/test_dp`. Confirms double-precision fix.

| Metric | float32 L=300 | float64 test |
|:-------|:-------------:|:------------:|
| Accept rate | ~38% | **84%** |
| logp std | 13.2 | **55.5** |
| Alm mixing | Frozen | Exploring |

**Status: FIX CONFIRMED.** Full 4-chain double-precision run is Job 11383627 (`cmb_gibbs_L300_dp`), scheduled start **2026-06-16 18:00** on `dine2/gc005`.

---

## Visualizations (auto-generated)

- `traces_real_gibbs_L200.png`, `traces_real_gibbs_L300.png`
- `power_spectrum_real_gibbs_L200.png`, `power_spectrum_real_gibbs_L300.png`
- `rhat_real_gibbs_L200.png`, `rhat_real_gibbs_L300.png`

*Last Updated: 2026-06-14*
