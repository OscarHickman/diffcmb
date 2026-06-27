"""
Comprehensive MCMC chain analysis for the diffcmb CMB sampling project.
No scipy/arviz/special MCMC libs - numpy + matplotlib only.
"""

import matplotlib
import numpy as np

matplotlib.use('Agg')
import os
import sys

import matplotlib.pyplot as plt

# ─── Configuration ────────────────────────────────────────────────────────────

BASE    = '/cosma/apps/durham/dc-hick2/diffcmb/results'
OUT_DIR = os.path.join(BASE, 'analysis')
os.makedirs(OUT_DIR, exist_ok=True)

RUNS = {
    'lmax300_double': {
        'dir'    : 'lmax300_nside256_gibbs_real_double',
        'lmax'   : 300,
        'label'  : 'lmax300 float64 Gibbs',
        'is_key' : True,
    },
    'lmax300_float32': {
        'dir'   : 'lmax300_nside256_gibbs_real',
        'lmax'  : 300,
        'label' : 'lmax300 float32 Gibbs',
    },
    'lmax200_gibbs': {
        'dir'   : 'lmax200_nside128_gibbs_real',
        'lmax'  : 200,
        'label' : 'lmax200 float32 Gibbs',
    },
    'lmax200_hmc': {
        'dir'   : 'lmax200_nside128_hmc_real_preconditioned',
        'lmax'  : 200,
        'label' : 'lmax200 float32 HMC precond',
    },
    'lmax64_nuts': {
        'dir'   : 'lmax64_nside32_nuts_real',
        'lmax'  : 64,
        'label' : 'lmax64 NUTS (reference)',
    },
}

BURN_FRAC = 0.20  # discard first 20% as burn-in
N_ALM_SAMPLE = 200  # number of alm params to sample for R-hat
RNG = np.random.default_rng(42)

# ─── Index helpers ─────────────────────────────────────────────────────────────

def param_slices(lmax):
    """Return (lncl_slice, realalm_slice, imagalm_slice) for given lmax."""
    n_lncl    = lmax - 2
    n_realalm = lmax * (lmax + 1) // 2 - 3
    lncl_sl      = slice(0, n_lncl)
    realalm_sl   = slice(n_lncl, n_lncl + n_realalm)
    imagalm_sl   = slice(n_lncl + n_realalm, None)
    return lncl_sl, realalm_sl, imagalm_sl

def lncl_index_for_ell(lmax, ell):
    """Return the index into lncl array for a given ell (ell=2..lmax-1)."""
    return ell - 2

# ─── Load all chains ───────────────────────────────────────────────────────────

def load_run(run_dir, lmax):
    """Load all 4 chains for a run. Returns (samples, logp, accept_rates)."""
    path = os.path.join(BASE, run_dir)
    chains = sorted([f for f in os.listdir(path) if f.startswith('chain_') and
                     f.endswith('.npz') and not f.startswith('checkpoint')])
    all_samples, all_logp, accept_rates = [], [], []
    for c in chains:
        d = np.load(os.path.join(path, c))
        all_samples.append(d['samples'].astype(np.float64))
        all_logp.append(d['logp'].astype(np.float64))
        accept_rates.append(float(d['accept_rate']))
    return np.stack(all_samples, axis=0), np.stack(all_logp, axis=0), accept_rates

# ─── Statistical helpers ───────────────────────────────────────────────────────

def gelman_rubin(chains):
    """
    Compute split R-hat for an array of shape (n_chains, n_samples).
    Split each chain in half first for better small-sample behaviour.
    Returns scalar R-hat.
    """
    n_chains, n = chains.shape
    # split each chain in half
    n_half = n // 2
    split = np.concatenate([chains[:, :n_half], chains[:, n_half:2*n_half]], axis=0)
    M, N = split.shape   # M = 2*n_chains, N = n_half

    chain_means = split.mean(axis=1)           # (M,)
    B = N * np.var(chain_means, ddof=1)        # between-chain variance * N
    W = np.mean(np.var(split, ddof=1, axis=1)) # within-chain variance
    if W == 0:
        return np.nan
    var_hat = (N - 1) / N * W + B / N
    return np.sqrt(var_hat / W)

def compute_rhat_batch(chains_3d):
    """
    chains_3d: (n_chains, n_samples, n_params)
    Returns rhat array of shape (n_params,)
    """
    n_chains, n_samples, n_params = chains_3d.shape
    rhat = np.zeros(n_params)
    for i in range(n_params):
        rhat[i] = gelman_rubin(chains_3d[:, :, i])
    return rhat

def autocorr(x):
    """Normalised autocorrelation of 1-D array x."""
    x = x - x.mean()
    n = len(x)
    # use FFT for speed
    f = np.fft.rfft(x, n=2*n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0]
    return acf

def ess_from_acf(x):
    """
    ESS = N / (1 + 2*sum(rho_k)) where rho_k is ACF up to first negative lag.
    """
    n = len(x)
    acf = autocorr(x)
    # sum until first negative value
    cumsum = 0.0
    for k in range(1, n):
        if acf[k] < 0:
            break
        cumsum += acf[k]
    return n / max(1.0 + 2.0 * cumsum, 1.0)

# ─── Main analysis ─────────────────────────────────────────────────────────────

summary_rows = []  # will hold dicts for summary table

def analyse_run(run_key, run_cfg):
    run_dir = run_cfg['dir']
    lmax    = run_cfg['lmax']
    label   = run_cfg['label']
    print(f"\n{'='*70}")
    print(f"  {label}  ({run_dir})")
    print(f"{'='*70}")

    # --- load ---
    samples, logp, accept_rates = load_run(run_dir, lmax)
    # samples: (n_chains, n_samples, n_params)
    n_chains, n_samples, n_params = samples.shape
    burn = int(n_samples * BURN_FRAC)
    post_samples = samples[:, burn:, :]   # post burn-in
    post_logp    = logp[:, burn:]

    lncl_sl, realalm_sl, imagalm_sl = param_slices(lmax)
    n_lncl = lncl_sl.stop - lncl_sl.start

    # L values for lncl (ell = 2..lmax-1)
    ells = np.arange(2, lmax)

    print(f"  chains: {n_chains}  samples: {n_samples}  params: {n_params}")
    print(f"  burn-in: {burn}  post-burn samples: {n_samples - burn}")
    print(f"  accept rates: {[f'{a:.3f}' for a in accept_rates]}")

    mean_accept = float(np.mean(accept_rates))
    logp_std    = float(np.std(post_logp))
    print(f"  mean accept: {mean_accept:.3f}   logp_std (post burn): {logp_std:.2f}")

    # ── 1. R-hat for ln C_l ──────────────────────────────────────────────
    print("\n  [1] R-hat for ln C_l ...")
    lncl_chains = post_samples[:, :, lncl_sl]   # (n_chains, post_n, n_lncl)
    rhat_lncl = compute_rhat_batch(lncl_chains)
    print(f"      median R-hat(lnCl): {np.nanmedian(rhat_lncl):.4f}")
    print(f"      max    R-hat(lnCl): {np.nanmax(rhat_lncl):.4f}")
    print(f"      frac > 1.1:          {np.mean(rhat_lncl > 1.1):.3f}")

    # ── 1b. R-hat for alm sample ─────────────────────────────────────────
    print("\n  [1b] R-hat for alm sample ...")
    n_realalm = realalm_sl.stop - realalm_sl.start
    n_imagalm = n_params - realalm_sl.stop
    n_alm_total = n_realalm + n_imagalm
    sample_size = min(N_ALM_SAMPLE, n_alm_total)
    alm_global_indices = RNG.choice(n_alm_total, size=sample_size, replace=False)
    # map back to global param indices
    alm_start = realalm_sl.start
    alm_param_indices = alm_start + alm_global_indices  # all alm params are contiguous after lncl
    alm_chains = post_samples[:, :, alm_param_indices]
    rhat_alm = compute_rhat_batch(alm_chains)
    print(f"      median R-hat(alm):  {np.nanmedian(rhat_alm):.4f}")
    print(f"      max    R-hat(alm):  {np.nanmax(rhat_alm):.4f}")
    print(f"      frac > 1.1:          {np.mean(rhat_alm > 1.1):.3f}")

    # ── 2. ESS for ln C_l ────────────────────────────────────────────────
    print("\n  [2] ESS for ln C_l ...")
    # Compute ESS per parameter, per chain, then average over chains
    ess_per_param = np.zeros((n_chains, n_lncl))
    for ci in range(n_chains):
        for pi in range(n_lncl):
            ess_per_param[ci, pi] = ess_from_acf(lncl_chains[ci, :, pi])
    ess_lncl = ess_per_param.mean(axis=0)   # average over chains
    print(f"      median ESS(lnCl):   {np.nanmedian(ess_lncl):.1f}")
    print(f"      min    ESS(lnCl):   {np.nanmin(ess_lncl):.1f}")
    print(f"      max    ESS(lnCl):   {np.nanmax(ess_lncl):.1f}")

    # ── 3. Trace plots ───────────────────────────────────────────────────
    print("\n  [3] Trace plots ...")
    target_ells = [e for e in [2, 10, 50, 100] if e < lmax]
    fig, axes = plt.subplots(len(target_ells) + 1, 1, figsize=(14, 3*(len(target_ells)+1)))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for ax, ell in zip(axes[:-1], target_ells, strict=False):
        idx = lncl_index_for_ell(lmax, ell)
        for ci in range(n_chains):
            ax.plot(samples[ci, :, idx], color=colors[ci], alpha=0.7, lw=0.6, label=f'chain {ci+1}')
        ax.axvline(burn, color='k', lw=1.0, ls='--', alpha=0.5)
        ax.set_ylabel(f'ln C_l  l={ell}', fontsize=9)
        ax.tick_params(labelsize=8)
    ax = axes[-1]
    for ci in range(n_chains):
        ax.plot(logp[ci], color=colors[ci], alpha=0.7, lw=0.6, label=f'chain {ci+1}')
    ax.axvline(burn, color='k', lw=1.0, ls='--', alpha=0.5)
    ax.set_ylabel('log p', fontsize=9)
    ax.set_xlabel('sample', fontsize=9)
    ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=7, ncol=4)
    fig.suptitle(f'Trace plots — {label}', fontsize=11)
    plt.tight_layout()
    fname = f'traces_{run_key}.png'
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"      saved {fname}")

    # ── 4. Power spectrum recovery ────────────────────────────────────────
    print("\n  [4] Power spectrum recovery ...")
    # lncl post-burn samples: (n_chains, post_n, n_lncl)
    all_lncl = lncl_chains.reshape(-1, n_lncl)   # (n_chains*post_n, n_lncl)
    cl_samples = np.exp(all_lncl)
    cl_mean = cl_samples.mean(axis=0)
    cl_lo   = np.percentile(cl_samples, 2.5, axis=0)
    cl_hi   = np.percentile(cl_samples, 97.5, axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(ells, cl_lo * ells*(ells+1)/(2*np.pi),
                    cl_hi * ells*(ells+1)/(2*np.pi), alpha=0.35, label='95% CI', color='steelblue')
    ax.plot(ells, cl_mean * ells*(ells+1)/(2*np.pi), color='navy', lw=1.2, label='Posterior mean')
    ax.set_xlabel('Multipole ℓ', fontsize=11)
    ax.set_ylabel('D_ℓ = ℓ(ℓ+1)Cℓ/2π  [μK²]', fontsize=10)
    ax.set_title(f'CMB Power Spectrum — {label}', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_yscale('log')
    plt.tight_layout()
    fname = f'power_spectrum_{run_key}.png'
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"      saved {fname}")

    # ── R-hat bar chart for ln C_l ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(ells, rhat_lncl, width=0.8, color='steelblue', alpha=0.8)
    ax.axhline(1.1, color='r', lw=1.2, ls='--', label='R-hat = 1.1')
    ax.axhline(1.0, color='k', lw=0.8, ls=':')
    ax.set_xlabel('Multipole ℓ', fontsize=11)
    ax.set_ylabel('R-hat', fontsize=11)
    ax.set_title(f'Gelman–Rubin R-hat for ln Cℓ — {label}', fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fname = f'rhat_lncl_{run_key}.png'
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"      saved {fname}")

    # ── Autocorrelation for key run ───────────────────────────────────────
    if run_cfg.get('is_key'):
        print("\n  [6] Autocorrelation for ln C_l at l=10 ...")
        idx10 = lncl_index_for_ell(lmax, 10)
        max_lag = min(500, (n_samples - burn) // 2)
        fig, ax = plt.subplots(figsize=(10, 4))
        for ci in range(n_chains):
            x = lncl_chains[ci, :, idx10]
            acf = autocorr(x)[:max_lag]
            ax.plot(acf, color=colors[ci], alpha=0.75, lw=1.0, label=f'chain {ci+1}')
        ax.axhline(0, color='k', lw=0.8)
        ax.set_xlabel('Lag', fontsize=11)
        ax.set_ylabel('ACF', fontsize=11)
        ax.set_title(f'Autocorrelation of ln C_ℓ (ℓ=10) — {label}', fontsize=11)
        ax.legend(fontsize=9)
        plt.tight_layout()
        fname = 'autocorr_lncl_l10_double.png'
        fig.savefig(os.path.join(OUT_DIR, fname), dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f"      saved {fname}")

        # Also: ESS over ell for the key run
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(ells, ess_lncl, width=0.8, color='teal', alpha=0.8)
        ax.set_xlabel('Multipole ℓ', fontsize=11)
        ax.set_ylabel('ESS (avg over chains)', fontsize=11)
        ax.set_title(f'Effective Sample Size for ln Cℓ — {label}', fontsize=11)
        plt.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, 'ess_lncl_double.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)
        print("      saved ess_lncl_double.png")

    # ── Store summary row ─────────────────────────────────────────────────
    summary_rows.append({
        'label'            : label,
        'n_chains'         : n_chains,
        'n_samples'        : n_samples,
        'burn'             : burn,
        'accept'           : mean_accept,
        'logp_std'         : logp_std,
        'median_rhat_cl'   : float(np.nanmedian(rhat_lncl)),
        'max_rhat_cl'      : float(np.nanmax(rhat_lncl)),
        'frac_rhat_cl_gt1p1': float(np.mean(rhat_lncl > 1.1)),
        'median_rhat_alm'  : float(np.nanmedian(rhat_alm)),
        'max_rhat_alm'     : float(np.nanmax(rhat_alm)),
        'frac_rhat_alm_gt1p1': float(np.mean(rhat_alm > 1.1)),
        'median_ess_cl'    : float(np.nanmedian(ess_lncl)),
        'min_ess_cl'       : float(np.nanmin(ess_lncl)),
    })

    return rhat_lncl, rhat_alm, ess_lncl, ells, cl_mean, cl_lo, cl_hi

# ─── Run all ──────────────────────────────────────────────────────────────────

results_by_key = {}
for run_key, run_cfg in RUNS.items():
    run_path = os.path.join(BASE, run_cfg['dir'])
    if not os.path.exists(run_path):
        print(f"\nSkipping {run_key}: directory not found")
        continue
    out = analyse_run(run_key, run_cfg)
    results_by_key[run_key] = out

# ─── 5. Summary table ─────────────────────────────────────────────────────────

print("\n\n" + "="*90)
print("  SUMMARY TABLE")
print("="*90)
hdr = (f"{'Run':<40} {'accept':>7} {'logp_std':>9} {'med_Rhat_Cl':>12} "
       f"{'med_Rhat_alm':>13} {'med_ESS_Cl':>11} {'frac_Rhat>1.1_Cl':>17}")
print(hdr)
print("-"*90)
for row in summary_rows:
    print(f"{row['label']:<40} {row['accept']:>7.3f} {row['logp_std']:>9.2f} "
          f"{row['median_rhat_cl']:>12.4f} {row['median_rhat_alm']:>13.4f} "
          f"{row['median_ess_cl']:>11.1f} {row['frac_rhat_cl_gt1p1']:>17.3f}")
print("="*90)

# ─── Multi-run overlay plots ──────────────────────────────────────────────────

# 5a. R-hat comparison for ln C_l (runs that have same lmax=300)
print("\n  [5] Multi-run comparison plots ...")

# Power spectrum overlay: lmax300 runs
if 'lmax300_double' in results_by_key and 'lmax300_float32' in results_by_key:
    fig, ax = plt.subplots(figsize=(11, 5))
    for run_key, color, ls in [
        ('lmax300_double',  'navy',   '-'),
        ('lmax300_float32', 'firebrick', '--'),
    ]:
        if run_key not in results_by_key:
            continue
        _, _, _, ells, cl_mean, cl_lo, cl_hi = results_by_key[run_key]
        dl_mean = cl_mean * ells*(ells+1)/(2*np.pi)
        dl_lo   = cl_lo   * ells*(ells+1)/(2*np.pi)
        dl_hi   = cl_hi   * ells*(ells+1)/(2*np.pi)
        ax.plot(ells, dl_mean, color=color, lw=1.5, ls=ls,
                label=RUNS[run_key]['label'])
        ax.fill_between(ells, dl_lo, dl_hi, alpha=0.2, color=color)
    ax.set_xlabel('Multipole ℓ', fontsize=11)
    ax.set_ylabel('D_ℓ = ℓ(ℓ+1)Cℓ/2π  [μK²]', fontsize=10)
    ax.set_title('Power Spectrum Comparison — lmax300', fontsize=11)
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'power_spectrum_lmax300_comparison.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)
    print("      saved power_spectrum_lmax300_comparison.png")

# All-run R-hat comparison (median R-hat per run as bar)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
labels = [r['label'] for r in summary_rows]
med_cl  = [r['median_rhat_cl']  for r in summary_rows]
med_alm = [r['median_rhat_alm'] for r in summary_rows]
x = np.arange(len(labels))
width = 0.35
axes[0].bar(x, med_cl, width, color='steelblue', alpha=0.85)
axes[0].axhline(1.1, color='r', lw=1.2, ls='--', label='1.1')
axes[0].axhline(1.0, color='k', lw=0.8, ls=':')
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels, rotation=25, ha='right', fontsize=8)
axes[0].set_ylabel('Median R-hat', fontsize=10)
axes[0].set_title('Median R-hat (ln Cℓ)', fontsize=10)
axes[0].legend(fontsize=8)

axes[1].bar(x, med_alm, width, color='darkorange', alpha=0.85)
axes[1].axhline(1.1, color='r', lw=1.2, ls='--', label='1.1')
axes[1].axhline(1.0, color='k', lw=0.8, ls=':')
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, rotation=25, ha='right', fontsize=8)
axes[1].set_ylabel('Median R-hat', fontsize=10)
axes[1].set_title('Median R-hat (alm sample)', fontsize=10)
axes[1].legend(fontsize=8)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'rhat_summary_all_runs.png'), dpi=120, bbox_inches='tight')
plt.close(fig)
print("      saved rhat_summary_all_runs.png")

# Final text summary
print("\n\n" + "="*70)
print("  KEY FINDINGS")
print("="*70)
for row in summary_rows:
    print(f"\n  {row['label']}")
    print(f"    accept rate     : {row['accept']:.3f}")
    print(f"    logp std        : {row['logp_std']:.2f}")
    print(f"    median R-hat Cl : {row['median_rhat_cl']:.4f}  (max {row['max_rhat_cl']:.4f})")
    print(f"    frac R-hat>1.1 Cl: {row['frac_rhat_cl_gt1p1']*100:.1f}%")
    print(f"    median R-hat alm: {row['median_rhat_alm']:.4f}  (max {row['max_rhat_alm']:.4f})")
    print(f"    frac R-hat>1.1 alm: {row['frac_rhat_alm_gt1p1']*100:.1f}%")
    print(f"    median ESS Cl   : {row['median_ess_cl']:.1f}  (min {row['min_ess_cl']:.1f})")

print("\nDone. All plots saved to:", OUT_DIR)
