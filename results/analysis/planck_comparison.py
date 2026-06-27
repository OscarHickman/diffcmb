"""
Planck 2018 comparison plot for the lmax=300 float64 Gibbs chain.

Units/normalization notes printed to stdout at runtime.

Parameter layout (lmax=300, 89996 params per sample):
  params[0:298]    = ln C_l  for l=2..299   (lncl_sl)
  params[298:...]  = real/imag alm components (not used here)

Power spectrum conventions:
  Our chains: C_l in μK²  (standard CMB convention, same as hp.anafast output)
  CAMB call_CAMB_map returns _CL[i] = D_l[i] / (l*(l+1)) = C_l[i]/(2π)  for i>0
    → so D_l_camb = l*(l+1) * _CL[i]   [μK²]
  Planck data: D_l directly in μK²

Plot:
  Panel 1: D_l vs l (log scale), with Planck data, CAMB best-fit, and our posterior
  Panel 2: ratio  posterior_mean_D_l / D_l_camb

Saved to: results/analysis/planck_comparison.png
"""

import os
import sys
import warnings

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Paths ──────────────────────────────────────────────────────────────────────
CHAIN_DIR = '/cosma/apps/durham/dc-hick2/diffcmb/results/lmax300_nside256_gibbs_real_double'
OUT_DIR   = '/cosma/apps/durham/dc-hick2/diffcmb/results/analysis'
PLANCK_FILE = '/tmp/planck_tt.txt'
OUT_PNG   = os.path.join(OUT_DIR, 'planck_comparison.png')

LMAX      = 300
BURN_FRAC = 0.20   # discard first 20% of samples

os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load chains ──────────────────────────────────────────────────────────────
print("=" * 70)
print("  Loading chains from", CHAIN_DIR)
print("=" * 70)

chain_files = sorted([
    f for f in os.listdir(CHAIN_DIR)
    if f.startswith('chain_') and f.endswith('.npz') and 'checkpoint' not in f
])
print(f"  Found chain files: {chain_files}")

all_samples = []
for cf in chain_files:
    d = np.load(os.path.join(CHAIN_DIR, cf))
    all_samples.append(d['samples'].astype(np.float64))
    print(f"    {cf}: shape={d['samples'].shape}, accept_rate={float(d['accept_rate']):.3f}")

# Stack: (n_chains, n_samples, n_params)
samples = np.stack(all_samples, axis=0)
n_chains, n_samples, n_params = samples.shape
print(f"\n  Combined shape: {samples.shape}")

# Burn-in
burn = int(n_samples * BURN_FRAC)
post_samples = samples[:, burn:, :]   # (n_chains, post_n, n_params)
print(f"  Burn-in: {burn} samples (first {BURN_FRAC*100:.0f}%)")
print(f"  Post-burn samples per chain: {n_samples - burn}")

# ── 2. Extract ln C_l posterior ────────────────────────────────────────────────
# params[0:lmax-2] = ln C_l for l=2..lmax-1
n_lncl   = LMAX - 2   # 298 values (l=2..299)
lncl_sl  = slice(0, n_lncl)

lncl_chains  = post_samples[:, :, lncl_sl]          # (n_chains, post_n, n_lncl)
lncl_all     = lncl_chains.reshape(-1, n_lncl)       # (n_chains*post_n, n_lncl)

ells         = np.arange(2, LMAX)                    # l = 2..299  (length 298)
cl_samples   = np.exp(lncl_all)                      # C_l in μK²
cl_mean      = cl_samples.mean(axis=0)               # posterior mean C_l
cl_lo        = np.percentile(cl_samples,  2.5, axis=0)
cl_hi        = np.percentile(cl_samples, 97.5, axis=0)

# Convert to D_l = l(l+1) C_l / (2π)  [μK²]
dl_factor    = ells * (ells + 1) / (2.0 * np.pi)
dl_mean      = cl_mean * dl_factor
dl_lo        = cl_lo   * dl_factor
dl_hi        = cl_hi   * dl_factor

print(f"\n  Posterior C_l range: [{cl_mean.min():.3e}, {cl_mean.max():.3e}] μK²")
print(f"  Posterior D_l range: [{dl_mean.min():.3e}, {dl_mean.max():.3e}] μK²")
print(f"  D_l at l=220 (≈1st peak): {dl_mean[218]:.1f} μK²")

# ── 3. CAMB best-fit ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  Running CAMB (Planck 2018 ΛCDM best-fit parameters)")
print("=" * 70)

dl_camb    = None
cl_camb    = None   # proper C_l (not divided by 2π)
camb_ells  = None

try:
    sys.path.insert(0, '/cosma/apps/durham/dc-hick2/diffcmb/diffcmb/diffcmb')
    from power import call_CAMB_map

    planck_params = [67.74, 0.02230, 0.1188, 0.06, 0.0, 0.0544]
    print(f"  Parameters: H0={planck_params[0]}, ombh2={planck_params[1]}, "
          f"omch2={planck_params[2]}, mnu={planck_params[3]}, "
          f"omk={planck_params[4]}, tau={planck_params[5]}")

    raw_cl = call_CAMB_map(planck_params, LMAX)  # length LMAX array

    # call_CAMB_map returns _CL[i] = D_l[i]/(l*(l+1)) = C_l[i]/(2π)  for i>0
    # so _CL is NOT the standard C_l — it is C_l/(2π).
    # Therefore D_l = l*(l+1) * raw_cl[i]  (NOT dividing by 2π again).
    camb_ells  = np.arange(len(raw_cl))
    mask       = (camb_ells >= 2) & (camb_ells < LMAX)
    camb_ells  = camb_ells[mask]
    raw_cl_cut = raw_cl[mask]

    # D_l from CAMB: l*(l+1) * raw_cl (raw_cl already has the 1/(2π) factor built in)
    dl_camb    = camb_ells * (camb_ells + 1) * raw_cl_cut   # [μK²]

    # Proper C_l for ratio comparison: C_l = 2π * raw_cl
    cl_camb    = 2.0 * np.pi * raw_cl_cut                  # [μK²]

    print("\n  UNITS CHECK — call_CAMB_map output:")
    print(f"    raw_cl[2..10]  = {raw_cl[2:11]}")
    print("    These are C_l/(2π) in μK², so multiply by l*(l+1) to get D_l.")
    print(f"    D_l_camb at l=2..10 = {dl_camb[:9]}")
    print(f"    D_l_camb range (l=2..299): [{dl_camb.min():.3e}, {dl_camb.max():.3e}] μK²")
    if 218 < len(dl_camb):
        print(f"    D_l_camb at l≈220:  {dl_camb[218]:.1f} μK²")

    print("\n  NORMALIZATION COMPARISON (posterior mean vs CAMB) at l=10,50,100,200:")
    for l_ref in [10, 50, 100, 200]:
        idx_our  = l_ref - 2
        idx_camb = l_ref - 2   # camb_ells starts at 2
        if idx_our < len(dl_mean) and idx_camb < len(dl_camb):
            ratio = dl_mean[idx_our] / dl_camb[idx_camb] if dl_camb[idx_camb] > 0 else np.nan
            print(f"    l={l_ref:3d}:  D_l_ours={dl_mean[idx_our]:.2f}  "
                  f"D_l_camb={dl_camb[idx_camb]:.2f}  ratio={ratio:.3f}")

except Exception as exc:
    print(f"  CAMB failed: {exc}")
    print("  Proceeding without CAMB best-fit line.")

# ── 4. Planck 2018 data ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  Loading Planck 2018 TT data from", PLANCK_FILE)
print("=" * 70)

pl_ells = pl_dl = pl_err_lo = pl_err_hi = None

try:
    pl_data  = np.loadtxt(PLANCK_FILE, comments='#')
    pl_ells  = pl_data[:, 0].astype(int)
    pl_dl    = pl_data[:, 1]      # D_l in μK²
    pl_err_lo = pl_data[:, 2]     # lower 1-sigma error bar
    pl_err_hi = pl_data[:, 3]     # upper 1-sigma error bar

    # Restrict to l <= LMAX
    mask      = pl_ells <= LMAX
    pl_ells   = pl_ells[mask]
    pl_dl     = pl_dl[mask]
    pl_err_lo = pl_err_lo[mask]
    pl_err_hi = pl_err_hi[mask]

    print(f"  Loaded {len(pl_ells)} data points, l={pl_ells[0]}..{pl_ells[-1]}")
    print(f"  D_l range: [{pl_dl.min():.3e}, {pl_dl.max():.3e}] μK²")
    print(f"  Planck D_l at l=10: {pl_dl[pl_ells == 10][0] if 10 in pl_ells else 'N/A':.2f} μK²")
    print(f"  Planck D_l at l=220: {pl_dl[pl_ells == 220][0] if 220 in pl_ells else 'N/A':.2f} μK²")

except Exception as exc:
    print(f"  Planck data load failed: {exc}")
    print("  Proceeding without Planck data points.")

# ── 5. Build figure ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  Building figure ...")
print("=" * 70)

fig = plt.figure(figsize=(12, 10))
gs  = GridSpec(2, 1, figure=fig, height_ratios=[3, 1.2], hspace=0.08)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)

# ── Panel 1: D_l comparison ────────────────────────────────────────────────────

# Planck data points (grey error bars)
if pl_ells is not None:
    ax1.errorbar(
        pl_ells, pl_dl,
        yerr=[pl_err_lo, pl_err_hi],
        fmt='o', markersize=2.0, color='#888888', elinewidth=0.8, capsize=1.5,
        alpha=0.8, zorder=2, label='Planck 2018 (TT, R3.01)'
    )

# CAMB best-fit line (black dashed)
if dl_camb is not None:
    ax1.plot(
        camb_ells, dl_camb,
        color='black', lw=1.5, ls='--', zorder=4,
        label='CAMB ΛCDM best-fit (Planck 2018 params)'
    )

# Our posterior — 95% CI shaded, then mean line
ax1.fill_between(
    ells, dl_lo, dl_hi,
    color='#4ea6dc', alpha=0.35, zorder=3,
    label='Posterior 95% CI'
)
ax1.plot(
    ells, dl_mean,
    color='#1565c0', lw=1.8, zorder=5,
    label='Posterior mean (this work)'
)

ax1.set_yscale('log')
ax1.set_xlim(2, LMAX)

# Set y-axis limits: 10 to a bit above the Planck acoustic peak
ymin = 10.0
ymax_candidates = []
if pl_dl is not None:
    ymax_candidates.append(pl_dl.max() * 3.0)
if dl_camb is not None:
    ymax_candidates.append(dl_camb.max() * 3.0)
ymax_candidates.append(dl_hi.max() * 2.0)
ymax = max(ymax_candidates) if ymax_candidates else 2e4
ax1.set_ylim(ymin, ymax)

ax1.set_ylabel(r'$D_\ell = \ell(\ell+1)C_\ell / 2\pi \;\; [\mu\mathrm{K}^2]$', fontsize=13)
ax1.legend(fontsize=10, loc='upper left', framealpha=0.85)
ax1.set_title(
    'CMB TT Power Spectrum — Planck 2018 comparison\n'
    r'$\ell_{\max}=300$, NSIDE=256, float64 Gibbs (4 chains $\times$ 800 post-burn samples)',
    fontsize=11
)
ax1.tick_params(labelsize=10)
ax1.grid(True, which='both', alpha=0.25, lw=0.5)
ax1.tick_params(axis='x', labelbottom=False)

# ── Panel 2: ratio posterior mean / CAMB ──────────────────────────────────────
if dl_camb is not None:
    # camb_ells and ells both start at l=2; camb covers l=2..lmax-1
    # find the common ells
    common_ells   = np.intersect1d(ells, camb_ells)
    idx_our       = np.searchsorted(ells, common_ells)
    idx_camb_arr  = np.searchsorted(camb_ells, common_ells)

    ratio_mean = dl_mean[idx_our] / dl_camb[idx_camb_arr]
    ratio_lo   = dl_lo[idx_our]   / dl_camb[idx_camb_arr]
    ratio_hi   = dl_hi[idx_our]   / dl_camb[idx_camb_arr]

    ax2.fill_between(common_ells, ratio_lo, ratio_hi,
                     color='#4ea6dc', alpha=0.35)
    ax2.plot(common_ells, ratio_mean, color='#1565c0', lw=1.5)

    ax2.axhline(1.0, color='black', lw=1.2, ls='--', alpha=0.8)
    ax2.set_ylim(0.05, 20.0)
    ax2.set_yscale('log')
    ax2.set_yticks([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    ax2.set_yticklabels(['0.1', '0.5', '1', '2', '5', '10'])
    ax2.set_ylabel('Ratio\n(ours / CAMB)', fontsize=10)
    ax2.grid(True, which='both', alpha=0.25, lw=0.5)

    print("  Ratio posterior/CAMB at selected multipoles:")
    for l_ref in [10, 50, 100, 200, 299]:
        if l_ref in common_ells:
            idx = np.searchsorted(common_ells, l_ref)
            print(f"    l={l_ref:3d}: ratio = {ratio_mean[idx]:.3f}  "
                  f"(95% CI: [{ratio_lo[idx]:.3f}, {ratio_hi[idx]:.3f}])")
else:
    ax2.text(0.5, 0.5, 'CAMB not available — ratio panel empty',
             ha='center', va='center', transform=ax2.transAxes, fontsize=10)
    ax2.set_ylabel('Ratio', fontsize=10)

ax2.set_xlabel(r'Multipole $\ell$', fontsize=13)
ax2.tick_params(labelsize=10)

plt.tight_layout()
fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\n  Saved plot to: {OUT_PNG}")

# ── 6. Final summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  UNITS / NORMALIZATION SUMMARY")
print("=" * 70)
print("""
  Sampled parameter:  ln C_l  where C_l is in μK²  (standard CMB convention)
  This matches hp.anafast() output units from the Planck SMICA map (×1e6 K→μK).

  Conversion to plot units:
    D_l = l*(l+1) * C_l / (2π)   [μK²]   ← what we plot

  call_CAMB_map() returns _CL[i] = D_l[i]/(l*(l+1)) = C_l[i]/(2π)
    → to get D_l from CAMB:  D_l = l*(l+1) * _CL[i]   (no extra 2π needed)

  Planck data:  D_l directly in μK².  File: COM_PowerSpect_CMB-TT-full_R3.01.txt
""")

if dl_camb is not None and pl_dl is not None:
    # Quick sanity: compare Planck data with CAMB at a few multipoles
    print("  Sanity: Planck data vs CAMB at selected l:")
    for l_ref in [10, 50, 100, 200]:
        pl_mask = pl_ells == l_ref
        cb_mask = camb_ells == l_ref
        if pl_mask.any() and cb_mask.any():
            print(f"    l={l_ref:3d}:  Planck D_l={pl_dl[pl_mask][0]:.2f}  "
                  f"CAMB D_l={dl_camb[cb_mask][0]:.2f}  "
                  f"ratio={pl_dl[pl_mask][0]/dl_camb[cb_mask][0]:.3f}")

print("\nDone.")
