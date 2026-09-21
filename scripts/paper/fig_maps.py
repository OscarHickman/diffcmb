"""Paper figure -- Full-sky field-level recovery maps on the sphere.

Produces standalone Mollweide projection vector/raster hybrid PDFs:
  (a) map_true_unlensed.pdf  -- True unlensed CMB primary temperature T(n)
  (b) map_observed_data.pdf  -- Observed lensed + noisy data map d(n)
  (c) map_true_phi.pdf       -- True lensing potential phi(n)
  (d) map_mean_phi.pdf       -- Posterior mean reconstructed potential <phi>(n)
  (e) map_residual_phi.pdf   -- Residual <phi>(n) - phi_true(n) with correlation

Adheres strictly to paper_style.py:
- No prose in panels (explanatory text in caption)
- Stat box in corner with quantitative metric (e.g. Pearson r = 0.816)
- Proper dimensions and Okabe-Ito / scientific diverging palettes

Usage:
  PYTHONPATH=diffcmb:scripts:scripts/paper .venv/bin/python scripts/paper/fig_maps.py \
      --chain results/analysis/coverage_ensemble_lmax64_prior_cl4_properprior_packingv2/chain_r000.npz \
      --outdir /cosma/apps/durham/dc-hick2/papers/7_DiffCMB/plots/figure_maps
"""

import argparse
import os
import sys

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import paper_style
import sbc_joint_likelihood

from diffcmb import CosmologyAdvancedSampling, alm_utils


def project_mollweide(m):
    """Render a HEALPix map into a 2D Mollweide projected grid."""
    fig_dummy = plt.figure(figsize=(4, 2))
    img = hp.mollview(m, return_projected_map=True)
    plt.close(fig_dummy)
    return np.ma.masked_equal(img, hp.UNSEEN)


def plot_single_mollweide(
    img,
    outpath: str,
    cmap: str,
    vmin: float = None,
    vmax: float = None,
    cbar_label: str = "",
    stat_text: str = None,
):
    paper_style.apply()
    fig, ax = plt.subplots(figsize=paper_style.FIG_1COL)

    im = ax.imshow(
        img,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )
    ax.axis("off")

    cbar = fig.colorbar(
        im,
        ax=ax,
        orientation="horizontal",
        fraction=0.048,
        pad=0.04,
        shrink=0.72,
    )
    cbar.ax.tick_params(labelsize=6)
    cbar.locator = plt.MaxNLocator(nbins=5)
    cbar.update_ticks()
    if cbar_label:
        cbar.set_label(cbar_label, fontsize=6.5, labelpad=2)

    if stat_text:
        paper_style.stat_box(
            ax, stat_text, loc="upper right", xy=(0.95, 0.90),
            color="0.1",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.85},
        )

    paper_style.save(fig, outpath)


def generate_maps(chain_path: str, outdir: str, burn_frac: float = 0.1) -> None:
    os.makedirs(outdir, exist_ok=True)
    paper_style.apply()

    print(f"Loading chain from {chain_path}...")
    d = np.load(chain_path)
    lmax = int(d["lmax"])
    nside = int(d["nside"])
    n_real, n_imag = alm_utils.packed_sizes(lmax)

    # 1. True unlensed CMB temperature map
    alm_true_packed = d["alm_true_packed"]
    alm_true_complex = np.asarray(
        alm_utils.splittosingularalm(
            alm_true_packed[:n_real], alm_true_packed[n_real:], lmax
        ),
        dtype=np.complex128,
    )
    T_unlensed = hp.alm2map(alm_true_complex, nside=nside)

    # 2. Observed noisy lensed data map d
    print("Reconstructing observed data map d...")
    model = CosmologyAdvancedSampling(
        _lmax=lmax,
        _NSIDE=nside,
        _noisesig=1.0,
        data_mode="synthetic",
        dtype=tf.complex128,
        use_matrixfree_sht=True,
    )
    model._ensure_tf_tensors()
    _, _, y_data = sbc_joint_likelihood.rebuild_truth_and_data(
        model, d, lmax, 1.0
    )

    # 3. True lensing potential phi (scaled by 1e4 for clean display)
    scale_phi = 1e4
    phi_true_packed = d["phi_true_packed"]
    phi_true_complex = np.asarray(
        alm_utils.splittosingularalm(
            phi_true_packed[:n_real], phi_true_packed[n_real:], lmax
        ),
        dtype=np.complex128,
    )
    phi_true = hp.alm2map(phi_true_complex, nside=nside) * scale_phi

    # 4. Posterior mean phi
    phi_samples = d["phi_samples"]
    n_samples = phi_samples.shape[0]
    burn = int(burn_frac * n_samples)
    phi_mean_packed = np.mean(phi_samples[burn:], axis=0)
    phi_mean_complex = np.asarray(
        alm_utils.splittosingularalm(
            phi_mean_packed[:n_real], phi_mean_packed[n_real:], lmax
        ),
        dtype=np.complex128,
    )
    phi_mean = hp.alm2map(phi_mean_complex, nside=nside) * scale_phi

    # 5. Residual and correlation
    phi_residual = phi_mean - phi_true
    corr = float(np.corrcoef(phi_true, phi_mean)[0, 1])

    # Scales
    t_lim = float(np.percentile(np.abs(T_unlensed), 99))
    d_lim = float(np.percentile(np.abs(y_data), 99))
    phi_lim = float(np.percentile(np.abs(phi_true), 99))
    res_lim = float(np.percentile(np.abs(phi_residual), 99))

    print(f"Generating projected Mollweide maps (Pearson r = {corr:.3f})...")

    # Render & save
    img_t = project_mollweide(T_unlensed)
    plot_single_mollweide(
        img_t,
        os.path.join(outdir, "map_true_unlensed.pdf"),
        cmap="RdYlBu_r",
        vmin=-t_lim,
        vmax=t_lim,
        cbar_label=r"Unlensed $T\,[\mu\mathrm{K}]$",
    )

    img_d = project_mollweide(y_data)
    plot_single_mollweide(
        img_d,
        os.path.join(outdir, "map_observed_data.pdf"),
        cmap="RdYlBu_r",
        vmin=-d_lim,
        vmax=d_lim,
        cbar_label=r"Observed $d\,[\mu\mathrm{K}]$",
    )

    img_phi = project_mollweide(phi_true)
    plot_single_mollweide(
        img_phi,
        os.path.join(outdir, "map_true_phi.pdf"),
        cmap="magma",
        vmin=-phi_lim,
        vmax=phi_lim,
        cbar_label=r"True $\phi \times 10^4$",
    )

    img_mean_phi = project_mollweide(phi_mean)
    plot_single_mollweide(
        img_mean_phi,
        os.path.join(outdir, "map_mean_phi.pdf"),
        cmap="magma",
        vmin=-phi_lim,
        vmax=phi_lim,
        cbar_label=r"Posterior Mean $\langle\phi\rangle \times 10^4$",
        stat_text=f"Pearson $r = {corr:.3f}$",
    )

    img_res = project_mollweide(phi_residual)
    plot_single_mollweide(
        img_res,
        os.path.join(outdir, "map_residual_phi.pdf"),
        cmap="coolwarm",
        vmin=-res_lim,
        vmax=res_lim,
        cbar_label=r"Residual $(\langle\phi\rangle - \phi_{\mathrm{true}}) \times 10^4$",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chain",
        default="results/analysis/coverage_ensemble_lmax64_prior_cl4_properprior_packingv2/chain_r000.npz",
        help="Path to production chain file.",
    )
    parser.add_argument(
        "--outdir",
        default="/cosma/apps/durham/dc-hick2/papers/7_DiffCMB/plots/figure_maps",
        help="Directory to write panel PDFs.",
    )
    parser.add_argument(
        "--burn_frac",
        type=float,
        default=0.1,
        help="Fraction of chain to discard as burn-in.",
    )
    args = parser.parse_args()
    generate_maps(args.chain, args.outdir, args.burn_frac)


if __name__ == "__main__":
    main()
