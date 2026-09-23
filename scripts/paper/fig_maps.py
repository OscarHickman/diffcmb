"""Paper figure -- full-sky field-level recovery on the sphere (one realization).

Standalone Mollweide panels, one PDF each (plots/STORY.md panel convention):

  map_true_unlensed.pdf    true unlensed temperature T(n)
  map_observed_data.pdf    observed data d(n) = lensed T + noise
  map_lensing_signal.pdf   noise-free lensing signal T_lensed(n) - T(n)
  map_true_phi.pdf         true lensing potential phi(n)
  map_mean_phi.pdf         posterior mean <phi>(n)
  map_residual_phi.pdf     (<phi> - phi_true) / sigma_phi(n), per-pixel posterior sd

The last panel is the one a referee can check: under a calibrated posterior
the normalised residual has unit variance. Its stat box carries the PER-MODE
sd of z_LM = (<phi_LM> - phi_LM^true)/sigma_LM, not the pixel sd: the phi map
is dominated by a handful of L = 2-5 modes (C_L^phiphi falls steeply), so the
pixel sd of one sky is a noisy statistic of ~10 numbers (1.26 on realization 0
while its per-mode sd is 1.005). With sigma estimated from ~20 effective
samples the expected per-mode sd is ~1.03-1.05 (Student-t), not exactly 1.

ORDERING. Chain vectors are in the packed *author* ordering. Every conversion
to a healpy alm goes through `lensing._alm_packed_to_hp` (tested round-trip).
The first version of this script called `splittosingularalm` and handed the
result straight to `hp.alm2map`, which reads m-major healpy ordering: every
coefficient landed at the wrong (l, m), producing zonal stripes and a
meaningless correlation coefficient. Do not bypass the converter.

Burn-in: the production chains discard `n_burnin` sweeps before saving, so
every saved sample is post-burn-in; nothing further is dropped here.

Usage:
  PYTHONPATH=diffcmb:scripts:scripts/paper .venv/bin/python scripts/paper/fig_maps.py
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paper_style  # noqa: E402

from diffcmb.alm_utils import packed_length  # noqa: E402
from diffcmb.lensing import _alm_packed_to_hp  # noqa: E402

DEFAULT_CHAIN = "results/analysis/ens_exact_l64_A3000_n30/chain_r000.npz"
DEFAULT_OUT = "/cosma/apps/durham/dc-hick2/papers/7_DiffCMB/plots/figure_maps"
PHI_SCALE = 1e4          # display phi x 1e4
XSIZE = 1200             # Mollweide raster width (pixels) before placement


def to_map(packed: np.ndarray, lmax: int, nside: int) -> np.ndarray:
    return hp.alm2map(_alm_packed_to_hp(np.asarray(packed, np.float64), lmax),
                      nside, lmax=lmax - 1)


def mollweide(m: np.ndarray) -> np.ma.MaskedArray:
    proj = hp.projector.MollweideProj(xsize=XSIZE)
    nside = hp.npix2nside(m.size)
    img = proj.projmap(m, lambda x, y, z: hp.vec2pix(nside, x, y, z))
    return np.ma.masked_invalid(np.where(img == hp.UNSEEN, np.nan, img))


def panel(img, path, cmap, lim, label, stat=None, symmetric=True):
    fig, ax = plt.subplots(figsize=paper_style.FIG_MAP)
    vmin = -lim if symmetric else 0.0
    im = ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=lim,
                   interpolation="nearest", rasterized=True)
    ax.set_axis_off()
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.06,
                      pad=0.03, shrink=0.70, aspect=28)
    cb.set_ticks(np.linspace(vmin, lim, 5))
    cb.ax.tick_params(labelsize=6.5, length=2, width=0.5, direction="out")
    cb.outline.set_linewidth(0.5)
    cb.set_label(label, fontsize=7, labelpad=1.5)
    if stat:
        paper_style.stat_box(ax, stat, loc="upper right", xy=(1.0, 1.0))
    paper_style.save(fig, path)


def mode_z(samples: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-packed-coordinate (<x> - x_true)/sd over the chain."""
    return (samples.mean(axis=0) - truth) / samples.std(axis=0, ddof=1)


def nice_lim(x: np.ndarray) -> float:
    """Symmetric colour limit: 99.5th percentile of |x|, rounded to 2 s.f."""
    v = float(np.percentile(np.abs(x), 99.5))
    return float(f"{v:.2g}")


def rebuild_data(d, lmax, nside, noisesig):
    """Observed map via the verified replay path (checks truth against the chain)."""
    import sbc_joint_likelihood
    import tensorflow as tf

    from diffcmb import CosmologyAdvancedSampling

    model = CosmologyAdvancedSampling(
        _lmax=lmax, _NSIDE=nside, _noisesig=noisesig, data_mode="synthetic",
        dtype=tf.complex128, use_matrixfree_sht=True,
        lensing_operator=sbc_joint_likelihood.chain_lensing_operator(d))
    model._ensure_tf_tensors()
    from diffcmb.lensing import lens_map_tf

    alm_p, phi_p, y = sbc_joint_likelihood.rebuild_truth_and_data(model, d, lmax, noisesig)
    T_len = lens_map_tf(model, tf.constant(alm_p, tf.float64),
                        _alm_packed_to_hp(phi_p, lmax)).numpy()
    return np.asarray(y), np.asarray(T_len)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chain", default=DEFAULT_CHAIN)
    p.add_argument("--outdir", default=DEFAULT_OUT)
    p.add_argument("--noisesig", type=float, default=1.0,
                   help="Must match the ensemble's --noisesig (production default 1.0).")
    p.add_argument("--thin", type=int, default=2)
    a = p.parse_args()

    paper_style.apply()
    os.makedirs(a.outdir, exist_ok=True)
    d = np.load(a.chain)
    lmax, nside = int(d["lmax"]), int(d["nside"])
    n_pk = packed_length(lmax)
    if d["phi_true_packed"].size != n_pk or d["phi_samples"].shape[1] != n_pk:
        raise SystemExit(f"{a.chain}: packed width != packed_length({lmax}) = {n_pk}; "
                         "pre-2026-09-06 chain, refusing.")

    T_unl = to_map(d["alm_true_packed"], lmax, nside)
    import sbc_joint_likelihood

    noisesig = sbc_joint_likelihood.chain_noisesig(d, a.noisesig)
    y, T_len = rebuild_data(d, lmax, nside, noisesig)
    dT = T_len - T_unl
    phi_true = to_map(d["phi_true_packed"], lmax, nside) * PHI_SCALE

    phi_maps = np.array([to_map(s, lmax, nside) for s in d["phi_samples"][::a.thin]])
    phi_maps *= PHI_SCALE
    phi_mean = phi_maps.mean(axis=0)
    phi_sd = phi_maps.std(axis=0, ddof=1)
    z = (phi_mean - phi_true) / phi_sd
    r = float(np.corrcoef(phi_true, phi_mean)[0, 1])
    z_mode = mode_z(d["phi_samples"], d["phi_true_packed"])
    z_ens = [mode_z(e["phi_samples"], e["phi_true_packed"])
             for e in map(np.load, sorted(glob.glob(
                 os.path.join(os.path.dirname(a.chain), "chain_r[0-9][0-9][0-9].npz"))))]

    t_lim = nice_lim(np.concatenate([T_unl, y]))
    phi_lim = nice_lim(phi_true)
    stem = lambda n: os.path.join(a.outdir, n)  # noqa: E731
    panel(mollweide(T_unl), stem("map_true_unlensed.pdf"), "RdBu_r", t_lim,
          r"unlensed $T$ [$\mu$K]")
    panel(mollweide(y), stem("map_observed_data.pdf"), "RdBu_r", t_lim,
          r"observed $d$ [$\mu$K]")
    panel(mollweide(dT), stem("map_lensing_signal.pdf"), "RdBu_r", nice_lim(dT),
          r"lensing signal $\tilde T-T$ [$\mu$K]",
          stat=r"$\sigma_{\rm pix}=" + f"{dT.std():.1f}" + r"\,\mu$K")
    panel(mollweide(phi_true), stem("map_true_phi.pdf"), "PuOr_r", phi_lim,
          r"true $\phi\;[\times10^{-4}]$")
    panel(mollweide(phi_mean), stem("map_mean_phi.pdf"), "PuOr_r", phi_lim,
          r"posterior mean $\langle\phi\rangle\;[\times10^{-4}]$",
          stat=f"$r = {r:.2f}$")
    panel(mollweide(z), stem("map_residual_phi.pdf"), "RdBu_r", 3.0,
          r"$(\langle\phi\rangle-\phi_{\rm true})/\sigma_\phi$",
          stat=r"$\mathrm{sd}_{LM}(z)=" + f"{z_mode.std():.2f}$")

    n_used = phi_maps.shape[0]
    print("\nCAPTION FACTS (do not draw these into the panels):")
    print(f"  chain: {a.chain} (realization {int(d['realization'])}, Block 4 "
          f"{'ON' if bool(d['sample_cl_phiphi']) else 'OFF'}, "
          f"nu={float(d['cl_phiphi_prior_nu'])})")
    print(f"  lmax={lmax}, nside={nside}, noise sigma={noisesig} muK per pixel, "
          f"operator={sbc_joint_likelihood.chain_lensing_operator(d)}")
    print(f"  posterior mean / sd from {n_used} samples (thin={a.thin}, all post-burn-in)")
    print(f"  pixel correlation r(<phi>, phi_true) = {r:.3f}")
    print(f"  per-mode z_LM: this sky mean {z_mode.mean():+.3f} sd {z_mode.std():.3f}; "
          f"pooled over {len(z_ens)} skies sd {np.concatenate(z_ens).std():.3f} "
          "(~1.03-1.05 expected at ESS~20)")
    print(f"  pixel z (map panel): mean {z.mean():+.3f}, sd {z.std():.3f} -- dominated by "
          "L=2-5, do not quote")
    print(f"  lensing signal rms {dT.std():.2f} muK vs noise {noisesig} muK per pixel")
    print(f"  colour limits: T/d +-{t_lim} muK, phi +-{phi_lim}e-4, residual +-3 sigma")


if __name__ == "__main__":
    main()
