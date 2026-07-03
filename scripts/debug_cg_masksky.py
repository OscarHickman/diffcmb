"""Sanity check: does the diagonal-preconditioned CG solver in sample_alm_cg
converge fine on a full-sky problem but stall on a masked-sky problem?

Small lmax/NSIDE synthetic problem so it runs on CPU in well under a minute.
Builds two models sharing the same underlying sky realization: one full-sky
(unmasked_idx = all pixels) and one with an artificial contiguous polar mask
applied post-construction (same trick used in tests/test_cg_matvec.py's
_force_multi_gpu_split for monkey-patching model state before the lazy TF
tensors are built).

Usage: PYTHONPATH=diffcmb .venv/bin/python scripts/debug_cg_masksky.py
"""
import numpy as np

from diffcmb.model import CosmologyAdvancedSampling
from diffcmb.samplers import sample_alm_cg

LMAX = 20
NSIDE = 16
NOISE = 1.0
N_PCG_ITER = 200
MASK_FSKY = 0.75


def build_model(apply_mask):
    model = CosmologyAdvancedSampling(
        LMAX, NSIDE, NOISE, data_mode='synthetic', dtype=None,
    )
    if apply_mask:
        import healpy as hp
        theta, _ = hp.pix2ang(NSIDE, np.arange(model.NPIX))
        # Keep pixels within a polar cap chosen to give ~MASK_FSKY sky fraction.
        cutoff = np.arccos(1 - 2 * MASK_FSKY)
        model.unmasked_idx = np.where(theta < cutoff)[0]
        print(f"  applied mask: f_sky = {len(model.unmasked_idx) / model.NPIX:.3f}")
    model._ensure_tf_tensors()
    return model


def run(label, apply_mask):
    print(f"=== {label} ===")
    model = build_model(apply_mask)
    rng = np.random.default_rng(0)
    lncl_np = np.log(np.maximum(model.prior_cls[2:LMAX], 1e-8))
    _, residual_norms = sample_alm_cg(
        model, lncl_np, rng, n_pcg_iter=N_PCG_ITER, tol=1e-6, verbose_pcg=False,
    )
    r0, rN = residual_norms[0], residual_norms[-1]
    print(f"  iters run: {len(residual_norms) - 1}")
    print(f"  |r| start: {r0:.3e}   |r| end: {rN:.3e}   ratio: {rN / r0:.3e}")
    return residual_norms


if __name__ == "__main__":
    full = run("full sky (f_sky=1.0)", apply_mask=False)
    masked = run("masked sky (f_sky~%.2f)" % MASK_FSKY, apply_mask=True)

    print()
    print("Summary:")
    print(f"  full-sky   residual reduction over {len(full)-1} iters: {full[-1]/full[0]:.3e}")
    print(f"  masked-sky residual reduction over {len(masked)-1} iters: {masked[-1]/masked[0]:.3e}")
