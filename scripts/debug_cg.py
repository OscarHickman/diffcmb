"""Diagnose PCG convergence failure in sample_alm_cg.

Checks:
  A) Per-iteration ||r||, alpha, pAp/rz for the first 10 CG steps.
  B) Is Ap := alm_grad(p) - alm_grad(0) actually linear in p?
     (Test at p, 2p, -p — linearity of a quadratic gradient.)
  C) Is A symmetric? Check dot(p, Ap) vs dot(Ap_q, q) for random q.
  D) Is A positive definite? Check pAp > 0.
  E) Ratio pAp/rz — determines alpha; should be ~2 for our preconditioner.

Run with:
  python scripts/debug_cg.py [checkpoint_dir]
"""
import sys

import numpy as np

sys.path.insert(0, '/cosma/apps/durham/dc-hick2/diffcmb/diffcmb')

try:
    import tensorflow as tf
except ImportError:
    tf = None

from diffcmb.model import CosmologyAdvancedSampling
from diffcmb.samplers import _build_inv_cl_diag

DATA_DIR = '/cosma8/data/dp004/dc-hick2/Plank'
LMAX = 300
NSIDE = 256
NOISE = 1.0

CHECKPOINT_DIR = sys.argv[1] if len(sys.argv) > 1 else (
    '/cosma/apps/durham/dc-hick2/diffcmb/results/lmax300_nside256_gibbs_real_double'
)

print("=" * 65)
print("DEBUG: PCG convergence diagnostics")
print("=" * 65)

print("Building model...")
model = CosmologyAdvancedSampling(
    _lmax=LMAX, _NSIDE=NSIDE, _noisesig=NOISE, data_mode='real',
    data_dir=DATA_DIR, parameterization='centered', dtype=tf.complex128,
)
model._ensure_tf_tensors()

# Load latest checkpoint for lncl
import glob
import os

ckpt_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, 'chain1_checkpoint_*.npz')))
if ckpt_files:
    ckpt = np.load(ckpt_files[-1])
    lncl_np = ckpt['lncl']
    print(f"Loaded checkpoint: {ckpt_files[-1]}")
    print(f"  lncl range: [{lncl_np.min():.3f}, {lncl_np.max():.3f}]")
else:
    lncl_np = np.log(model.prior_cls[2:LMAX] + 1e-30)
    print("No checkpoint found, using prior cls for lncl")

lmax = model.lmax
n_real = lmax * (lmax + 1) // 2 - 3
n_imag = (lmax - 2) * (lmax - 1) // 2
n_alm = n_real + n_imag
cl_full = np.zeros(lmax)
cl_full[2:] = np.exp(lncl_np)

inv_cl_diag = _build_inv_cl_diag(lmax, cl_full, n_real, n_imag)
mass_sq = model.build_posterior_mass_sqrt(cl_full) ** 2

print(f"n_alm = {n_alm}, n_real = {n_real}, n_imag = {n_imag}")
print(f"inv_cl_diag: min={inv_cl_diag.min():.3e}, max={inv_cl_diag.max():.3e}")
print(f"mass_sq:     min={mass_sq.min():.3e}, max={mass_sq.max():.3e}")

# Build alm_grad
lncl_tf_c = tf.constant(lncl_np, dtype=tf.float64)

if not hasattr(model, "_cg_grad_fn"):
    @tf.function(jit_compile=False)
    def _grad_fn(lncl_tf, alm_tf):
        with tf.GradientTape() as tape:
            tape.watch(alm_tf)
            full_params = tf.concat([lncl_tf, alm_tf], axis=0)
            val = model._psi_tf_raw(full_params)
        return tape.gradient(val, alm_tf)
    model._cg_grad_fn = _grad_fn

def alm_grad(alm_np):
    return model._cg_grad_fn(lncl_tf_c, tf.constant(alm_np, dtype=tf.float64)).numpy()

zeros = np.zeros(n_alm, dtype=np.float64)
print("\nComputing minus_b_data = alm_grad(0)...")
minus_b_data = alm_grad(zeros)
print(f"  ||minus_b_data|| = {np.linalg.norm(minus_b_data):.6e}")
print(f"  min/max: {minus_b_data.min():.3e} / {minus_b_data.max():.3e}")

# ----------------------------------------------------------------
# CHECK B: Is the gradient LINEAR in alm? (should be for quadratic ψ)
# ----------------------------------------------------------------
print("\n--- CHECK B: Gradient linearity ---")
rng = np.random.default_rng(42)
p_test = rng.standard_normal(n_alm) * 0.01  # small test vector

Ap_1x = alm_grad(p_test) - minus_b_data
Ap_2x = alm_grad(2 * p_test) - minus_b_data
Ap_neg = alm_grad(-p_test) - minus_b_data

ratio_2x = np.linalg.norm(Ap_2x) / np.linalg.norm(Ap_1x)
ratio_neg = np.linalg.norm(Ap_neg) / np.linalg.norm(Ap_1x)
err_2x = np.linalg.norm(Ap_2x - 2 * Ap_1x) / np.linalg.norm(Ap_1x)
err_neg = np.linalg.norm(Ap_neg + Ap_1x) / np.linalg.norm(Ap_1x)

print(f"  ||A(2p)|| / ||A(p)|| = {ratio_2x:.6f}  (expect 2.0)")
print(f"  ||A(-p)|| / ||A(p)|| = {ratio_neg:.6f}  (expect 1.0)")
print(f"  ||A(2p) - 2*A(p)|| / ||A(p)|| = {err_2x:.2e}  (expect ~0)")
print(f"  ||A(-p) + A(p)|| / ||A(p)|| = {err_neg:.2e}  (expect ~0)")

# ----------------------------------------------------------------
# CHECK C & D: Symmetry and positive-definiteness
# ----------------------------------------------------------------
print("\n--- CHECK C&D: A symmetry and positive-definiteness ---")
q_test = rng.standard_normal(n_alm) * 0.01
Aq = alm_grad(q_test) - minus_b_data

pAq = float(np.dot(p_test, Aq))
qAp = float(np.dot(q_test, Ap_1x))
pAp = float(np.dot(p_test, Ap_1x))
qAq = float(np.dot(q_test, Aq))

print(f"  dot(p, Aq) = {pAq:.6e}")
print(f"  dot(q, Ap) = {qAp:.6e}")
print(f"  Symmetry error = {abs(pAq - qAp) / (abs(pAq) + 1e-30):.2e}  (expect ~0)")
print(f"  dot(p, Ap) = {pAp:.6e}  (expect > 0 if A is PD)")
print(f"  dot(q, Aq) = {qAq:.6e}  (expect > 0 if A is PD)")

# ----------------------------------------------------------------
# CHECK E: pAp/rz ratio (determines alpha; should be ~2)
# ----------------------------------------------------------------
print("\n--- CHECK E: pAp/rz ratio for CG preconditioner ---")
# Simulate first CG step
omega1 = rng.standard_normal(n_alm)
noise_prior = np.sqrt(inv_cl_diag) * omega1

Ninv_np = np.concatenate([model.Ninv_parts[i].numpy() for i in range(len(model.sph_parts))])
omega2 = rng.standard_normal(len(Ninv_np))
v_pix = np.sqrt(np.maximum(Ninv_np, 0.0)) * omega2

if not hasattr(model, "_cg_jt_v_fn"):
    _part_sizes = [int(sph_p.shape[0]) for sph_p in model.sph_parts]
    _n_real_cap = n_real
    _lmax_cap = lmax
    from diffcmb.alm_utils import splittosingularalm_tf
    from diffcmb.model import matvec_on_device

    @tf.function(jit_compile=False)
    def _jt_v_fn(v_concat, alm_zero):
        v_parts = tf.split(v_concat, _part_sizes)
        with tf.GradientTape() as tape:
            tape.watch(alm_zero)
            _rp = alm_zero[:_n_real_cap]
            _ip = alm_zero[_n_real_cap:]
            _a = splittosingularalm_tf(_rp, _ip, _lmax_cap)
            _a_c = model.alm_weights * tf.cast(_a, model.dtype)
            inner = tf.zeros((), dtype=tf.float64)
            for i, sph_p in enumerate(model.sph_parts):
                _Ya = 2.0 * tf.math.real(matvec_on_device(sph_p, _a_c))
                inner = inner + tf.reduce_sum(tf.cast(_Ya, tf.float64) * v_parts[i])
        return tape.gradient(inner, alm_zero)
    model._cg_jt_v_fn = _jt_v_fn

noise_pix = model._cg_jt_v_fn(
    tf.constant(v_pix, dtype=tf.float64),
    tf.zeros(n_alm, dtype=tf.float64),
).numpy()
noise_target = noise_prior + noise_pix

r = minus_b_data - noise_target
z = r / mass_sq
p = -z.copy()
rz = float(np.dot(r, z))
Ap = alm_grad(p) - minus_b_data
pAp = float(np.dot(p, Ap))
alpha = rz / pAp if abs(pAp) > 1e-300 else float('nan')

print(f"  ||r_0|| = {np.linalg.norm(r):.6e}")
print(f"  rz      = {rz:.6e}")
print(f"  pAp     = {pAp:.6e}")
print(f"  pAp/rz  = {pAp/rz:.6f}  (expect ~2 for optimal preconditioner)")
print(f"  alpha   = {alpha:.6e}")

# Run 10 CG steps with full logging
print("\n--- CG residual progression (10 steps) ---")
print("  iter | ||r||          | alpha          | pAp/rz")
print(f"  {0:4d} | {np.linalg.norm(r):.6e}   |                |")

r_cur = r.copy()
p_cur = p.copy()
rz_cur = rz

for k in range(10):
    Ap_k = alm_grad(p_cur) - minus_b_data
    pAp_k = float(np.dot(p_cur, Ap_k))
    if abs(pAp_k) < 1e-300:
        print("  pAp too small, stopping")
        break
    alpha_k = rz_cur / pAp_k
    r_new = r_cur + alpha_k * Ap_k
    z_new = r_new / mass_sq
    rz_new = float(np.dot(r_new, z_new))
    beta_k = rz_new / rz_cur
    p_new = -z_new + beta_k * p_cur
    ratio = pAp_k / rz_cur
    print(f"  {k+1:4d} | {np.linalg.norm(r_new):.6e}   | {alpha_k:.6e}   | {ratio:.6f}")
    r_cur = r_new
    p_cur = p_new
    rz_cur = rz_new
    if rz_new < 1e-20:
        print("  Converged!")
        break

print("\nDone.")
