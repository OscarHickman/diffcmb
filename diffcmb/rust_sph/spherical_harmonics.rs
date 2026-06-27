use ndarray::Array2;
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1};
use pyo3::{prelude::*, types::PyModule};
use rayon::prelude::*;
use std::f64::consts::PI;

// Precomputed recurrence coefficients for the Holmes & Featherstone (2002)
// normalized associated Legendre polynomial recurrence. All coefficients
// depend only on (l, m), so they are computed once per lmax and shared
// across all pixels during the parallel phase.
//
// The normalized function is:
//   q_lm = sqrt((2l+1)/(4π) * (l-m)!/(l+m)!) * P_lm(cos θ)
//
// so that Y_lm(θ, φ) = q_lm * e^{imφ}, matching the scipy.special.sph_harm
// Condon-Shortley convention (the (-1)^m phase is absorbed into the
// diagonal recurrence via the leading minus sign in `diag`).
struct Coeffs {
    lmax: usize,
    len_alm: usize,
    // diag[m]  = -sqrt((2m+1)/(2m))      for the q_mm <- q_{m-1,m-1} step, m >= 1
    diag: Vec<f64>,
    // offd[m]  = sqrt(2m+3)               for the q_{m+1,m} <- q_mm step, m+1 < lmax
    offd: Vec<f64>,
    // alpha[mo_idx(l,m)] and beta[mo_idx(l,m)] for the general l >= m+2 step
    alpha: Vec<f64>,
    beta: Vec<f64>,
}

impl Coeffs {
    fn new(lmax: usize) -> Self {
        let len_alm = lmax * (lmax + 1) / 2;
        let mut diag = vec![0.0f64; lmax];
        let mut offd = vec![0.0f64; lmax];
        let mut alpha = vec![0.0f64; len_alm];
        let mut beta = vec![0.0f64; len_alm];

        for m in 1..lmax {
            let mf = m as f64;
            diag[m] = -((2.0 * mf + 1.0) / (2.0 * mf)).sqrt();
        }
        for m in 0..lmax {
            if m + 1 < lmax {
                offd[m] = (2.0 * m as f64 + 3.0).sqrt();
            }
        }
        for l in 2..lmax {
            let lf = l as f64;
            let l2 = lf * lf;
            // General recurrence applies for m in 0..=l-2
            for m in 0..(l - 1) {
                let mf = m as f64;
                let m2 = mf * mf;
                let idx = l * (l + 1) / 2 + m;
                alpha[idx] = ((4.0 * l2 - 1.0) / (l2 - m2)).sqrt();
                let num = (2.0 * lf + 1.0) * (lf - 1.0 - mf) * (lf - 1.0 + mf);
                let den = (2.0 * lf - 3.0) * (l2 - m2);
                beta[idx] = (num / den).sqrt();
            }
        }

        Self { lmax, len_alm, diag, offd, alpha, beta }
    }
}

// Compute all Y_lm(θ, φ) for l < lmax, m <= l, returned in author (mo)
// ordering: [(0,0), (1,0), (1,1), (2,0), (2,1), (2,2), ...].
fn pixel_sph(theta: f64, phi: f64, coeffs: &Coeffs) -> Vec<Complex64> {
    let lmax = coeffs.lmax;
    let len_alm = coeffs.len_alm;
    let x = theta.cos();
    let s = theta.sin();
    let q00 = 1.0 / (4.0 * PI).sqrt();

    // Phase powers: phase[m] = e^{i m φ}, built by repeated multiplication
    // to avoid lmax calls to cos/sin.
    let ephi = Complex64::new(phi.cos(), phi.sin());
    let mut phase = Vec::with_capacity(lmax);
    let mut p = Complex64::new(1.0, 0.0);
    for _ in 0..lmax {
        phase.push(p);
        p *= ephi;
    }

    // Build the normalized ALF values q[mo_idx(l,m)] column by column (by m).
    let mut q = vec![0.0f64; len_alm];
    let mut qmm_prev = q00; // tracks q_{m-1, m-1} for the diagonal step

    for m in 0..lmax {
        // Diagonal step: q_mm
        let qmm = if m == 0 {
            q00
        } else {
            coeffs.diag[m] * s * qmm_prev
        };
        q[m * (m + 1) / 2 + m] = qmm;
        qmm_prev = qmm;

        // First off-diagonal: q_{m+1, m}
        if m + 1 < lmax {
            q[(m + 1) * (m + 2) / 2 + m] = coeffs.offd[m] * x * qmm;
        }

        // General recurrence: l = m+2, ..., lmax-1
        for l in (m + 2)..lmax {
            let idx = l * (l + 1) / 2 + m;
            let idx_m1 = (l - 1) * l / 2 + m;
            let idx_m2 = (l - 2) * (l - 1) / 2 + m;
            q[idx] = coeffs.alpha[idx] * x * q[idx_m1] - coeffs.beta[idx] * q[idx_m2];
        }
    }

    // Assemble complex harmonics Y_lm = q_lm * e^{imφ}.
    // The L=0 term is forced to zero imaginary part to match the convention
    // used in the Python codebase (see _ensure_tf_tensors in model.py).
    let mut result = vec![Complex64::new(0.0, 0.0); len_alm];
    let mut col = 0usize;
    for l in 0..lmax {
        for m in 0..=l {
            let y = Complex64::new(q[col], 0.0) * phase[m];
            result[col] = if l == 0 {
                Complex64::new(y.re, 0.0)
            } else {
                y
            };
            col += 1;
        }
    }

    result
}

/// Compute the full spherical harmonic matrix for all HEALPix pixels.
///
/// Returns a (NPIX, len_alm) complex128 numpy array where
/// ``len_alm = lmax*(lmax+1)//2`` and columns are in author (mo) ordering:
/// (L=0,m=0), (L=1,m=0), (L=1,m=1), (L=2,m=0), ...
///
/// Pixels are processed in parallel using Rayon.
///
/// Args:
///     thetas: 1-D float64 array of co-latitude angles (radians), length NPIX.
///     phis:   1-D float64 array of azimuthal angles (radians), length NPIX.
///     lmax:   Maximum ell (exclusive); output has columns for l = 0..lmax-1.
#[pyfunction]
fn compute_sph<'py>(
    py: Python<'py>,
    thetas: PyReadonlyArray1<'py, f64>,
    phis: PyReadonlyArray1<'py, f64>,
    lmax: usize,
) -> Bound<'py, PyArray2<Complex64>> {
    let thetas_arr = thetas.as_array();
    let phis_arr = phis.as_array();
    let npix = thetas_arr.len();
    let len_alm = lmax * (lmax + 1) / 2;

    // Copy pixel angles into owned Vecs so the Rayon threads do not hold
    // references into the Python-managed buffer across the GIL release.
    let thetas_vec: Vec<f64> = thetas_arr.iter().copied().collect();
    let phis_vec: Vec<f64> = phis_arr.iter().copied().collect();

    let coeffs = Coeffs::new(lmax);

    // Release the GIL while computing — pure Rust arithmetic, no Python calls.
    let flat: Vec<Complex64> = py.allow_threads(|| {
        (0..npix)
            .into_par_iter()
            .flat_map(|i| pixel_sph(thetas_vec[i], phis_vec[i], &coeffs))
            .collect()
    });

    Array2::from_shape_vec((npix, len_alm), flat)
        .expect("shape mismatch — bug in pixel_sph length")
        .into_pyarray_bound(py)
}

#[pymodule]
fn cmb_sph(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_sph, m)?)?;
    Ok(())
}
