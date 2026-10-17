"""scripts/realdata_summary.py: multi-chain R-hat, the one-data-set refusal and the Planck comparison."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import realdata_summary as rs  # noqa: E402

LMAX = 12


def _chain(path, rng, data, cl_tt, cl_pp_fid, shift=0.0, n=400, amp=3000.0):
    n_l = LMAX - 2
    ln_tt = np.log(cl_tt[2:]) + shift + 0.1 * rng.standard_normal((n, n_l))
    ln_pp = np.log(cl_pp_fid[2:] * 0.01) + 0.2 * rng.standard_normal((n, n_l))
    np.savez(path, lmax=LMAX, alm_samples=np.concatenate([ln_tt, rng.standard_normal((n, 5))], 1),
             cl_phiphi_samples=ln_pp, cl_phiphi_fid=cl_pp_fid, phi_amplitude=amp,
             data_masked=data, data_map="smica.fits")


@pytest.fixture
def spectra():
    ell = np.arange(LMAX, dtype=float)
    cl_tt = np.r_[0, 0, 2 * np.pi * 1000.0 / (ell[2:] * (ell[2:] + 1))]
    return cl_tt, np.r_[0, 0, 3000.0 * 1e-7 / ell[2:] ** 4]


def test_chains_of_one_target_have_unit_rhat_and_a_shifted_chain_does_not(spectra):
    rng = np.random.default_rng(0)
    same = [rng.standard_normal((500, 3)) for _ in range(4)]
    assert np.all(rs.multichain_rhat(same) < 1.02)
    off = same[:3] + [same[3] + 1.0]
    assert np.all(rs.multichain_rhat(off) > 1.1)


def test_summary_reads_planck_and_the_phi_amplitude(tmp_path, spectra):
    cl_tt, cl_pp = spectra
    rng = np.random.default_rng(1)
    data = rng.standard_normal(30)
    for i in range(3):
        _chain(tmp_path / f"data_chain_r{i:03d}.npz", rng, data, cl_tt, cl_pp)
    files, ds = rs.load_chains(str(tmp_path))
    planck_dl = np.full(LMAX - 2, 1000.0)  # D_l of cl_tt
    s = rs.summarise(ds, planck_dl)
    np.testing.assert_allclose(s["C_l^TT/planck_over_median"], 1.0, atol=0.02)
    # posterior 0.01 x the prior centre, i.e. 30 x the physical fiducial at A_phi = 3000
    np.testing.assert_allclose(s["C_L^phiphi/over_prior_q50"], 0.01, rtol=0.05)
    np.testing.assert_allclose(s["C_L^phiphi/over_physical_q50"], 30.0, rtol=0.05)
    assert np.all(s["C_l^TT/rhat"] < 1.05)


def test_chains_that_saw_different_data_are_refused(tmp_path, spectra):
    rng = np.random.default_rng(2)
    _chain(tmp_path / "data_chain_r000.npz", rng, np.zeros(30), *spectra)
    _chain(tmp_path / "data_chain_r001.npz", rng, np.ones(30), *spectra)
    with pytest.raises(SystemExit, match="different data"):
        rs.load_chains(str(tmp_path))


def test_planck_table_is_read_by_multipole(tmp_path):
    t = np.column_stack([np.arange(2, 40), 10.0 * np.arange(2, 40), np.ones(38), np.ones(38)])
    f = tmp_path / "tt.txt"
    np.savetxt(f, t)
    np.testing.assert_array_equal(rs.read_planck_dl(str(f), LMAX), 10.0 * np.arange(2, LMAX))
    with pytest.raises(SystemExit):
        rs.read_planck_dl(str(f), 64)
