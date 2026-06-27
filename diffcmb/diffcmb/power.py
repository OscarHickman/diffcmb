from typing import List

import numpy as np

try:
    import camb
except Exception:  # keep module import-safe for environments without camb
    camb = None


def call_CAMB_map(_parameters: List[float], _lmax: int) -> np.ndarray:
    """Use CAMB to generate a power spectrum.

    If CAMB is not installed the function raises ImportError.
    """
    if camb is None:
        raise ImportError("camb is required for call_CAMB_map but is not installed")

    if _lmax <= 2551:
        pars = camb.CAMBparams()
        pars.set_cosmology(
            H0=_parameters[0],
            ombh2=_parameters[1],
            omch2=_parameters[2],
            mnu=_parameters[3],
            omk=_parameters[4],
            tau=_parameters[5],
        )
        pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
        pars.set_for_lmax(_lmax, lens_potential_accuracy=0)

        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
        totCL = powers["total"]
        _DL = totCL[:, 0]

        _l = np.arange(len(_DL))
        _CL = []
        for i in range(_lmax):
            if i == 0:
                _CL.append(_DL[i])
            else:
                _CL.append(_DL[i] / (_l[i] * (_l[i] + 1)))

        return np.array(_CL)

    raise ValueError("lmax value is larger than the available data.")
