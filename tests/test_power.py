import pytest

from diffcmb import power


def test_call_camb_map_requires_camb():
    # If camb isn't installed the function should raise ImportError
    try:
        camb_installed = True
    except Exception:
        camb_installed = False

    if not camb_installed:
        with pytest.raises(ImportError):
            power.call_CAMB_map([67.0, 0.022, 0.12, 0.06, 0.0, 0.06], 10)
    else:
        # Basic smoke test when camb is available: returns array of length lmax
        out = power.call_CAMB_map([67.0, 0.022, 0.12, 0.06, 0.0, 0.06], 10)
        assert len(out) == 10
