import numpy as np

from diffcmb.alm import noisemapfunc


def test_noisemapfunc_shapes_and_stats():
    np.random.seed(0)
    sample_map = np.zeros(1000)
    var = 2.0

    noisy_map, noise = noisemapfunc(sample_map, var)

    # shapes
    assert noisy_map.shape == sample_map.shape
    assert noise.shape == sample_map.shape

    # noise statistics (std close to var)
    assert abs(np.std(noise) - var) < 0.5
    # noisy_map equals original plus noise
    assert np.allclose(noisy_map, sample_map + noise)
