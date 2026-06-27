import numpy as np

try:
    import tensorflow as tf
    import tensorflow_probability as tfp
except Exception:
    tf = None
    tfp = None


def multtensor(_lmax, _lenalm):
    """Return the shape tensor used in psi calculations."""
    if tf is None:
        raise ImportError("tensorflow is required for multtensor")
    _shape = np.zeros([_lmax, _lenalm])
    _count = 0
    for i in range(_lmax):
        for j in np.arange(0, i + 1):
            if j == 0:
                _shape[i][_count] = 1.0
                _count = _count + 1
            else:
                _shape[i][_count] = 2.0
                _count = _count + 1
    return tf.convert_to_tensor(_shape, dtype=np.float64)
