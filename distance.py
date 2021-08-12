import numba
import numpy as np

@numba.njit(nopython=True, fastmath=True)
def euclidean(x, y):
    """Standard euclidean distance.
    ..math::
        D(x, y) = \sqrt{\sum_i (x_i - y_i)^2}
    """
    result = 0.0
    for i in range(x.shape[0]):
        result += (x[i] - y[i]) ** 2
    return np.sqrt(result)