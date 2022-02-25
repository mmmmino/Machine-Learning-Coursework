import numpy as np


def softmax(x, axis):
    """
    Implements a *stabilized* softmax along the correct index
    https://www.deeplearningbook.org/contents/numerical.html

    """
    x = np.atleast_2d(x)
    softmax_ = np.zeros(np.shape(x))
    max_ = np.max(x, axis=axis)
    for i in range(0, x.shape[0]):
        numerator = np.exp(x[i] - max_[i], dtype=np.float32)
        softmax_[i] = numerator / np.sum(numerator)
    return softmax_

