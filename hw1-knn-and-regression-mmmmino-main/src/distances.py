import numpy as np 

def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.

    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """
    [m, k] = np.shape(X)
    [n, k] = np.shape(Y)
    d = np.zeros([m, n])
    for row in range(m):
        for col in range(n):
            d[row, col] = np.linalg.norm(X[row] - Y[col])
    return d


def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.


    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """
    [m, k] = np.shape(X)
    [n, k] = np.shape(Y)
    d = np.zeros([m, n])
    for row in range(m):
        for col in range(n):
            d[row, col] = np.linalg.norm(X[row] - Y[col], ord=1)
    return d


