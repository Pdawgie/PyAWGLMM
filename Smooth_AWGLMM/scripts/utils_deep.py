import numpy as np
from scipy.stats import multivariate_normal

def nmi(labels1, labels2):
    """
    Compute the normalized mutual information (NMI) between two label arrays.
    labels1 and labels2 are arrays of equal length with discrete assignments.

    NMI(X;Y) = 2*I(X;Y) / (H(X) + H(Y))
    """
    labels1 = np.asarray(labels1)
    labels2 = np.asarray(labels2)

    # Get unique classes and their frequencies
    # Joint distribution
    unique1 = np.unique(labels1)
    unique2 = np.unique(labels2)

    # Construct joint frequency table
    N = len(labels1)
    contingency = np.zeros((len(unique1), len(unique2)))
    for i, c1 in enumerate(unique1):
        for j, c2 in enumerate(unique2):
            contingency[i, j] = np.sum((labels1 == c1) & (labels2 == c2))

    # Joint probability
    Pxy = contingency / N

    # Marginal probabilities
    Px = Pxy.sum(axis=1)
    Py = Pxy.sum(axis=0)

    # Compute entropies
    def entropy(prob):
        prob = prob[prob > 0]
        return -np.sum(prob * np.log(prob))

    Hx = entropy(Px)
    Hy = entropy(Py)

    # Mutual information I(X;Y)
    # I(X;Y) = sum_x,y P(x,y)*log(P(x,y)/(P(x)*P(y)))
    # Only sum where Pxy > 0
    Pxy_nonzero = Pxy[Pxy > 0]
    # corresponding Px,Py:
    # We need indices from Pxy to match Px,Py
    # Easier way: double loop or vectorized log:
    I = 0.0
    for i in range(len(unique1)):
        for j in range(len(unique2)):
            if Pxy[i, j] > 0:
                I += Pxy[i, j] * np.log(Pxy[i, j] / (Px[i]*Py[j]))

    # NMI
    denominator = (Hx + Hy)
    if denominator == 0:
        return 1.0 if I == 0 else 0.0
    NMI = 2*I/denominator
    return NMI



def mvnpdf(X, mean, cov):
    """
    A Python wrapper for mvnpdf that mimics MATLAB's mvnpdf behavior.

    Parameters
    ----------
    X : ndarray
        Input data matrix of shape (N, D) where N is the number of observations
        and D is the dimension of each observation.
    mean : ndarray
        Mean vector of length D.
    cov : ndarray
        Covariance matrix of shape (D, D).

    Returns
    -------
    pdf_vals : ndarray
        An N-by-1 array of probability density values for each observation in X.
        (Returns a 1D array of length N, which you can treat like N-by-1.)

    Notes
    -----
    - This function ensures the input shapes match what MATLAB's mvnpdf expects.
    - If X is (N,) and mean is (D,), we reshape X to (1, D).
    - If shapes do not match, it raises a ValueError.
    """

    X = np.asarray(X, dtype=float)
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)

    # If X is a single sample (D-dimensional), reshape it to (1, D)
    if X.ndim == 1:
        X = X[np.newaxis, :]

    N, D = X.shape
    if mean.ndim != 1 or mean.shape[0] != D:
        raise ValueError("Mean vector must be of length D.")
    if cov.shape[0] != D or cov.shape[1] != D:
        raise ValueError("Covariance matrix must be D-by-D.")

    # Create the multivariate normal distribution
    rv = multivariate_normal(mean=mean, cov=cov)

    # Compute pdf values for each row of X
    pdf_vals = rv.pdf(X)

    # In MATLAB, mvnpdf returns an N-by-1 vector. Here we return a 1D array of length N.
    # If you strictly need N-by-1, you can reshape: pdf_vals = pdf_vals[:, np.newaxis]
    return pdf_vals

def gsp_compute_theta_bounds(Z, geom_mean=0, is_sorted=0):
    """
    Python equivalent of gsp_compute_theta_bounds.m
    
    Parameters
    ----------
    Z : ndarray (n x n)
        Zero-diagonal pairwise distance matrix between nodes.
    geom_mean : int or bool, optional
        If 0, use arithmetic mean. If 1, use geometric mean. Default: 0.
    is_sorted : int or bool, optional
        If 1, Z is already sorted row-wise (excluding diagonal). Default: 0.

    Returns
    -------
    theta_l : ndarray
        Lower bounds of theta for each sparsity level.
    theta_u : ndarray
        Upper bounds of theta for each sparsity level.
    Z_sorted : ndarray
        The sorted version of Z (excluding diagonals).
    """
    Z = np.asarray(Z, dtype=np.float64)
    n = Z.shape[0]

    # If Z is a matrix
    if not is_sorted:
        # We must sort each row excluding the diagonal
        Z_sorted = np.zeros((n, n-1), dtype=np.float64)
        for i in range(n):
            row_indices = list(range(n))
            row_indices.remove(i)
            row_vals = Z[i, row_indices]
            Z_sorted[i,:] = np.sort(row_vals)
    else:
        Z_sorted = Z

    m, n_minus_1 = Z_sorted.shape
    # B_k = cumsum(Z_sorted, axis=1)
    B_k = np.cumsum(Z_sorted, axis=1)

    # K_mat = repmat((1:n-1), m, 1)
    K_mat = np.tile(np.arange(1, n_minus_1+1), (m,1))

    # Compute upper bounds
    # term inside sqrt: K_mat * Z_sorted.^2 - B_k * Z_sorted
    # Careful with elementwise operations
    # Z_sorted^2:
    Z_sq = Z_sorted**2
    numerator = K_mat * Z_sq
    denominator = B_k * Z_sorted
    inside_sqrt = numerator - denominator

    # We must ensure no negative values inside sqrt:
    # Theoretically it should be positive, but due to numerical issues, clip if needed:
    inside_sqrt = np.maximum(inside_sqrt, 1e-14)

    # 1./sqrt(...)
    values = 1.0 / np.sqrt(inside_sqrt)

    if geom_mean == 0:
        # Arithmetic mean along rows, then mean along rows
        theta_u = np.mean(values, axis=0)
    else:
        # Geometric mean:
        # geometric mean = exp(mean(log(x))) for x>0
        # ensure x>0
        log_vals = np.log(values)
        theta_u = np.exp(np.mean(log_vals, axis=0))

    # theta_l = [theta_u(2:end), 0]
    # In MATLAB indexing, theta_l(k) = theta_u(k+1) for k=1,...,n-2 and theta_l(n-1)=0
    # In Python indexing:
    theta_l = np.zeros_like(theta_u)
    if len(theta_u) > 1:
        theta_l[:-1] = theta_u[1:]
    theta_l[-1] = 0

    return theta_l, theta_u, Z_sorted
