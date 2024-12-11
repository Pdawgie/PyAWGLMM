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
