import time
import numpy as np  
import networkx as nx 
import matplotlib.pyplot as plt 
import scipy.sparse as sp
from scipy.spatial.distance import squareform  
from scipy.sparse import csr_matrix  
from scipy.sparse.linalg import svds  
from numpy.linalg import eigvals  
import numpy as np
from scipy.sparse import csr_matrix, find, isspmatrix
from math import sqrt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    normalized_mutual_info_score,
)  




def visualize_glmm(Ls, gamma_hats):
    k = gamma_hats.shape[1]
    n = Ls.shape[0]
    
    G = nx.Graph()
    
    for i in range(n):
        G.add_node(i)
    
    for l in range(k):
        L = Ls[:, :, l]
        for i in range(n):
            for j in range(i+1, n):
                if L[i, j] != 0:
                    G.add_edge(i, j, weight=L[i, j])
    
    colors = []
    for i in range(n):
        cluster = np.argmax(gamma_hats[i])
        colors.append(cluster)
    
    pos = nx.spring_layout(G)  
    
    nx.draw_networkx_nodes(G, pos, node_size=700, cmap=plt.cm.RdYlBu, node_color=colors)
    
    edges = G.edges(data=True)
    weights = [edata['weight'] for _, _, edata in edges]
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights)
    
    nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')
    
    plt.title('Graph Laplacian Mixture Model (GLMM) Visualization')
    plt.show()
    

# def graph_learning_perf_eval(L_0, L):
#     """
#     Evaluates the performance of a learned graph by comparing it with the ground truth Laplacian matrix.

#     Parameters:
#     -----------
#     L_0 : np.ndarray
#         Ground truth Laplacian matrix of shape (n, n).
    
#     L : np.ndarray
#         Learned Laplacian matrix of shape (n, n).

#     Returns:
#     --------
#     precision : float
#         Precision score of the learned graph.
    
#     recall : float
#         Recall score of the learned graph.
    
#     f : float
#         F1 score of the learned graph.
    
#     NMI_score : float
#         Normalized Mutual Information score of the learned graph.
    
#     num_of_edges : int
#         Number of edges in the learned graph.
#     """
#     # Edges in the ground truth graph
#     L_0tmp = L_0 - np.diag(np.diag(L_0))
#     L_0tmp = (L_0tmp + L_0tmp.T) / 2  
#     edges_groundtruth = squareform(L_0tmp) != 0

#     # Edges in the learned graph
#     Ltmp = L - np.diag(np.diag(L))
#     Ltmp = (Ltmp + Ltmp.T) / 2  
#     edges_learned = squareform(Ltmp) != 0

#     # Compute precision, recall, F1-score
#     precision = precision_score(edges_groundtruth.astype(int), edges_learned.astype(int), zero_division=0)
#     recall = recall_score(edges_groundtruth.astype(int), edges_learned.astype(int), zero_division=0)
#     f = f1_score(edges_groundtruth.astype(int), edges_learned.astype(int), zero_division=0)

#     # NMI
#     NMI_score = normalized_mutual_info_score(edges_learned.astype(int), edges_groundtruth.astype(int))
#     if np.isnan(NMI_score):
#         NMI_score = 0

#     # Number of edges in the learned graph
#     num_of_edges = np.sum(edges_learned)

#     return precision, recall, f, NMI_score, num_of_edges

def graph_learning_perf_eval(L_0, L):
    import numpy as np
    from scipy.spatial.distance import squareform
    from sklearn.metrics import normalized_mutual_info_score

    # Edges in the ground truth graph
    L_0tmp = L_0 - np.diag(np.diag(L_0))
    edges_groundtruth = squareform(L_0tmp) != 0

    # Edges in the learned graph
    Ltmp = L - np.diag(np.diag(L))
    edges_learned = squareform(Ltmp) != 0

    # Compute True Positives, False Positives, True Negatives, False Negatives
    TP = np.sum((edges_learned == 1) & (edges_groundtruth == 1))
    FP = np.sum((edges_learned == 1) & (edges_groundtruth == 0))
    FN = np.sum((edges_learned == 0) & (edges_groundtruth == 1))
    TN = np.sum((edges_learned == 0) & (edges_groundtruth == 0))

    # Compute precision and recall from first principles
    if TP + FP > 0:
        precision = TP / (TP + FP)
    else:
        precision = 0.0

    if TP + FN > 0:
        recall = TP / (TP + FN)
    else:
        recall = 0.0

    # Compute F1 score
    if precision + recall > 0:
        f = 2 * precision * recall / (precision + recall)
    else:
        f = 0.0

    # NMI
    NMI_score = normalized_mutual_info_score(edges_learned.astype(int), edges_groundtruth.astype(int))
    if np.isnan(NMI_score):
        NMI_score = 0.0

    # Number of edges in the learned graph
    num_of_edges = np.sum(edges_learned)

    return precision, recall, f, NMI_score, num_of_edges


def identify_and_compare(Ls, Lap, gamma_hats, gamma_cut, k):
    """
    Identifies and compares clusters based on given Laplacian matrices and cluster assignments.

    Parameters:
    -----------
    Ls : np.ndarray
        Estimated Laplacian matrices of shape (n, n, k).
    
    Lap : np.ndarray
        Ground truth Laplacian matrices of shape (n, n, k).
    
    gamma_hats : np.ndarray
        Estimated cluster assignments of shape (m, k).
    
    gamma_cut : np.ndarray
        Ground truth cluster assignments of shape (m, k).
    
    k : int
        Number of clusters.

    Returns:
    --------
    identify : np.ndarray
        Indices of the identified clusters of shape (k,).
    
    precision : np.ndarray
        Precision scores for each cluster of shape (k,).
    
    recall : np.ndarray
        Recall scores for each cluster of shape (k,).
    
    f : np.ndarray
        F1 scores for each cluster of shape (k,).
    
    cl_errors : np.ndarray
        Clustering errors for each cluster of shape (k,).
    
    NMI_score : np.ndarray
        Normalized Mutual Information scores for each cluster of shape (k,).
    
    num_of_edges : np.ndarray
        Number of edges for each cluster of shape (k,).
    """
    identify = np.zeros(k, dtype=int)
    cl_err = np.inf * np.ones(k)
    precision = np.zeros(k)
    recall = np.zeros(k)
    f = np.zeros(k)
    NMI_score = np.zeros(k)
    num_of_edges = np.zeros(k)

    for i in range(k):
        W = np.diag(np.diag(Ls[:, :, i])) - Ls[:, :, i]
        W[W < 0.001] = 0
        Ls[:, :, i] = np.diag(np.sum(W, axis=1)) - W
        for j in range(k):
            er = np.linalg.norm(gamma_hats[:, i] - gamma_cut[:, j])
            if cl_err[i] > er:
                cl_err[i] = er
                identify[i] = j

    for i in range(k):
        idx = identify[i]
        precision[i], recall[i], f[i], NMI_score[i], num_of_edges[i] = graph_learning_perf_eval(Lap[:, :, idx], Ls[:, :, i])

    # Compute clustering errors
    cl_errors = np.array([np.linalg.norm(gamma_hats[:, i] - gamma_cut[:, identify[i]])**2 for i in range(k)])

    return identify, precision, recall, f, cl_errors, NMI_score, num_of_edges



def generate_connected_graph(n, p, zero_thresh):
    """
    Generates a connected Erdos-Renyi graph and returns its Laplacian matrix.

    Parameters:
    -----------
    n : int
        Number of nodes in the graph.
    
    p : float
        Probability for edge creation in the Erdos-Renyi graph.
    
    zero_thresh : float
        Threshold for the second smallest eigenvalue of the Laplacian matrix
        to ensure graph connectivity.

    Returns:
    --------
    np.ndarray
        Laplacian matrix of the generated connected graph.
    """
    while True:
        g = nx.erdos_renyi_graph(n, p)
        L = nx.laplacian_matrix(g).toarray()
        eigs = np.sort(eigvals(L))
        if eigs[1] > zero_thresh:
            return L
        
def normest(S):
    """
    Estimate the 2-norm (largest singular value) of a sparse matrix S.
    """
    u, s, vt = svds(S, k=1)
    return s[0]


def lin_map(X, lims_out, lims_in=None):
    """
    Map linearly from a given range to another.

    Parameters:
    X : array-like
        Input array.
    lims_out : list or tuple
        Output limits [c, d].
    lims_in : list or tuple, optional
        Input limits [a, b]. If not specified, the minimum and maximum values of X are used.

    Returns:
    Y : array-like
        Linearly mapped output array.
    """
    X = np.asarray(X)
    
    if lims_in is None:
        lims_in = [np.min(X), np.max(X)]
    
    a, b = lims_in
    c, d = lims_out
    
    Y = (X - a) * ((d - c) / (b - a)) + c
    return Y



def squareform_sp(w):
    """
    Sparse counterpart of MATLAB's squareform for sparse vectors or matrices.

    Parameters
    ----------
    w : array-like or sparse matrix
        If w is a vector of length n*(n-1)/2, it returns an n-by-n symmetric
        sparse matrix with zero diagonal.
        If w is an n-by-n sparse matrix with zero diagonal, returns a sparse
        vector of length n*(n-1)/2 representing the upper-triangular part.

    Returns
    -------
    w_out : sparse matrix or sparse vector
        If input was vector, output is an n-by-n sparse matrix.
        If input was matrix, output is a sparse vector.

    Notes
    -----
    This is a sparse adaptation of MATLAB's squareform function, with zero-based
    indexing adjustments for Python.
    """

    # Convert input to a known sparse type if not already
    if not isspmatrix(w) and isinstance(w, np.ndarray):
        density = np.count_nonzero(w)/w.size if w.size > 0 else 0
        if density > 1/10:
            # If too dense, fallback to dense squareform
            from scipy.spatial.distance import squareform
            return squareform(w)
        else:
            w = csr_matrix(w)
    elif not isspmatrix(w):
        # Convert lists or other array-likes to sparse
        arr = np.array(w)
        if arr.ndim == 1:
            w = csr_matrix(arr.reshape(-1, 1))
        else:
            w = csr_matrix(arr)

    # Distinguish between vector and matrix based on shape
    if w.ndim == 2 and w.shape[1] != 1:
        # Matrix -> Vector
        m, n = w.shape
        if m != n:
            raise ValueError('Matrix must be square!')
        diag_vals = w.diagonal()
        if not np.allclose(diag_vals, 0):
            raise ValueError('Matrix diagonal must be zero!')

        ind_i, ind_j, s = find(w)
        # Keep only upper triangular entries
        mask_upper = ind_i < ind_j
        ind_i = ind_i[mask_upper]
        ind_j = ind_j[mask_upper]
        s = s[mask_upper]

        # Convert (i,j) to condensed index k using zero-based formula:
        # k = i*(n - 1) - (i*(i-1))//2 + (j - i - 1)
        new_ind = ind_i*(n-1) - (ind_i*(ind_i-1))//2 + (ind_j - ind_i - 1)
        length_vec = n*(n-1)//2
        if new_ind.max() >= length_vec or new_ind.min() < 0:
            raise ValueError("Indexing error: computed condensed index out of range.")

        # Create sparse vector
        w_out = csr_matrix((s, (new_ind, np.zeros_like(new_ind))), shape=(length_vec, 1))

    else:
        # Vector -> Matrix
        # w is a vector (length_vec x 1)
        if w.shape[1] != 1:
            raise ValueError("Vector input should be a single column sparse vector.")
        l = w.shape[0]
        n = int(round((1 + sqrt(1+8*l))/2))
        if l != n*(n-1)//2:
            raise ValueError('Bad vector size!')

        ind_vec, _, s = find(w)
        num_nz = len(ind_vec)
        ind_i = np.zeros(num_nz, dtype=int)
        ind_j = np.zeros(num_nz, dtype=int)

        # Inverse mapping: given k in [0, ..., l-1], find (i,j)
        # We use the loop logic given in the original MATLAB code
        curr_row = 1
        offset = 0
        length_line = n - 1
        for idx in range(num_nz):
            ind_val = ind_vec[idx] + 1  # Convert to 1-based index
            while ind_val > (length_line + offset):
                offset += length_line
                length_line -= 1
                curr_row += 1
            # Now curr_row indicates the row i (1-based)
            # The corresponding column j is ind_val - offset + (n - length_line)
            # Convert both to zero-based indexing:
            i_zero = curr_row - 1
            j_zero = ind_val - offset + (n - length_line) - 1
            ind_i[idx] = i_zero
            ind_j[idx] = j_zero

        # Construct the symmetric matrix
        row_inds = np.concatenate((ind_i, ind_j))
        col_inds = np.concatenate((ind_j, ind_i))
        data = np.concatenate((s, s))
        w_out = csr_matrix((data, (row_inds, col_inds)), shape=(n, n))

    return w_out


def sum_squareform(n, mask=None):
    """
    Creates sparse matrices S and St so that S*w = sum(W), w = squareform(W).
    """
    mask_given = mask is not None

    if mask_given:
        mask = np.array(mask).ravel()
        expected_size = n*(n-1)//2
        if len(mask) != expected_size:
            raise ValueError('mask size must be n*(n-1)/2')
        ind_vec = np.flatnonzero(mask)
        ncols = len(ind_vec)
        I = np.zeros(ncols, dtype=int)
        J = np.zeros(ncols, dtype=int)

        curr_row = 1
        offset = 0
        length_line = n - 1
        for ii in range(ncols):
            ind_val = ind_vec[ii] + 1
            while ind_val > (length_line + offset):
                offset += length_line
                length_line -= 1
                curr_row += 1
            I[ii] = curr_row
            J[ii] = ind_val - offset + (n-length_line)
        I -= 1
        J -= 1
    else:
        ncols = (n-1)*n//2
        I = np.zeros(ncols, dtype=int)
        J = np.zeros(ncols, dtype=int)

        # Replicate the exact MATLAB indexing
        k = 0
        for i in range(2, n+1):
            line_len = (n - i + 1)
            I[k:k+line_len] = np.arange(i-1, n)  # i:n in 1-based is i-1:n-1 in 0-based
            k += line_len

        k = 0
        for i in range(2, n+1):
            line_len = (n - i + 1)
            # J(k : k+(n-i)) = i-1 repeated
            # In MATLAB indexing, i-1 is scalar. Assigning a vector slice to scalar repeats it.
            J[k:k+line_len] = (i-2)  # i-1 in 1-based is i-2 in 0-based indexing
            k += line_len

    # Now ensure that I < J for consistency with upper triangular indexing
    # squareform expects edges with i < j
    swap_mask = I > J
    I[swap_mask], J[swap_mask] = J[swap_mask], I[swap_mask]

    # Build St and S
    rows = np.concatenate((np.arange(ncols), np.arange(ncols)))
    cols = np.concatenate((I, J))
    data = np.ones(2*ncols, dtype=float)
    St = csr_matrix((data, (rows, cols)), shape=(ncols, n))
    S = St.transpose()

    return S, St



# def prox_sum_log(x, gamma, param=None):
#     """
#     Proximal operator of the log-barrier function - sum(log(x)).

#     This function solves:
#         sol = argmin_z (0.5 * ||x - z||_2^2 - gamma * sum(log(z)))

#     Parameters
#     ----------
#     x : ndarray
#         Input signal (vector or matrix).
#     gamma : float
#         Regularization parameter (gamma >= 0).
#     param : dict, optional
#         Parameter dictionary with fields:
#         - verbose: int, default 1
#             0: no output
#             1: print -sum(log(x))
#             2: additionally report number of negative elements in x

#     Returns
#     -------
#     sol : ndarray
#         The solution.
#     info : dict
#         Dictionary containing:
#         - 'algo': str, name of the algorithm
#         - 'iter': int, number of iterations (here 0 since closed-form)
#         - 'time': float, execution time
#         - 'final_eval': float, final evaluation of the chosen measure
#         - 'crit': str, stopping criterion (here '--')

#     Notes
#     -----
#     The formula for the solution is:
#         sol = (x + sqrt(x^2 + 4*gamma)) / 2.

#     The final_eval is computed as -gamma * sum(log(x)) according to the
#     original MATLAB code, even though it might make more sense to evaluate
#     the objective at sol. We keep this choice to remain consistent with the
#     original code.

#     See also:
#     prox_l1, prox_l2, prox_tv, prox_sum_log_norm2
#     """
#     if param is None:
#         param = {}
#     verbose = param.get('verbose', 1)

#     # Start timing
#     t_start = time.time()

#     # Check gamma
#     if gamma < 0:
#         raise ValueError("Gamma cannot be negative.")
#     elif gamma == 0:
#         # If gamma=0, solution is x and final_eval=0
#         sol = x
#         info = {
#             'algo': 'prox_sum_log',
#             'iter': 0,
#             'final_eval': 0,
#             'crit': '--',
#             'time': time.time() - t_start
#         }
#         return sol, info

#     # Ensure x is a numpy array
#     x = np.asarray(x, dtype=float)

#     # Compute the solution
#     sol = (x + np.sqrt(x**2 + 4*gamma)) / 2.0

#     # Prepare info dictionary
#     info = {
#         'algo': 'prox_sum_log',
#         'iter': 0,
#         'final_eval': -gamma * np.sum(np.log(x.ravel())),  # from original code
#         'crit': '--',
#         'time': time.time() - t_start
#     }

#     # Logging
#     if verbose >= 1:
#         val_per_gamma = info['final_eval'] / gamma  # = -sum(log(x))
#         msg = f"  prox_sum_log: - sum(log(x)) = {val_per_gamma:e}"
#         if verbose > 1:
#             # count negative elements
#             n_neg = np.count_nonzero(x <= 0)
#             if n_neg > 0:
#                 msg += f" ({n_neg} negative elements, log not defined, check stability)"
#         print(msg)

#     return sol, info

def prox_sum_log(x, gamma, param=None):
    """
    Proximal operator of the log-barrier function - sum(log(x)).

    This function solves:
        sol = argmin_z (0.5 * ||x - z||_2^2 - gamma * sum(log(z)))

    Parameters
    ----------
    x : ndarray
        Input signal (vector or matrix).
    gamma : float
        Regularization parameter (gamma >= 0).
    param : dict, optional
        Parameter dictionary with fields:
        - verbose: int, default 1
            0: no output
            1: print -sum(log(x))
            2: additionally report number of negative elements in x

    Returns
    -------
    sol : ndarray
        The solution.
    info : dict
        Dictionary containing:
        - 'algo': str, name of the algorithm
        - 'iter': int, number of iterations (here 0 since closed-form)
        - 'time': float, execution time
        - 'final_eval': float, final evaluation of the chosen measure
        - 'crit': str, stopping criterion (here '--')

    Notes
    -----
    If x contains non-positive values, the log is not defined. We will print
    a warning and set final_eval to inf in that case. This prevents NaN issues.

    The formula for the solution is:
        sol = (x + sqrt(x^2 + 4*gamma)) / 2.

    We keep consistency with the original MATLAB code by using -gamma*sum(log(x))
    as final_eval when possible.
    """
    if param is None:
        param = {}
    verbose = param.get('verbose', 1)

    # Start timing
    t_start = time.time()

    # Check gamma
    if gamma < 0:
        raise ValueError("Gamma cannot be negative.")
    elif gamma == 0:
        # If gamma=0, solution is x and final_eval=0
        sol = x
        info = {
            'algo': 'prox_sum_log',
            'iter': 0,
            'final_eval': 0,
            'crit': '--',
            'time': time.time() - t_start
        }
        return sol, info

    # Ensure x is a numpy array
    x = np.asarray(x, dtype=float)

    # Compute the solution
    sol = (x + np.sqrt(x**2 + 4*gamma)) / 2.0

    # Check for non-positive values in x
    if np.any(x <= 0):
        final_eval = np.inf
    else:
        final_eval = -gamma * np.sum(np.log(x.ravel()))

    # Prepare info dictionary
    info = {
        'algo': 'prox_sum_log',
        'iter': 0,
        'final_eval': final_eval,
        'crit': '--',
        'time': time.time() - t_start
    }

    # Logging
    if verbose >= 1:
        if np.isinf(final_eval):
            msg = "  prox_sum_log: x contains non-positive values; - sum(log(x)) = inf"
        else:
            val_per_gamma = final_eval / gamma  # = -sum(log(x))
            msg = f"  prox_sum_log: - sum(log(x)) = {val_per_gamma:e}"
        if verbose > 1:
            # count negative elements
            n_neg = np.count_nonzero(x <= 0)
            if n_neg > 0:
                msg += f" ({n_neg} negative or zero elements, log not defined, check stability)"
        print(msg)

    return sol, info



def gsp_distanz(X, Y=None, P=None):
    """
    gsp_distanz calculates the distances between all vectors in X and Y.

    Parameters
    ----------
    X : ndarray
        Matrix with column vectors (shape: d x n, where d is dimension and n is number of vectors).
    Y : ndarray, optional
        Matrix with column vectors (shape: d x m). Default is X.
    P : ndarray, optional
        Distance matrix (d x d). If given, computes distance under metric defined by P.

    Returns
    -------
    D : ndarray
        Distance matrix of size (n x m), where D[i,j] = distance between X[:,i] and Y[:,j].

    Notes
    -----
    This code computes:
        D = sqrt((X - Y)^T * P * (X - Y))

    If P is not provided, it assumes the standard Euclidean metric:
        D[i,j] = ||X[:,i] - Y[:,j]||_2

    If Y is not provided, Y = X and the diagonal of D is set to zero.
    """

    if X is None:
        raise ValueError("Not enough input parameters: X must be provided")

    # Default Y = X if not provided
    if Y is None:
        Y = X

    # Check dimensions
    rx, cx = X.shape
    ry, cy = Y.shape

    if rx != ry:
        raise ValueError("The sizes of X and Y do not match")

    # If P is not provided, use the standard Euclidean metric
    if P is None:
        # ||X||^2 for each vector in X
        xx = np.sum(X * X, axis=0)  # shape: (cx,)
        # ||Y||^2 for each vector in Y
        yy = np.sum(Y * Y, axis=0)  # shape: (cy,)
        # <X[:,i], Y[:,j]>
        xy = X.T @ Y  # shape: (cx, cy)

        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 <x,y>
        D = (xx[:, None] + yy[None, :] - 2 * xy)
    else:
        # Check P dimensions
        rp, rp2 = P.shape
        if rx != rp:
            raise ValueError("The sizes of X and P do not match")
        if rp2 != rp:
            raise ValueError("P must be square")

        # x^T P x for each vector in X
        xx = np.sum(X * (P @ X), axis=0)  # shape: (cx,)
        # y^T P y for each vector in Y
        yy = np.sum(Y * (P @ Y), axis=0)  # shape: (cy,)
        # x^T P y
        xy = X.T @ (P @ Y)  # shape: (cx, cy)
        # y^T P x (transpose of the above)
        yx = Y.T @ (P @ X)  # shape: (cy, cx)

        D = (xx[:, None] + yy[None, :] - xy - yx.T)

    # Check for negative values due to potential numerical issues
    if np.any(D < 0):
        # This warning matches the MATLAB warning
        print("Warning: gsp_distanz: P is not semipositive or X is not real!")

    # Take square root of distances
    D = np.sqrt(np.abs(D))

    # If Y = X, set diagonal to zero
    if Y is X:
        np.fill_diagonal(D, 0.0)

    return D