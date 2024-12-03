import time
import numpy as np  
import networkx as nx 
import matplotlib.pyplot as plt 
import scipy.sparse as sp
from scipy.spatial.distance import squareform  
from scipy.sparse import csr_matrix  
from scipy.sparse.linalg import svds  
from numpy.linalg import eigvals  
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
    Sparse counterpart of numpy's squareform
    
    Parameters:
    w : sparse or dense vector with n(n-1)/2 elements OR matrix with size [n, n] and zero diagonal
    
    Returns:
    W : matrix form of input vector w OR vector form of input matrix W
    """
    import numpy as np
    import scipy.sparse as sp
    from scipy.spatial.distance import squareform

    if sp.issparse(w):
        is_sparse = True
    else:
        is_sparse = False
        w = np.asarray(w)
    
    # Determine if input is a vector
    if w.ndim == 1 or w.shape[0] == 1 or w.shape[1] == 1:
        # VECTOR -> MATRIX
        if w.ndim == 1:
            l = w.shape[0]
        else:
            l = w.shape[0] * w.shape[1]
        n = int(round((1 + np.sqrt(1 + 8*l)) / 2))
        
        # Check input
        if l != n*(n-1)//2:
            raise ValueError("Bad vector size!")
        
        if is_sparse:
            ind_vec = w.nonzero()[0]
            s = w.data
        else:
            ind_vec = np.nonzero(w)[0]
            s = w[ind_vec]
        
        num_nz = len(ind_vec)
        
        ind_i = np.zeros(num_nz, dtype=int)
        ind_j = np.zeros(num_nz, dtype=int)
        
        curr_row = 0
        offset = 0
        len_row = n - 1
        for idx in range(num_nz):
            ind_vec_i = ind_vec[idx]
            while ind_vec_i >= (len_row + offset):
                offset += len_row
                len_row -= 1
                curr_row += 1
            ind_i[idx] = curr_row
            ind_j[idx] = ind_vec_i - offset + curr_row + 1
        
        # For the lower triangular part, add the transposed matrix
        data = np.concatenate([s, s])
        row_indices = np.concatenate([ind_i, ind_j])
        col_indices = np.concatenate([ind_j, ind_i])
        W = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
        return W

    else:
        # MATRIX -> VECTOR
        m, n = w.shape
        if m != n or not np.all(w.diagonal() == 0):
            raise ValueError("Matrix has to be square with zero diagonal!")
        
        if is_sparse:
            ind_i, ind_j = w.nonzero()
            s = w.data
        else:
            ind_i, ind_j = np.nonzero(w)
            s = w[ind_i, ind_j]
        
        # Keep only upper triangular part
        ind_upper = ind_i < ind_j
        ind_i = ind_i[ind_upper]
        ind_j = ind_j[ind_upper]
        s = s[ind_upper]
        
        # Compute new (vector) index from (i,j) (matrix) indices
        new_ind = n * ind_i - ind_i * (ind_i + 1) // 2 + ind_j - ind_i - 1
        l = n * (n - 1) // 2
        w_vec = sp.csr_matrix((s, (new_ind, np.zeros_like(new_ind))), shape=(l, 1))
        return w_vec



def sum_squareform(n, mask=None):
    """
    Computes the sum and transpose sum matrices in a squareform format.

    Parameters:
    -----------
    n : int
        The size of the squareform matrix.
    
    mask : array-like, optional
        A mask to filter the indices. The length of the mask must be n(n-1)/2.
        If provided, only the elements corresponding to the non-zero values in
        the mask are considered.

    Returns:
    --------
    S : csr_matrix
        A sparse matrix so that S * w = sum(W), where w = squareform(W)
    
    St : csr_matrix
        The adjoint (transpose) of S.

    Raises:
    -------
    ValueError
        If the length of the mask is not equal to n(n-1)/2.
    """
    if mask is not None:
        mask = np.asarray(mask).flatten()
        if len(mask) != n * (n - 1) // 2:
            raise ValueError('Mask size has to be n(n-1)/2')

        ind_vec = np.flatnonzero(mask)
        ncols = len(ind_vec)

        I = np.zeros(ncols, dtype=int)
        J = np.zeros(ncols, dtype=int)

        curr_row = 0
        offset = 0
        len_row = n - 1
        for ii in range(ncols):
            ind_vec_i = ind_vec[ii]
            while ind_vec_i > (len_row + offset - 1):
                offset += len_row
                len_row -= 1
                curr_row += 1
            I[ii] = curr_row
            J[ii] = ind_vec_i - offset + curr_row + 1
    else:
        ncols = n * (n - 1) // 2
        I = np.zeros(ncols, dtype=int)
        J = np.zeros(ncols, dtype=int)

        k = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                I[k] = i
                J[k] = j
                k += 1

    # Construct St
    row_indices = np.concatenate([np.arange(ncols), np.arange(ncols)])
    col_indices = np.concatenate([I, J])
    data = np.ones(2 * ncols)

    St = csr_matrix((data, (np.arange(2 * ncols), col_indices)), shape=(2 * ncols, n))
    St = csr_matrix((data, (row_indices, col_indices)), shape=(ncols, n))
    S = St.transpose()

    return S, St





def prox_sum_log(x, gamma, param=None):
    """
    Proximal operator of log-barrier - sum(log(x))

    Solves:
        sol = argmin_{z} 0.5*||x - z||_2^2 - gamma * sum(log(z))

    Parameters:
    ----------
    x : array-like
        Input signal (vector or matrix).
    gamma : float
        Regularization parameter.
    param : dict, optional
        Dictionary of optional parameters:
        - 'verbose': int, verbosity level (default: 1).

    Returns:
    -------
    sol : numpy.ndarray
        Solution to the optimization problem.
    info : dict
        Dictionary summarizing information at convergence.
    """
    if param is None:
        param = {}
    
    verbose = param.get('verbose', 1)
    
    if gamma < 0:
        raise ValueError('Gamma cannot be negative')
    elif gamma == 0:
        stop_error = True
    else:
        stop_error = False
    
    t1 = time.time()
    
    if stop_error:
        sol = x
        info = {
            'algo': 'prox_sum_log',
            'iter': 0,
            'final_eval': 0,
            'crit': '--',
            'time': time.time() - t1
        }
        return sol, info
    
    # Compute the solution
    sol = (x + np.sqrt(x**2 + 4*gamma)) / 2
    
    # Compute the final evaluation of the function at the solution
    final_eval = 0.5 * np.sum((x - sol)**2) - gamma * np.sum(np.log(sol.flatten()))
    
    info = {
        'algo': 'prox_sum_log',
        'iter': 0,
        'final_eval': final_eval,
        'crit': '--',
        'time': time.time() - t1
    }
    
    # Verbose output
    if verbose >= 1:
        print(f'  prox_sum_log: final evaluation = {info["final_eval"]:.6e}')
        if verbose > 1:
            n_neg = np.sum(sol.flatten() <= 0)
            if n_neg > 0:
                print(f' ({n_neg} negative elements in solution, log not defined, check stability)')
        print()
    
    return sol, info



def gsp_distanz(X, Y=None, P=None):
    """
    Calculates the distances between all vectors in X and Y using a provided matrix P for scaling (weighted distances).
    
    Parameters:
        X (numpy.ndarray): Matrix with column vectors of shape (n_features, n_samples_X).
        Y (numpy.ndarray, optional): Matrix with column vectors of shape (n_features, n_samples_Y). Defaults to X.
        P (numpy.ndarray, optional): Weight matrix of shape (n_features, n_features). Defaults to the identity matrix.

    Returns:
        numpy.ndarray: Distance matrix of shape (n_samples_X, n_samples_Y), not squared.
        
    Raises:
        ValueError: If the dimensions of X and Y do not match.
        ValueError: If the dimensions of P do not match the number of features in X.

    Usage:
        D = gsp_distanz(X, Y, P)
        
    Notes:
        This function computes the following:
        
            D = sqrt((X - Y)^T P (X - Y))
        
        for all vectors in X and Y. If P is not provided, it defaults to the identity matrix, reducing the calculation to the Euclidean distance.
        The function is optimized for speed using vectorized operations, avoiding explicit loops.
    """
    if Y is None:
        Y = X

    if X.shape[0] != Y.shape[0]:
        raise ValueError("The sizes of X and Y do not match!")

    n_features, n_samples_X = X.shape
    _, n_samples_Y = Y.shape

    if P is None:
        xx = np.sum(X**2, axis=0)  # ||x||^2, shape (n_samples_X,)
        yy = np.sum(Y**2, axis=0)  # ||y||^2, shape (n_samples_Y,)
        xy = X.T @ Y               # x^T y, shape (n_samples_X, n_samples_Y)
        D = np.abs(np.add.outer(xx, yy) - 2 * xy)
    else:
        rp, rp2 = P.shape
        if n_features != rp or rp != rp2:
            raise ValueError("P must be square and match the dimension of X!")

        # Compute x^T P x and y^T P y
        xx = np.sum(X * (P @ X), axis=0)  # shape (n_samples_X,)
        yy = np.sum(Y * (P @ Y), axis=0)  # shape (n_samples_Y,)

        # Compute x^T P y and y^T P x
        xy = X.T @ (P @ Y)  # shape (n_samples_X, n_samples_Y)
        yx = Y.T @ (P @ X)  # shape (n_samples_Y, n_samples_X)

        # D = |xx_i + yy_j - (x_i^T P y_j) - (y_j^T P x_i)|
        # Since yx.T has shape (n_samples_X, n_samples_Y), we can subtract it directly
        D = np.abs(np.add.outer(xx, yy) - xy - yx.T)

    # Check for negative values in D
    if np.any(D < 0):
        print('Warning: P is not semipositive or x is not real!')

    # Take the square root
    D = np.sqrt(D)

    # If Y is X, set the diagonal to zero
    if Y is X:
        np.fill_diagonal(D, 0)

    return D


