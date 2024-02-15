import torch
import scipy
import math

from pytictoc import TicToc

timer = TicToc()

def top_eigensystem(K, X, q, method='scipy.linalg.eigh'):
    assert method in {"scipy.linalg.eigh", "torch.lobpcg"}
    """
      Top-q eigen system of kernel_fn(X, X)/n
      where n = len(X)

      Args: 
        kernel_fn: kernel that takes 2 arguments.
        X: of shape (n, d).
        q: number of eigen-modes

      Returns:
        E: top-q eigenvectors
        L: top-q eigenvalues of
        lqp1: q+1 st eigenvalue
        beta: max{i} of kernel_fn(xi, xi) - \sum_j=1^q (L[i]-lqp1) psi_j(xi)**2
    """
  
    n = X.shape[0]
    kmat = K(X, X)
    if method == "scipy.linalg.eigh":
        L, E = scipy.linalg.eigh(kmat.cpu().numpy(), subset_by_index=[n-q-1,n-1])
        L, E = torch.from_numpy(L).to(kmat.device).flipud(), torch.from_numpy(E).to(kmat.device).fliplr()
    elif method == "torch.lobpcg":
        L, E = torch.lobpcg(kmat, q+1)
    beta = (kmat.diag() - (E[:,:q].pow(2)*(L[:q]-L[q])).sum(-1)).max()
  
    return E[:,:q], L[:q], L[q], beta

def smallest_eigenvalue(K, X):
    return scipy.linalg.eigh(K(X, X).cpu(),
                           eigvals_only=True, subset_by_index=[0, 0])[0]


def KmV(K, X, Z, v, out=None, row_chunk_size=None, col_chunk_size=None):
    """
        calculate kernel matrix vector product K(X, Z) @ v without storing kernel matrix
        If argument `out` is provided, the result is added to `out`
    """
    n_r, n_c = len(X), len(Z)
    b_r = n_r if row_chunk_size is None else row_chunk_size
    b_c = n_c if col_chunk_size is None else col_chunk_size
    return_flag = False
    if out is None:
        return_flag = True
        out = torch.zeros(n_r, *v.shape[1:], device=v.device)

    for i in range(math.ceil(n_r/b_r)):
        for j in range(math.ceil(n_c/b_c)):
             out[i*b_r:(i+1)*b_r] += K(X[i*b_r:(i+1)*b_r], Z[j*b_c:(j+1)*b_c]) @ v[j*b_c:(j+1)*b_c]

    if return_flag: return out