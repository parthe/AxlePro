import gpytorch
import torch
import math
from tqdm.auto import tqdm

from linear_operator.operators import KernelLinearOperator
from torchkernels.kernels.radial import gaussian as K
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
torch.set_default_dtype(torch.float64)

def gaussian_ker(x1, x2):

    sq_dist = (x1.unsqueeze(-2) - x2.unsqueeze(-3)).square().sum(dim=-1)
    kern = sq_dist.div(-2.0).exp()
    return kern

def laplacian_ker(x1, x2):
    sq_dist = (x1.unsqueeze(-2) - x2.unsqueeze(-3)).square().sum(dim=-1).sqrt()
    kern = sq_dist.div(-1.0).exp()
    return kern

def covar_func(x1, x2, lengthscale, outputscale):
    # RBF kernel function
    # x1: ... x N x D
    # x2: ... x M x D
    # lengthscale: ... x 1 x D
    # outputscale: ...
    x1 = x1.div(lengthscale)
    x2 = x2.div(lengthscale)
    sq_dist = (x1.unsqueeze(-2) - x2.unsqueeze(-3)).square().sum(dim=-1)
    kern = sq_dist.div(-2.0).exp().mul(outputscale[..., None, None].square())
    return kern
def precond(L, sigma, y):

    return y/(sigma**2) - (L @ torch.linalg.solve(torch.eye(L.shape[1], device=y.device) + (L.t() @ L)/(sigma**2), (L.t() @ y)))/(sigma**4)


def PKmV(K, X, Z, v, L, sigma, out=None, row_chunk_size=10000, col_chunk_size=10000):
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
        out = torch.zeros(n_r, *v.shape[1:], device=X.device)
    for i in range(math.ceil(n_r / b_r)):
        for j in range(math.ceil(n_c / b_c)):
            out[i * b_r:(i + 1) * b_r, :] += K(X[i * b_r:(i + 1) * b_r], Z[j * b_c:(j + 1) * b_c]) @ v[
                                                                                                  j * b_c:(j + 1) * b_c, :]
    out = precond(L, sigma, out)
    if return_flag: return out


# ker_operator = KernelLinearOperator(
#             X, X, covar_func=gaussian_ker, num_nonbatch_dimensions={"outputscale": 0}
#         )
# sigma = 1
# k = 300
# # KXX=KKK(X,X)
# L = gpytorch.pivoted_cholesky(ker_operator, k, error_tol=None, return_pivots=False)
# endtime = time.time()
# setup_time = endtime - starttime
# print('setup_time for pcg', setup_time)
#
# starttime = time.time()
# a, pcg_mse = precond_conjugate_gradient(K, X, y, L, sigma, epochs=T, save_steps=save_steps, save_KXX=save_KXX)