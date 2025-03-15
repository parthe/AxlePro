from torchkernels.kernels.radial import gaussian as K

import torchvision
import os
import math
import numpy as np
import torch
from tqdm.auto import tqdm

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.set_default_dtype(torch.float64)


def KmV(K, X, Z, v, out=None, row_chunk_size=20000, col_chunk_size=20000):
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
            out[i * b_r:(i + 1) * b_r, :] += K(X[i * b_r:(i + 1) * b_r], Z[j * b_c:(j + 1) * b_c]) @ v[j * b_c:(
                                                                                                                           j + 1) * b_c,
                                                                                                     :]

    if return_flag: return out




def conjgrad(funA, r, tmax, T, A, save_steps=20):
    p = r.clone()
    rsold = r.pow(2).sum(0)
    beta = torch.zeros_like(r)
    param = [0] * (save_steps + 1)
    param[0] = torch.linalg.solve(T, torch.linalg.solve(A, beta)) * 1
    save_count = 0
    for i in tqdm(range(tmax)):
        Ap = funA(p)
        a = rsold / (p * Ap).sum(0)
        beta += a * p
        r -= a * Ap
        rsnew = r.pow(2).sum(0)
        p = r + (rsnew / rsold) * p
        rsold = rsnew
        if (i + 1) % (tmax / save_steps) == 0:
            save_count += 1
            param[save_count] = torch.linalg.solve(T, torch.linalg.solve(A, beta)) * 1
    return beta, param


def selectNystromCenters(X, lev_scores, M, n):
    if lev_scores is None:  # Uniform Nystrom
        D = torch.eye(M, device=X.device)
        indices = torch.randperm(n, device=X.device)[:M]
        C = X[indices, :]
    else:  # Approximate Lev. Scores Nystrom
        prob = lev_scores / torch.sum(lev_scores)
        count, ind = discrete_prob_sample(M, prob)

        D = torch.diag(1.0 / torch.sqrt(n * prob[ind] * count))
        C = X[ind, :]

    return C, D, indices


def discrete_prob_sample(M, prob):
    rand_samples = np.random.rand(M)
    edges = np.cumsum(prob)
    hist, _ = np.histogram(rand_samples, bins=np.append(0, edges))
    ind = np.where(hist > 0)[0]
    count = hist[ind]
    return torch.tensor(count), ind

