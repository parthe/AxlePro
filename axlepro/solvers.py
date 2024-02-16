from functools import cache, partial
from math import sqrt

import scipy
import torch
from axlepro.utils import top_eigensystem, smallest_eigenvalue, KmV
from torchmetrics.functional import mean_squared_error as mse

from axlepro.utils import timer


def hyperparameter_selection(m, n, beta, lqp1, lam_min):
    # assumes lqp1 and lam_min are normalized
    Lm = (beta + (m - 1) * lqp1) / m
    k_m = Lm / lam_min
    ktil_m = 1 + (n - 1) / m
    eta_1 = 1 / Lm
    t_ = sqrt(k_m * ktil_m)
    eta_2 = ((eta_1 * t_) / (t_ + 1)) * (1 - 1 / ktil_m)
    gamma = (t_ - 1) / (t_ + 1)
    return eta_1 / m, eta_2 / m, gamma


def axlepro_solver(kernel_fn, X, y, q, m=None, epochs=1, verbose=False):
    """
    Solves the kernel regression problem
        kernel_fn(X, X) @ weights = y
    using the AxlePro algorithm.
    The returned weights can be used to predict function value at any x
        x -> kernel_fn(x, X) @ weights
    :param kernel_fn: positive definite kernel function
    :param X: training inputs
    :param y: training targets
    :param q: level of preconditioning
    :param m: batch-size (defaults to a critical value minimizing training time)
    :param epochs: number of epochs to train
    :param verbose: whether function should print logging information. helpful when debugging
    :return: a tuple of (weights, error) where the error is computed per epoch
    """
    timer.tic()
    n = X.shape[0]
    F, L, lqp1, beta = top_eigensystem(kernel_fn, X, q, method="scipy.linalg.eigh")
    F.mul_((1 - lqp1 / L).sqrt())
    a = torch.zeros_like(y, dtype=F.dtype)
    b = torch.zeros_like(y, dtype=F.dtype)
    bs_crit = int(beta * n / lqp1) + 1
    m = bs_crit if m is None else m
    mu = smallest_eigenvalue(kernel_fn, X)
    lrs = cache(partial(hyperparameter_selection,
                        n=n, beta=beta, lqp1=lqp1 / n, lam_min=mu / n))
    lr1, lr2, damp = lrs(m)
    setup_time = timer.tocvalue(restart=True)
    if verbose:
        print(f"bs_crit={bs_crit}, m={m}, lr1={lr1.item()}, "
              f"lr2={lr2.item()}, damp={damp}")
        print(f"AxlePro setup time: {setup_time:.2f}s")
    err = torch.ones(epochs) * torch.nan
    time_per_epoch = torch.zeros(epochs)
    for t in range(epochs):
        batches = torch.randperm(n).split(m)
        for i, bids in enumerate(batches):
            lr1, lr2, damp = lrs(len(bids))
            Km = kernel_fn(X[bids], X)
            v = Km @ b - y[bids].type(a.type())
            w = F @ (F[bids].T @ v)
            a_ = a.clone()
            a = b
            a[bids] -= lr1 * v
            a += lr1 * w
            b = (1 + damp) * a - damp * a_
            b[bids] += lr2 * v
            b -= lr2 * w
        time_per_epoch[t] = timer.tocvalue(restart=True)
        err[t] = mse(KmV(kernel_fn, X, X, a), y)
        timer.tocvalue(restart=True)
    if verbose:
        print(f"AxlePro iteration time : {time_per_epoch.sum():.2f}s")
    return a, err


def lm_axlepro_solver(kernel_fn, X, y, s, q, m=None, epochs=1, verbose=False):
    """
    Solves the kernel regression problem
        kernel_fn(X, X) @ weights = y
    using the LM-AxlePro algorithm, a limited-memory version of AxlePro.
    The returned weights can be used to predict function value at any x
        x -> kernel_fn(x, X) @ weights.
    Note: LM-AxlePro uses a Nystrom extension to approximate the preconditioner
    which significantly reduces the setup time, storage requirement, and in some cases
    the per iteration overhead of preconditioning
    :param kernel_fn: positive definite kernel function
    :param X: training inputs
    :param y: training targets
    :param s: size of Nystrom subset to approximate the preconditioner
    :param q: level of preconditioning
    :param m: batch-size (defaults to a critical value minimizing training time)
    :param epochs: number of epochs to train
    :param verbose: whether function should print logging information. helpful when debugging
    :return: a tuple of (weights, error) where the error is computed per epoch
    """
    timer.tic()
    n = X.shape[0]
    nids = torch.randperm(n)[:s]
    G, D, lqp1, beta = top_eigensystem(kernel_fn, X[nids], q, method="scipy.linalg.eigh")
    G.mul_(((1 - lqp1 / D) / D).sqrt())
    a = torch.zeros_like(y, dtype=G.dtype)
    b = torch.zeros_like(y, dtype=G.dtype)
    bs_crit = int(beta * s / lqp1) + 1
    m = bs_crit if m is None else m
    mu = scipy.linalg.eigh(kernel_fn(X, X).cpu(),
                           eigvals_only=True, subset_by_index=[0, 0])[0]
    lrs = cache(partial(hyperparameter_selection,
                        n=n, beta=beta, lqp1=lqp1 / s, lam_min=mu / n))
    lr1, lr2, damp = lrs(m)
    setup_time = timer.tocvalue(restart=True)
    if verbose:
        print(f"bs_crit={bs_crit}, m={m}, lr1={lr1.item()}, "
              f"lr2={lr2.item()}, damp={damp}")
        print(f"LM-AxlePro setup time : {setup_time:.2f}s")

    err = torch.ones(epochs) * torch.nan
    time_per_epoch = torch.zeros(epochs)
    for t in range(epochs):
        batches = torch.randperm(n).split(m)
        for i, bids in enumerate(batches):
            lr1, lr2, damp = lrs(len(bids))
            Km = kernel_fn(X[bids], X)
            v = Km @ b - y[bids].type(a.type())
            w = G @ (G.T @ (Km.T[nids] @ v))
            a_ = a.clone()
            a = b
            a[bids] -= lr1 * v
            a[nids] += lr1 * w
            b = (1 + damp) * a - damp * a_
            b[bids] += lr2 * v
            b[nids] -= lr2 * w
        time_per_epoch[t] = timer.tocvalue(restart=True)
        err[t] = mse(KmV(kernel_fn, X, X, a), y)
        timer.tocvalue(restart=True)
    if verbose:
        print(f"LM-AxlePro iteration time : {time_per_epoch.sum():.2f}s")
    return a, err
