from functools import cache, partial
from math import sqrt
from pcg_utils import *
import scipy
import torch
from utils import top_eigensystem, smallest_eigenvalue, KmV, top_eigen_eigh, nystrom_extension, hyperparam_selection_nystrom
from torchmetrics.functional import mean_squared_error as mse
from linear_operator.operators import KernelLinearOperator
import gpytorch
from tqdm.auto import tqdm
from utils import timer
from torchkernels.kernels.radial import LaplacianKernel
from torchkernels.kernels.radial import GaussianKernel
import matplotlib.pyplot as plt
from falkon_utils import *
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

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

def nystrom_eigenpro_solver(kernel_fn, X, y, s, q, m=None, epochs=1, verbose=False, save_steps=20):
    timer.tic()
    n = X.shape[0]
    nids = torch.randperm(n)[:s]
    print('Computing Eigensystems!')
    E, L, lqp1, beta= top_eigensystem(kernel_fn, X[nids], q)
    L, lqp1 = L/s, lqp1/s
    E = nystrom_extension(kernel_fn, X, X[nids], E)
    E, R = torch.linalg.qr(E)
    F = E * (1 - lqp1 / L).sqrt()
    a = torch.zeros_like(y, dtype=X.dtype, device=X.device)
    bs_crit = int(beta / lqp1) + 1
    if m is None: m = bs_crit
    lr = lambda bs: 1 / beta if bs < bs_crit else 2 / (beta + (bs - 1) * lqp1)
    setup_time = timer.tocvalue(restart=True)
    if verbose:
        print(f"bs_crit={bs_crit}, m={m}, lr={lr(m).item()}")
        print(f"Nystrom-EigenPro setup time : {setup_time:.2f}s")

    param = [0] * (save_steps + 1)
    param[0] = a * 1
    save_count = 0
    gap = epochs / save_steps
    time_per_epoch = torch.zeros(epochs)
    for t in tqdm(range(epochs)):
        batches = torch.randperm(n).split(m)
        for i, bids in enumerate(batches):
            g = kernel_fn(X[bids], X) @ a - y[bids, :]
            a[bids, :] = a[bids, :] - lr(len(bids)) * g
            a += lr(len(bids)) * (F @ (F[bids].T @ g))
        if (t + 1) % gap == 0:
            save_count += 1
            param[save_count] = a * 1
        timer.tocvalue(restart=True)
    if verbose:
        print(f"Nystrom-EigenPro iteration time : {time_per_epoch.sum():.2f}s")
    return a, param, time_per_epoch.sum()

def eigenpro2(kernel_fn, X, y, s, q, m=None, epochs=1, verbose=False, save_steps=20):
    """
        Storage: (n x q) + s2
        FLOPS at setup: (s x q2) +
        FLOPS per batch: (n x m) + {(m x q) + (n x q)}
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
        print(f"LM-EigenPro setup time : {setup_time:.2f}s")

    param = [0] * (save_steps + 1)
    param[0] = a * 1
    save_count = 0
    gap = epochs / save_steps
    time_per_epoch = torch.zeros(epochs)
    for t in tqdm(range(epochs)):
        batches = torch.randperm(n).split(m)
        for i, bids in enumerate(batches):
            lr1, lr2, damp = lrs(len(bids))
            lr2, damp = 0, 0
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
        if (t + 1) % gap == 0:
            save_count += 1
            param[save_count] = a * 1 #mse(KmV(kernel_fn, X, X, a), y)
        timer.tocvalue(restart=True)
    if verbose:
        print(f"LM-EigenPro iteration time : {time_per_epoch.sum():.2f}s")
    return a, param, time_per_epoch.sum()


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

def nystrom_axlepro_solver(kernel_fn, X, y, s, q, m=None, epochs=1, verbose=False, save_steps=20):
    timer.tic()
    n = X.shape[0]
    nids = torch.randperm(n)[:s]
    print('Computing Eigensystems!')
    E, L, lqp1, beta, mu = top_eigen_eigh(kernel_fn, X[nids], q)
    E = nystrom_extension(kernel_fn, X, X[nids], E)
    E, R = torch.linalg.qr(E)
    F = E * (1 - lqp1 / L).sqrt()
    a = torch.zeros_like(y, dtype=X.dtype, device=X.device)
    b = torch.zeros_like(y, dtype=X.dtype, device=X.device)
    bs_crit = int(beta / lqp1) + 1

    if m is None: m = bs_crit
    eta1, eta2, gamma = hyperparam_selection_nystrom(m, n, beta, lqp1, mu)
    eta1, eta2 = eta1 / m, eta2 / m
    setup_time = timer.tocvalue(restart=True)
    if verbose:
        print(f"lr1={eta1.item()}, "
              f"lr2={eta2.item()}, damp={gamma}")
        print(f"Nystrom-AxlePro setup time : {setup_time:.2f}s")

    param = [0] * (save_steps + 1)
    param[0] = a * 1
    save_count = 0
    gap = epochs / save_steps
    time_per_epoch = torch.zeros(epochs)
    for t in tqdm(range(epochs)):
        batches = torch.randperm(n).split(m)
        for i, bids in enumerate(batches):
            gm = kernel_fn(X[bids], X) @ b - y[bids]
            h = F @ (F[bids].T @ gm)
            a_ = a.clone()
            a = b + eta1 * h
            a[bids, :] -= eta1 * gm
            b = a + gamma * (a - a_) - eta2 * h
            b[bids, :] += eta2 * gm
        if (t + 1) % gap == 0:
            save_count += 1
            param[save_count] = a * 1
        timer.tocvalue(restart=True)
    if verbose:
        print(f"Nystrom-AxlePro iteration time : {time_per_epoch.sum():.2f}s")
    return a, param, time_per_epoch.sum()

def lm_axlepro_solver(kernel_fn, X, y, s, q, m=None, epochs=1, verbose=False, save_steps=20):
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
    print('Computing Eigensystems!')
    G, D, lqp1, beta = top_eigensystem(kernel_fn, X[nids], q, method="scipy.linalg.eigh")#torch.lobpcg
    G.mul_(((1 - lqp1 / D) / D).sqrt())
    a = torch.zeros_like(y, dtype=G.dtype)
    b = torch.zeros_like(y, dtype=G.dtype)
    bs_crit = int(beta * s / lqp1) + 1
    m = bs_crit if m is None else m
    print('Computing smallest eigenvalue!')
    mu = scipy.linalg.eigh(kernel_fn(X[nids], X[nids]).cpu(),
                           eigvals_only=True, eigvals=(0, 0))#, subset_by_index=[0, 0])[0]
    mu = torch.from_numpy(mu).to(X.device).flipud()
    print('mu=', mu)
    lrs = cache(partial(hyperparameter_selection,
                        n=n, beta=beta, lqp1=lqp1 / s, lam_min=mu / n))
    lr1, lr2, damp = lrs(m)
    setup_time = timer.tocvalue(restart=True)
    if verbose:
        print(f"bs_crit={bs_crit}, m={m}, lr1={lr1.item()}, "
              f"lr2={lr2.item()}, damp={damp}")
        print(f"LM-AxlePro setup time : {setup_time:.2f}s")

    param = [0] * (save_steps + 1)
    param[0] = a * 1
    save_count = 0
    gap = epochs / save_steps
    time_per_epoch = torch.zeros(epochs)
    for t in tqdm(range(epochs)):
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
        if (t + 1) % gap == 0:
            save_count += 1
            param[save_count] = a * 1# mse(KmV(kernel_fn, X, X, a), y)
        timer.tocvalue(restart=True)
    if verbose:
        print(f"LM-AxlePro iteration time : {time_per_epoch.sum():.2f}s")
    return a, param, time_per_epoch.sum()


def eigenpro_4_solver(kernel_fn, Z, X, y, s, q, T, m=None, epochs=1, verbose=False, error_per_step=False):
    """
    Solves the kernel regression problem
        kernel_fn(X, X) @ weights = y
    using the LM-AxlePro algorithm, a limited-memory version of AxlePro.
    The returned weights can be used to predict function value at any x
        x -> kernel_fn(x, X) @ weights.
    Note: LM-AxlePro uses a Nystrom extension to approximate the preconditioner
    which significantly reduces the setup time, storage requirement, and in some cases
    the per iteration overhead of preconditioning
    :param Z: centers
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
    KZZ = kernel_fn(Z, Z)
    KZJ = kernel_fn(Z, X[nids])
    G, D, lqp1, beta = top_eigensystem(kernel_fn, X[nids], q, method="scipy.linalg.eigh")
    G.mul_(((1 - lqp1 / D) / D).sqrt())
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
        print(f"eigenpro4 setup time : {setup_time:.2f}s")
    time_per_epoch = torch.zeros(epochs)

    steps_before_proj = 0
    a_Z = torch.zeros(Z.size(0), y.size(1), dtype=G.dtype, device=X.device)
    a_X = torch.zeros(0, y.size(1), dtype=G.dtype, device=X.device)
    a_J = torch.zeros(s, y.size(1), dtype=G.dtype, device=X.device)
    C_t = []
    KZC = torch.zeros(Z.size(0), 0, dtype=G.dtype, device=X.device)
    if error_per_step==False:
        err = torch.ones(epochs) * torch.nan
    else:
        err = torch.zeros(0, device=X.device)
        projection_indices = []
        KXZ = kernel_fn(X, Z)
        KXJ = kernel_fn(X, X[nids])
        init_err = mse(KXZ @ a_Z + kernel_fn(X, X[C_t]) @ a_X + KXJ @ a_J, y).unsqueeze(0)
        err = torch.cat((err, init_err))
        num_of_steps = 0
        projection_indices.append(num_of_steps)

    for t in range(epochs):
        batches = torch.randperm(n).split(m)
        for i, bids in enumerate(batches):
            lr1, lr2, damp = lrs(len(bids))
            K_B_J = kernel_fn(X[bids], X[nids])
            KBZ = kernel_fn(X[bids], Z)
            v = KBZ @ a_Z + kernel_fn(X[bids], X[C_t]) @ a_X + K_B_J @ a_J - y[bids].type(a_Z.type())
            w = G @ (G.T @ (K_B_J.T @ v))
            a_J += lr1 * w
            a_X = torch.cat((a_X, -lr1 * v), 0)
            C_t = C_t + bids.tolist()
            steps_before_proj += 1
            KZC = torch.cat((KZC, KBZ.T), 1)
            num_of_steps += 1
            if steps_before_proj == T or i == len(batches) - 1:
                g_Z = KZZ @ a_Z + KZC @ a_X + KZJ @ a_J
                a_Z = torch.linalg.solve(KZZ, g_Z)
                #re-initialize
                a_X = torch.zeros(0, y.size(1), dtype=G.dtype, device=X.device)
                a_J = torch.zeros(s, y.size(1), dtype=G.dtype, device=X.device)
                C_t = []
                steps_before_proj = 0
                KZC = torch.zeros(Z.size(0), 0, dtype=G.dtype, device=X.device)
                projection_indices.append(num_of_steps)
            if error_per_step == True:
                train_err = mse(KXZ @ a_Z + kernel_fn(X, X[C_t]) @ a_X + KXJ @ a_J, y).unsqueeze(0)
                err = torch.cat((err, train_err))
        time_per_epoch[t] = timer.tocvalue(restart=True)
        if error_per_step == False:
            err[t] = mse(KmV(kernel_fn, X, Z, a_Z), y)
        timer.tocvalue(restart=True)
    if verbose:
        print(f"eigenpro4 iteration time : {time_per_epoch.sum():.2f}s")
    return a_Z, err, projection_indices




def axlepro_4_solver(kernel_fn, Z, X, y, s, q, T, m=None, epochs=1, verbose=False, error_per_step=False, Z_indices=None):
    timer.tic()
    n = X.shape[0]
    nids = torch.randperm(n)[:s]
    intersection_ind_J_Z = nids[torch.isin(nids, Z_indices)]
    indices_Z_in_J = [torch.nonzero(nids == x, as_tuple=True)[0].item() for x in intersection_ind_J_Z]
    indices_Z_in_J = torch.tensor(indices_Z_in_J)
    indices_J_in_Z = [torch.nonzero(Z_indices == x, as_tuple=True)[0].item() for x in intersection_ind_J_Z]
    indices_J_in_Z = torch.tensor(indices_J_in_Z)
    # X[Z_indices[indices_J_in_Z]] = X[nids[indices_Z_in_J]]
    KZZ = kernel_fn(Z, Z)
    KZJ = kernel_fn(Z, X[nids])
    G, D, lqp1, beta = top_eigensystem(kernel_fn, X[nids], q, method="scipy.linalg.eigh")
    G.mul_(((1 - lqp1 / D) / D).sqrt())
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
        print(f"axlepro_4 setup time : {setup_time:.2f}s")


    time_per_epoch = torch.zeros(epochs)
    steps_before_proj = 0
    a_Z = torch.zeros(Z.size(0), y.size(1), dtype=G.dtype, device=X.device)
    a_X = torch.zeros_like(y, dtype=G.dtype)
    a_J = torch.zeros(s, y.size(1), dtype=G.dtype, device=X.device)
    b_Z = torch.zeros(Z.size(0), y.size(1), dtype=G.dtype, device=X.device)
    b_X = torch.zeros_like(y, dtype=G.dtype)
    b_J = torch.zeros(s, y.size(1), dtype=G.dtype, device=X.device)
    if error_per_step==False:
        err = torch.ones(epochs) * torch.nan
    else:
        err = torch.zeros(0, device=X.device)
        projection_indices = []
        KXZ = kernel_fn(X, Z)
        KXJ = kernel_fn(X, X[nids])
        KXX = kernel_fn(X, X)
        init_err = mse(KXZ @ b_Z + KXX @ b_X + KXJ @ b_J, y).unsqueeze(0)
        err = torch.cat((err, init_err))
        num_of_steps = 0
        projection_indices.append(num_of_steps)

    for t in range(epochs):
        batches = torch.randperm(n).split(m)
        for i, bids in enumerate(batches):
            lr1, lr2, damp = lrs(len(bids))
            #Km = kernel_fn(X[bids], X)
            K_B_J = kernel_fn(X[bids], X[nids])
            KBZ = kernel_fn(X[bids], Z)
            v = KBZ @ b_Z + kernel_fn(X[bids], X) @ b_X + K_B_J @ b_J - y[bids].type(b_Z.type())
            w = G @ (G.T @ (K_B_J.T @ v))
            a_J_prev = a_J.clone()
            a_X_prev = a_X.clone()
            a_Z_prev = a_Z.clone()
            a_Z = b_Z.clone()
            b_Z = (1 + damp) * a_Z - damp * a_Z_prev
            a_J = b_J + lr1 * w
            a_X = b_X.clone()
            a_X[bids] -= lr1 * v
            b_J = (1 + damp) * a_J - damp * a_J_prev - lr2 * w
            b_X = (1 + damp) * a_X - damp * a_X_prev
            b_X[bids] += lr2 * v
            steps_before_proj += 1
            num_of_steps += 1
            if steps_before_proj == T or i == len(batches) - 1:
                f_Z_a_X = KXZ.T @ a_X
                f_Z_a_J = KZJ @ a_J
                g_Z_b_X = KXZ.T @ b_X
                g_Z_b_J = KZJ @ b_J
                a_X = torch.zeros_like(y, dtype=G.dtype)
                b_X = torch.zeros_like(y, dtype=G.dtype)
                a_J = torch.zeros(s, y.size(1), dtype=G.dtype, device=X.device)
                b_J = torch.zeros(s, y.size(1), dtype=G.dtype, device=X.device)

                a_Z += torch.linalg.solve(KZZ, f_Z_a_X)
                b_Z += torch.linalg.solve(KZZ, g_Z_b_X)
                a_Z += torch.linalg.solve(KZZ, f_Z_a_J)
                b_Z += torch.linalg.solve(KZZ, g_Z_b_J)

                steps_before_proj = 0
                projection_indices.append(num_of_steps)
            if error_per_step==True:
                train_err = mse(KXZ @ b_Z + KXX @ b_X + KXJ @ b_J, y).unsqueeze(0)
                err = torch.cat((err, train_err))
        time_per_epoch[t] = timer.tocvalue(restart=True)
        if error_per_step==False:
            err[t] = mse(KmV(kernel_fn, X, Z, b_Z), y)
        timer.tocvalue(restart=True)
    if verbose:
        print(f"axlepro_4 iteration time : {time_per_epoch.sum():.2f}s")
    return b_Z, err, projection_indices


def PCG(K, X, y, k, sigma, epochs=None, save_steps=20, kernel=None):
    n = X.shape[0]
    if epochs is None: epochs = n
    if kernel=='laplacian':
        ker_operator = KernelLinearOperator(
            X, X, covar_func=laplacian_ker, num_nonbatch_dimensions={"outputscale": 0}
        )
    elif kernel =='gaussian':
        ker_operator = KernelLinearOperator(
            X, X, covar_func=gaussian_ker, num_nonbatch_dimensions={"outputscale": 0}
        )
    L = gpytorch.pivoted_cholesky(ker_operator, k, error_tol=None, return_pivots=False)
    gap = epochs / save_steps
    a = torch.zeros_like(y, dtype=X.dtype, device=X.device)
    y = precond(L, sigma, y)
    r = y.type(X.type()).clone()
    p = r.clone()
    param = [0]*(save_steps + 1)
    param[0] = a*1
    save_count = 0
    time_per_epoch = torch.zeros(epochs)
    timer.tic()
    for t in tqdm(range(epochs)):
        Kp = PKmV(K, X, X, p, L, sigma)
        r_norm2 = r.pow(2).sum(0)
        alpha = r_norm2 / (p * Kp).sum(0)
        a += alpha * p
        r -= alpha * Kp
        beta = r.pow(2).sum(0) / r_norm2
        p = r + beta * p
        time_per_epoch[t] = timer.tocvalue(restart=True)
        if (t + 1) % gap == 0:
            save_count += 1
            param[save_count] = a*1
        timer.tocvalue(restart=True)
    print(f"PCG iteration time : {time_per_epoch.sum():.2f}s")
    return a, param, time_per_epoch.sum()

def FALKON(X, Y, lev_scores, M, KernelMatrix, lambda_val, t, save_steps=20):
    n = X.size(0)
    C, D, indices = selectNystromCenters(X, lev_scores, M, n)
    KMM = KernelMatrix(C, C)
    eps = torch.finfo(torch.float).eps
    T = torch.linalg.cholesky(D @ KMM @ D.t() + eps * M * torch.eye(M, device=X.device), upper=True)
    A = torch.linalg.cholesky(T @ T.t() / M + lambda_val * torch.eye(M, device=X.device), upper=True)
    def KnMtimesVector(u, v):
        c = max(v.shape[-1], u.shape[-1])
        w = torch.zeros(M, c, device=u.device)
        ms = np.ceil(np.linspace(0, n, int(np.ceil(n / M) + 1)))
        for i in range(1, math.ceil(n / M) + 1):
            Kr = KernelMatrix(X[int(ms[i - 1]): int(ms[i]), :], C)
            w += Kr.t() @ ((Kr @ u) + v[int(ms[i - 1]):int(ms[i]), :])
        return w

    def BHB(u):
        return torch.linalg.solve(A.t(), torch.linalg.solve(T.t(), KnMtimesVector(
            torch.linalg.solve(T, torch.linalg.solve(A, u)), torch.zeros(n, 1).to(u.device)) / n)
                                  + lambda_val * torch.linalg.solve(A, u))

    timer.tic()
    r = torch.linalg.solve(A.t(), torch.linalg.solve(T.t(), KnMtimesVector(torch.zeros(M, 1, device=X.device), Y / n)))
    beta, param = conjgrad(BHB, r, t, T, A, save_steps=save_steps)
    alpha = torch.linalg.solve(T, torch.linalg.solve(A, beta))
    run_time = timer.tocvalue()
    return alpha, indices, param, run_time

