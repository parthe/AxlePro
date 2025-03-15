import torch
import scipy
import math
from tqdm.auto import tqdm
import os
import pickle
from pytictoc import TicToc
torch.set_default_dtype(torch.float64)
timer = TicToc()

def hyperparam_selection_nystrom(m, n, L1, lqp1, lam_min):
    L = lqp1 / n
    mu = lam_min / n
    L_1 = L1
    kappa_tilde = n
    kappa_tilde_m = kappa_tilde / m + (m - 1) / m
    L_m = L_1 / m + (m - 1) * L / m
    kappa_m = L_m / mu
    eta_1 = 1 / L_m
    eta_2 = ((eta_1 * torch.sqrt(kappa_m * kappa_tilde_m)) / (torch.sqrt(kappa_m * kappa_tilde_m) + 1)) * (
            1 - 1 / kappa_tilde_m)
    gamma = (torch.sqrt(kappa_m * kappa_tilde_m) - 1) / (torch.sqrt(kappa_m * kappa_tilde_m) + 1)
    return eta_1, eta_2, gamma


def top_eigensystem(kernel_fn, X, q, method='scipy.linalg.eigh'):
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
        beta: max{i} of kernel_fn(xi, xi) - \\sum_j=1^q (L[i]-lqp1) psi_j(xi)**2
    """

    n = X.shape[0]
    kmat = kernel_fn(X, X)
    if method == "scipy.linalg.eigh":
        L, E = scipy.linalg.eigh(kmat.cpu().numpy(), subset_by_index=[n - q - 1, n - 1])
        L, E = torch.from_numpy(L).to(kmat.device).flipud(), torch.from_numpy(E).to(kmat.device).fliplr()
    elif method == "torch.lobpcg":
        L, E = torch.lobpcg(kmat, q + 1)
    else:
        raise ValueError("argument `method` can only be "
                         "'scipy.linalg.eigh' or 'torch.lobpcg'")
    beta = (kmat.diag() - (E[:, :q].pow(2) * (L[:q] - L[q])).sum(-1)).max()

    return E[:, :q], L[:q], L[q], beta

def top_eigen_eigh(K, X, q, scale=1):
    n = X.shape[0]
    KXX = scale * K(X, X)  ###or use nystrom
    L, E = scipy.linalg.eigh(KXX.cpu().numpy(), subset_by_index=[n-q-1,n-1])
    L, E = torch.from_numpy(L).to(KXX.device).flipud(), torch.from_numpy(E).to(KXX.device).fliplr()
    mu = scipy.linalg.eigh(KXX.cpu().numpy(), eigvals_only=True, eigvals=(0, 0))
    mu = torch.from_numpy(mu).to(KXX.device).flipud()
    beta = (KXX.diag()/scale - (E[:, :q].pow(2) * (L[:q] - L[q])/scale).sum(-1)).max()
    return E[:, :q], L[:q], L[q], beta, mu

def nystrom_extension(K, X, Xs, E):
    """
        Extend eigenvectors
    """
    #E_ = K(X, Xs) @ E
    E_ = torch.zeros(X.shape[0], E.shape[1], device=X.device)
    iter_num = int(X.shape[0]/20000)
    for i in range(iter_num):
        E_[i*20000: (i+1)*20000, :] = K(X[i*20000: (i+1)*20000, :], Xs) @ E
    if iter_num * 20000 < X.shape[0]:
        E_[iter_num*20000:, :] = K(X[iter_num*20000:, :], Xs) @ E
    return E_/E_.norm(dim=0, keepdim=True)


def smallest_eigenvalue(kernel_fn, X):
    return scipy.linalg.eigh(kernel_fn(X, X).cpu(),
                             eigvals_only=True, subset_by_index=[0, 0])[0]


def KmV(kernel_fn, X, Z, v, out=None, row_chunk_size=20000, col_chunk_size=20000):
    """
        calculate kernel matrix vector product kernel_fn(X, Z) @ v without storing kernel matrix
        If argument `out` is provided, the result is added to `out`
    """
    n_r, n_c = len(X), len(Z)
    b_r = n_r if row_chunk_size is None else row_chunk_size
    b_c = n_c if col_chunk_size is None else col_chunk_size
    return_flag = False
    if out is None:
        return_flag = True
        out = torch.zeros(n_r, *v.shape[1:], device=v.device)

    for i in range(math.ceil(n_r / b_r)):
        for j in range(math.ceil(n_c / b_c)):
            out[i * b_r:(i + 1) * b_r] += kernel_fn(
                X[i * b_r:(i + 1) * b_r],
                Z[j * b_c:(j + 1) * b_c]
            ) @ v[j * b_c:(j + 1) * b_c]

    if return_flag:
        return out

def compute_err(a, K, X, X_test, y, indices=None, compute_class_err=False):
    save_steps = len(a)
    mse = torch.zeros(save_steps)
    if compute_class_err:
        class_err = torch.zeros(save_steps)
        true_label = torch.argmax(y, dim=1)
    for i in tqdm(range(save_steps)):
        if indices is not None:
            alpha = torch.zeros(X.shape[0], y.shape[1], device=y.device)
            alpha[indices, :] = a[i]
        else:
            alpha = a[i]
        pred = KmV(K, X_test, X, alpha)
        mse[i] = (pred - y).pow(2).mean()
        if compute_class_err:
            max_indices = torch.argmax(pred, dim=1)
            zero_mask = (abs(max_indices - true_label) == 0)
            num_zeros = torch.sum(zero_mask).item()
            class_err[i] = num_zeros / y.shape[0]
            #print('max_indices and true_label', max_indices, true_label)
    if compute_class_err:
        return mse, class_err
    else:
        return mse


def smape(y_true, y_pred):
    """
    Compute the Symmetric Mean Absolute Percentage Error (SMAPE) between the true and predicted values.

    Args:
        y_true (torch.Tensor): The true values (ground truth).
        y_pred (torch.Tensor): The predicted values.

    Returns:
        torch.Tensor: The SMAPE value.
    """
    # Ensure the tensors have the same shape
    assert y_true.shape == y_pred.shape, "The true and predicted tensors must have the same shape"

    # Compute the absolute error
    diff = torch.abs(y_true - y_pred)

    # Compute the denominator (mean of absolute values of true and predicted)
    denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2.0

    # Compute SMAPE
    smape_value = torch.mean(diff / denominator) * 100

    return smape_value

def preprocess_data(X, X_test, dataset=None, kernel=None):
    mean = X.double().mean(dim=0)
    std = X.double().std(dim=0)
    if dataset == 'cifar-10-batches-py':
            X, X_test = (X) * 0.05, (X_test) * 0.05
    elif dataset == 'emnist-digits-train.csv':
        X = 0.001 * (X - mean)
        X_test = 0.001 * (X_test - mean)
    elif dataset == 'star_classification.csv':
        if kernel == 'Laplace':
            X = (X - mean)*0.25/std
            X_test = (X_test - mean)*0.25/std
        elif kernel == 'Gaussian':
            X = (X - mean)*2/std
            X_test = (X_test - mean)*2/std
    elif dataset == 'homo.mat':
        X = 1 * X
        X_test = 1 * X_test
    return X, X_test

def load_results(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    return {}

# Function to save results to the file
def save_results(file_path, results):
    with open(file_path, "wb") as f:
        pickle.dump(results, f)