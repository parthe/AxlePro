import torch
import scipy
from axlepro.utils import top_eigensystem, timer
from functools import cache
from math import sqrt
from torchmetrics.functional import mean_squared_error as mse
from torchkernels.linalg.fmm import KmV


class KernelModel:

    def __init__(
            self,
            kernel,
            centers: torch.Tensor = None,
            preconditioner_level = 1,
            nystrom_size: int = None,
            weights: torch.Tensor = None,
            verbose: bool = False,
    ):
        self.kernel = kernel
        self.nystrom_size: int = nystrom_size
        self.preconditioner_level = preconditioner_level
        self.nystrom_ids = None
        self.critical_batch_size = None
        self.centers = centers
        self.weights = weights
        self.state = None
        self.verbose = verbose
        self.train = False
        self.weights_prev = None
        self.kmat_bids_nids = None

        self.preconditioner_level = preconditioner_level
        if nystrom_size is None:
            self.lm_mode = False
            self.nystrom_size = self.size
            self.nystrom_ids = torch.arange(self.size)
        else:
            self.lm_mode = True
            self.nystrom_size = nystrom_size
            self.nystrom_ids = torch.randperm(self.size)[:self.nystrom_size]

        self.E, self.L, self.lqp1, self.beta = top_eigensystem(
            self.kernel, self.centers[self.nystrom_ids],
            self.preconditioner_level, method="scipy.linalg.eigh")
        self.critical_batch_size = int(self.beta * self.nystrom_size / self.lqp1) + 1
        self.lam_min = scipy.linalg.eigh(
            self.kernel(self.centers[self.nystrom_ids], self.centers[self.nystrom_ids]).cpu(),
            eigvals_only=True, subset_by_index=[0, 0])[0]
        if self.verbose:
            lr1, lr2, damp = self.lrs(self.critical_batch_size)
            print(f"critical batch size={self.critical_batch_size}, "
                  f"lr1={lr1.item()}, "
                  f"lr2={lr2.item()}, "
                  f"damp={damp}")
        if self.lm_mode:
            self.name = "LM-AxlePro"
            self.setup_lm()
        else:
            self.name = "AxlePro"
            self.setup()

    @property
    def size(self):
        return self.centers.shape[0]

    @property
    def dtype(self):
        return self.centers.dtype

    def __call__(self, X):
        return KmV(self.kernel, X, self.centers, self.weights)

    def forward_(self, batch_ids, y_batch):
        kmat = self.kernel(self.centers[batch_ids], self.centers)
        if self.train and self.lm_mode:
            if self.kmat_bids_nids is not None:
                raise RuntimeWarning("forward is overwriting self.kmat_bids_nids")
            self.kmat_bids_nids = kmat.T[self.nystrom_ids]
        v = kmat @ self.state - y_batch
        del kmat
        return v

    def backward_(self, batch_ids, v):
        if not self.train:
            raise RuntimeWarning("`self.backward_` is only allowed during training")
        if self.lm_mode:
            w = self.E @ (self.E.T @ (self.kmat_bids_nids @ v))
            self.kmat_bids_nids = None
        else:
            w = self.E @ (self.E[batch_ids].T @ v)
        return w

    @cache
    def lrs(self, m):
        mu = self.lam_min / self.nystrom_size / self.size
        ktil_m = self.size / m + (m - 1) / m
        Lm = (self.beta + (m - 1) * (self.lqp1 / self.nystrom_size)) / m
        k_m = Lm / mu
        eta_1 = 1 / Lm
        t_ = sqrt(k_m * ktil_m)
        eta_2 = ((eta_1 * t_) / (t_ + 1)) * (1 - 1 / ktil_m)
        gamma = (t_ - 1) / (t_ + 1)
        return eta_1 / m, eta_2 / m, gamma

    def setup(self, *args):
        self.E.mul_((1 - self.lqp1 / self.L).sqrt())

    def setup_lm(self, *args):
        self.E.mul_(((1 - self.lqp1 / self.L) / self.L).sqrt())

    def fit_batch(self, batch_ids, y_batch):
        v = self.forward_(batch_ids, y_batch.type(self.dtype))
        w = self.backward_(batch_ids, v)
        lr1, lr2, damp = self.lrs(len(batch_ids))
        weights_prev = self.weights.clone()
        self.weights = self.state
        self.weights[batch_ids] -= lr1 * v
        self.weights[self.nystrom_ids] += lr1 * w
        self.state = (1 + damp) * self.weights - damp * weights_prev
        self.state[batch_ids] += lr2 * v
        self.state[self.nystrom_ids] -= lr2 * w

    def fit(self, targets, epochs=1, batch_size=None):
        if self.weights is None:
            self.weights = torch.zeros_like(targets)
        elif self.size != targets.shape[0]:
            raise RuntimeError("Size of targets doesn't match size of the model.")
        elif self.weights.shape[1] != targets.shape[1]:
            raise RuntimeError(
                "Weights have been initialized "
                "to fit a different number of targets")
        self.state = self.weights.clone()
        if batch_size is None:
            batch_size = self.critical_batch_size
        self.train = True

        for t in range(epochs):
            batches = torch.randperm(self.size).split(batch_size)
            for i, batch_ids in enumerate(batches):
                self.fit_batch(batch_ids, targets[batch_ids])


        self.train = False

    def score(self, inputs, targets, score_fn=mse):
        return score_fn(self(inputs), targets)
