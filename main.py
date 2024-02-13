import torch
import math

from torchkernels.kernels.radial import LaplacianKernel
import matplotlib.pyplot as plt
from torchkernels.linalg.fmm import KmV
from axlepro import lm_axlepro
from torchmetrics.functional import mean_squared_error as mse

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

K = LaplacianKernel(bandwidth=1.)
n, d, c, s, q = 1000, 3, 2, 100, 10
epochs = 1000

X = torch.randn(n, d)
y = torch.randn(n, c)

ahat, err = lm_axlepro(K, X, y, s, q, epochs=epochs)
plt.plot(err, 'g', label='axlepro')
print(err[-1])

plt.yscale('log')
plt.title(f'Nyström subset size = {s}')
plt.legend()
plt.show()