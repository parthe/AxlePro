# AxlePro
Faster solver for training kernel machines. It accelerates [EigenPro](https://github.com/EigenPro/EigenPro-pytorch) via momentum.

## Installation
```
pip install git+ssh://git@github.com/parthe/AxlePro.git
```
requires installing `torchkernels` from [here](http://github.com/parthe/torchkernels)

# Test installation

```python
import torch
from torchkernels.kernels.radial import LaplacianKernel
import matplotlib.pyplot as plt
from axlepro.solvers import axlepro_solver, lm_axlepro_solver
from axlepro.models import AxleProKernelModel

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

K = LaplacianKernel(bandwidth=1.)
n, d, c, s, q = 100, 3, 2, 50, 4
epochs = 1

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X = torch.randn(n, d, device=DEVICE)
y = torch.randn(n, c, device=DEVICE)

ahat1, err1 = axlepro_solver(K, X, y, s, q, epochs=epochs)
ahat2, err2 = lm_axlepro_solver(K, X, y, s, q, epochs=epochs)

model1 = AxleProKernelModel(kernel=K, centers=X, preconditioner_level=q)
model1.fit(y, epochs=epochs)

model2 = AxleProKernelModel(kernel=K, centers=X, preconditioner_level=q, nystrom_size=s)
model2.fit(y, epochs=epochs)
print('Laplacian test complete!')
```
