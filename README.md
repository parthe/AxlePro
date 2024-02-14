# AxlePro
Fast solver for training kernel machines

## Installation
```
pip install git+ssh://git@github.com/parthe/AxlePro.git
```

# Test installation

```python
import torch
from torchkernels.kernels.radial import LaplacianKernel
import matplotlib.pyplot as plt
from axlepro.solvers import axlepro_solver, lm_axlepro_solver
from axlepro.models import KernelModel

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

K = LaplacianKernel(bandwidth=1.)
n, d, c, s, q = 100, 3, 2, 50, 4
epochs = 1

X = torch.randn(n, d)
y = torch.randn(n, c)

ahat2, err2 = lm_axlepro_solver(K, X, y, s, q, epochs=epochs, verbose=True)
plt.plot(err2, 'g', label='LM-AxlePro')
print(err2[-1])

model2 = KernelModel(kernel=K, centers=X, preconditioner_level=q, nystrom_size=s, verbose=True)
model2.fit(y, epochs=epochs)
print(model2.score(X, y))
```
