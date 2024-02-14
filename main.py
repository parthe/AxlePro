import torch
from torchkernels.kernels.radial import LaplacianKernel
import matplotlib.pyplot as plt
from axlepro.solvers import axlepro_solver, lm_axlepro_solver
from axlepro.models import KernelModel

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

K = LaplacianKernel(bandwidth=1.)
n, d, c, s, q = 1000, 3, 2, 100, 10
epochs = 1000

X = torch.randn(n, d)
y = torch.randn(n, c)

ahat1, err1 = axlepro_solver(K, X, y, q, epochs=epochs)
plt.plot(err1, 'b', label='AxlePro')
print(err1[-1])

model1 = KernelModel(kernel=K, centers=X, preconditioner_level=q, verbose=True)
model1.fit(y, epochs=epochs)
print(model1.score(X, y))

ahat2, err2 = lm_axlepro_solver(K, X, y, s, q, epochs=epochs)
plt.plot(err2, 'g', label='LM-AxlePro')
print(err2[-1])

model2 = KernelModel(kernel=K, centers=X, preconditioner_level=q, nystrom_size=s, verbose=True)
model2.fit(y, epochs=epochs)
print(model2.score(X, y))

plt.yscale('log')
plt.title(f'Nyström subset size = {s}')
plt.legend()
plt.show()