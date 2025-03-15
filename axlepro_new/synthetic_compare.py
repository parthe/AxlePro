from solvers import *
from utils import *

K = LaplacianKernel(bandwidth=1.)
n, d, c, s, q = 3000, 3, 2, 2000, 50
epochs = 60

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# DATADIR = 'cifar-10-batches-py'
# X, y = read_cifar10(DATADIR, parts=3, test=False)

X = torch.randn(n, d, device=DEVICE)
y = torch.randn(n, c, device=DEVICE)

X = X * 0.1
X = X.to(DEVICE)
y = y.to(DEVICE)
print(X.device)
print(X.shape)

ahat2, param, time = lm_axlepro_solver(K, X, y, s, 0, epochs=epochs, verbose=True)
err2_1 = compute_err(param, K, X, X, y, indices=None, compute_class_err=False)
plt.plot(err2_1, '-b^', label='AxlePro 2, q=0')
print(err2_1)

ahat2, param, time = lm_axlepro_solver(K, X, y, s, 10, epochs=epochs, verbose=True)
err2_1 = compute_err(param, K, X, X, y, indices=None, compute_class_err=False)
plt.plot(err2_1, '-gs', label='AxlePro 2, q=10')
print(err2_1)

ahat2, param, time = lm_axlepro_solver(K, X, y, s, 50, epochs=epochs, verbose=True)
err2_2 = compute_err(param, K, X, X, y, indices=None, compute_class_err=False)
plt.plot(err2_2, '-yd', label='AxlePro 2, q=50')
print(err2_2)

ahat2, param, time = lm_axlepro_solver(K, X, y, s, 100, epochs=epochs, verbose=True)
err2_3 = compute_err(param, K, X, X, y, indices=None, compute_class_err=False)
plt.plot(err2_3, '-ro', label='AxlePro 2, q=100')
print(err2_3)


ahat2, param, time = eigenpro2(K, X, y, s, 100, epochs=epochs, verbose=True)
err2_4 = compute_err(param, K, X, X, y, indices=None, compute_class_err=False)
plt.plot(err2_4, '-c*', label='EigenPro 2, q=100')
print(err2_4)

plt.yscale('log')
plt.title(f'Convergence rate with different precondition level')
plt.xlabel('Epochs')
plt.ylabel('MSE (train)')
plt.legend()
plt.savefig('compare_s')
plt.show()
