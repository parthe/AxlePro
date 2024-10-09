from sklearn import datasets
import numpy as np
import torch
from load_data import Load_Data
import falkon
import os
from torchmetrics.functional import mean_squared_error as mse

DEV = torch.device('cpu')

DATADIR = 'cifar-10-batches-py' #os.path.join(os.environ['DATA_DIR'],'cifar-10-batches-py')

dataloader = Load_Data(DATADIR)

X, y = dataloader.loader(DATADIR)
X_test, y_test = dataloader.loader(DATADIR, test=True)
n = X.shape[0]
# n = 500
X = X.to(DEV)
y = y.to(DEV)
X_test = X_test.to(DEV)
y_test = y_test.to(DEV)
print('moved to device')
X = X.double()
X_test = X_test.double()
y = y.double()
y_test = y_test.double()
# Kmat = K(X, X)
mean = X.mean(dim=0)
std = X.std(dim=0)
y_mean = y.mean(dim=0)
y_std = y.std(dim=0)
# Perform the normalization
# normalize X and X_test


X = (X - mean) * 0.05  # X/X.norm(dim=-1, keepdim=True)#(X - mean)*0.001 #X/X.norm(dim=-1, keepdim=True)
X_test = (X_test - mean) * 0.05  # X_test/X_test.norm(dim=-1, keepdim=True)#(X_test - mean)*0.001#
# y = (y - y_mean) / y_std
# y_test = (y_test - y_mean) / y_std

print(X.device)
print('shape of X is', X.shape)
print('shape of y is', y.shape)
print('shape of X_test is', X_test.shape)
print('shape of y_test is', y_test.shape)
print('initial training MSE', mse(torch.zeros_like(y), y))

options = falkon.FalkonOptions(keops_active="yes", use_cpu=True)

kernel = falkon.kernels.LaplacianKernel(sigma=1, opt=options)
flk = falkon.Falkon(
	kernel=kernel, penalty=1e-5, M=n, 
	maxiter=100, options=options, 
	error_fn=mse, error_every=1)

flk.fit(X, y)

train_pred = flk.predict(X)
test_pred = flk.predict(X_test)



print("Training RMSE: %.3f" % (mse(train_pred, y)))
print("Test RMSE: %.3f" % (mse(test_pred, y_test)))
