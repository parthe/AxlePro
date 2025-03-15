import argparse
from solvers import *
from load_data_utils import *
from utils import *
torch.set_default_dtype(torch.float64)

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
parser = argparse.ArgumentParser(description='AxlePro')


parser.add_argument('--dataset', type=str, default='synthetic')
parser.add_argument('--size', type=int, default=5000)
parser.add_argument('--kernel_func', type=str, default='Laplace')
parser.add_argument('--alg', type=str, default='AxlePro 2')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--bs', type=int, default=2000)
parser.add_argument('--q', type=int, default=300)
parser.add_argument('--implementation_number', type=int, default=5)
parser.add_argument('--nystrom', type=int, default=10000)
#for falkon
parser.add_argument('--M', type=int, default=10000)
parser.add_argument('--lambda_val', type=float, default=0.0)
#for pcg
parser.add_argument('--k', type=int, default=300)
parser.add_argument('--sigma', type=float, default=1.0)
args = parser.parse_args()

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    if args.alg == 'all':
        for algorithm in ['AxlePro 2', 'EigenPro 2', 'PCG', 'Falkon']:
            print('running algorithm:', algorithm)
            for implementation_number in range(args.implementation_number):
                print('implementation number=', implementation_number)
                implementation_number += 1
                if args.dataset == 'synthetic':
                    d, c = 3, 2
                    X = torch.randn(args.size, d, device=DEV)
                    y = torch.randn(args.size, c, device=DEV)
                    X_test = torch.randn(500, d, device=DEV)
                    y_test = torch.randn(500, c, device=DEV)
                else:
                    X, y, X_test, y_test = load_data(args.dataset)
                    X, X_test = preprocess_data(X, X_test, dataset=args.dataset, kernel=args.kernel_func)
                X = X.to(DEV)
                y = y.to(DEV)
                X_test = X_test.to(DEV)
                y_test = y_test.to(DEV)
                X = X.double()
                X_test = X_test.double()
                y = y.double()
                y_test = y_test.double()
                # print(X.device)
                # print('train data size=', X.size(0))
                # print('test data size=', X_test.size(0))

                if args.kernel_func == 'Laplace':
                    K = LaplacianKernel(bandwidth=1.)
                elif args.kernel_func == 'Gaussian':
                    K = GaussianKernel(bandwidth=1.)

                match algorithm:
                    case 'AxlePro 2':
                        a, param, runtime = nystrom_axlepro_solver(K, X, y, args.nystrom, args.q, epochs=args.epochs, verbose=True, m=args.bs, save_steps=20)
                        train_mse, train_class_err = compute_err(param, K, X, X, y, indices=None, compute_class_err=True)
                        test_mse, test_class_err = compute_err(param, K, X, X_test, y_test, indices=None, compute_class_err=True)
                    case 'EigenPro 2':
                        a, param, runtime = nystrom_eigenpro_solver(K, X, y, args.nystrom, args.q, epochs=args.epochs, verbose=True, m=args.bs, save_steps=20)
                        train_mse, train_class_err = compute_err(param, K, X, X, y, indices=None, compute_class_err=True)
                        test_mse, test_class_err = compute_err(param, K, X, X_test, y_test, indices=None, compute_class_err=True)
                    case 'PCG':
                        a, param, runtime = PCG(K, X, y, args.k, args.sigma, epochs=args.epochs, save_steps=20, kernel='laplacian')
                        train_mse, train_class_err = compute_err(param, K, X, X, y, indices=None, compute_class_err=True)
                        test_mse, test_class_err = compute_err(param, K, X, X_test, y_test, indices=None, compute_class_err=True)
                    case 'Falkon':
                        alpha, indices, param, runtime = FALKON(X, y, None, args.M, K, args.lambda_val, args.epochs, save_steps=20)
                        train_mse, train_class_err = compute_err(param, K, X, X, y, indices=indices, compute_class_err=True)
                        test_mse, test_class_err = compute_err(param, K, X, X_test, y_test, indices=indices, compute_class_err=True)

                print(train_mse)
                print(test_class_err)
                results_file = args.dataset[:4]+args.kernel_func+"_results.pkl"
                results = load_results(results_file)
                if args.dataset not in results:
                    results[algorithm] = {}
                    results[algorithm][implementation_number] = {
                        "train_mse": train_mse,
                        "test_class_err": test_class_err,
                        "test_mse_test": test_mse,
                        "runtime": runtime,
                        "epochs": args.epochs
                    }
                save_results(results_file, results)
    else:
        print('args.alg=', args.alg)
        for implementation_number in range(1):
            print('implementation number=', implementation_number)
            implementation_number += 1
            if args.dataset == 'synthetic':
                d, c = 3, 2
                X = torch.randn(args.size, d, device=DEV)
                y = torch.randn(args.size, c, device=DEV)
                X_test = torch.randn(500, d, device=DEV)
                y_test = torch.randn(500, c, device=DEV)
            else:
                X, y, X_test, y_test = load_data(args.dataset)
                X, X_test = preprocess_data(X, X_test, dataset=args.dataset, kernel=args.kernel_func)
            X = X.to(DEV)
            y = y.to(DEV)
            X_test = X_test.to(DEV)
            y_test = y_test.to(DEV)
            X = X.double()
            X_test = X_test.double()
            y = y.double()
            y_test = y_test.double()
            # print(X.device)
            # print('train data size=', X.size(0))
            # print('test data size=', X_test.size(0))

            if args.kernel_func == 'Laplace':
                K = LaplacianKernel(bandwidth=1.)
            elif args.kernel_func == 'Gaussian':
                K = GaussianKernel(bandwidth=1.)

            match args.alg:
                case 'AxlePro 2':
                    a, param, runtime = nystrom_axlepro_solver(K, X, y, args.nystrom, args.q, epochs=args.epochs, verbose=True, m=args.bs, save_steps=20)
                    train_mse, train_class_err = compute_err(param, K, X, X, y, indices=None, compute_class_err=True)
                    test_mse, test_class_err = compute_err(param, K, X, X_test, y_test, indices=None, compute_class_err=True)
                case 'EigenPro 2':
                    a, param, runtime = nystrom_eigenpro_solver(K, X, y, args.nystrom, args.q, epochs=args.epochs, verbose=True, m=args.bs, save_steps=20)
                    train_mse, train_class_err = compute_err(param, K, X, X, y, indices=None, compute_class_err=True)
                    test_mse, test_class_err = compute_err(param, K, X, X_test, y_test, indices=None, compute_class_err=True)
                case 'PCG':
                    a, param, runtime = PCG(K, X, y, args.k, args.sigma, epochs=args.epochs, save_steps=20, kernel='laplacian')
                    train_mse, train_class_err = compute_err(param, K, X, X, y, indices=None, compute_class_err=True)
                    test_mse, test_class_err = compute_err(param, K, X, X_test, y_test, indices=None, compute_class_err=True)
                case 'Falkon':
                    alpha, indices, param, runtime = FALKON(X, y, None, args.M, K, args.lambda_val, args.epochs, save_steps=20)
                    train_mse, train_class_err = compute_err(param, K, X, X, y, indices=indices, compute_class_err=True)
                    test_mse, test_class_err = compute_err(param, K, X, X_test, y_test, indices=indices, compute_class_err=True)

            print(train_mse)
            print(test_class_err)
            results_file = args.dataset[:4]+args.kernel_func+"_results.pkl"
            results = load_results(results_file)
            if args.dataset not in results:
                results[args.alg] = {}
                results[args.alg][implementation_number] = {
                    "train_mse": train_mse,
                    "test_class_err": test_class_err,
                    "test_mse_test": test_mse,
                    "runtime": runtime,
                    "epochs": args.epochs
                }
            save_results(results_file, results)
            if args.dataset == 'synthetic':


