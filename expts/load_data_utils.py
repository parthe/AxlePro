import os
import numpy as np
import torch
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def read_cifar10(DATADIR=None, parts=5, test=False, validation=False):
    X = []
    y = []
    if test==False:
        for i in range(1, min(6, parts + 1)):
            data_dict = unpickle(
                os.path.join(DATADIR, f'data_batch_{i}'))
            X.append(data_dict[b'data'])
            y.append(np.array(data_dict[b'labels']))
    if validation:
        data_dict = unpickle(
            os.path.join(DATADIR, f'data_batch_1'))
        X.append(data_dict[b'data'])
        y.append(np.array(data_dict[b'labels']))
    if test==True:
        data_dict = unpickle(
            os.path.join(DATADIR, f'test_batch'))
        X.append(data_dict[b'data'])
        y.append(np.array(data_dict[b'labels']))
    X = torch.from_numpy(np.concatenate(X) / 255.).float()
    y = torch.from_numpy(np.concatenate(y))
    y = torch.nn.functional.one_hot(y.long()).to(torch.float32)
    return X, y

def read_susy(DATADIR, train_ratio=None):
    susy = pd.read_csv(DATADIR)
    susy_tensor = torch.tensor(susy.to_numpy())
    if train_ratio==None:
        # num_classes = 2
        # one_hot_matrix = torch.eye(num_classes)
        # one_hot_encoded = one_hot_matrix[torch.squeeze(susy_tensor[:, 0:1].to(dtype=torch.int))]
        return susy_tensor[:1000000, 1:], susy_tensor[:1000000, 0].unsqueeze(dim=1)
    else:
        dataset_size = susy_tensor.shape[0]
        train_size = int(train_ratio * dataset_size)
    return susy_tensor[:train_size, 1:], susy_tensor[:train_size, 0:1], susy_tensor[train_size:, 1:], susy_tensor[train_size:, 0:1] #X_train, y_train, X_test, y_test


def read_star(DATADIR, test=False):
    if test==True:
        star = pd.read_csv(DATADIR)
        star_tensor = torch.tensor(star.to_numpy())
        one_hot_matrix = torch.eye(3)
        one_hot_encoded = one_hot_matrix[torch.squeeze(star_tensor[:, -1].to(dtype=torch.int))]
        return star_tensor[:5000, :-1], one_hot_encoded[:5000, :]
    else:
        star = pd.read_csv(DATADIR)
        star_tensor = torch.tensor(star.to_numpy())
        one_hot_matrix = torch.eye(3)
        one_hot_encoded = one_hot_matrix[torch.squeeze(star_tensor[:, -1].to(dtype=torch.int))]
        return star_tensor[5000:, :-1], one_hot_encoded[5000:, :]

def read_emnist(DATADIR, test=False):
    if test==True:
        emnist = pd.read_csv('emnist-digits-test.csv')
        emnist_tensor = torch.tensor(emnist.to_numpy())
        one_hot_matrix = torch.eye(10)
        one_hot_encoded = one_hot_matrix[torch.squeeze(emnist_tensor[:, 0].to(dtype=torch.int))]
        return emnist_tensor[:, 1:], one_hot_encoded
    else:
        emnist = pd.read_csv(DATADIR)
        emnist_tensor = torch.tensor(emnist.to_numpy())
        one_hot_matrix = torch.eye(10)
        one_hot_encoded = one_hot_matrix[torch.squeeze(emnist_tensor[:, 0].to(dtype=torch.int))]
        return emnist_tensor[:, 1:], one_hot_encoded



def standardize(data_tr, data_tst):
    reshaped = False

    # If data is one dimensional, reshape to 2D
    if len(data_tr.shape) == 1:
        reshaped = True
        data_tr = data_tr.reshape(-1, 1)
        data_tst = data_tst.reshape(-1, 1)

    scaler = StandardScaler()
    data_tr = scaler.fit_transform(data_tr)
    data_tst = scaler.transform(data_tst)

    if reshaped:
        data_tr = data_tr.flatten()
        data_tst = data_tst.flatten()

    return data_tr, data_tst


def read_homo(DATADIR, test=False):
    data = loadmat(os.path.join(DATADIR))
    X, y = data["X"], data["Y"]
    y = np.squeeze(y)  # Remove singleton dimension due to .mat format
    Xtr, Xtst, ytr, ytst = train_test_split(
        X, y, train_size=100000, random_state=0
    )
    Xtr, Xtst = standardize(Xtr, Xtst)
    return torch.from_numpy(Xtr), torch.from_numpy(ytr), torch.from_numpy(Xtst), torch.from_numpy(ytst)


def load_data(dataset):
    if dataset=='cifar-10-batches-py':
        X, y = read_cifar10(dataset)
        X_test, y_test = read_cifar10(dataset, test=True)
    elif dataset=='emnist-digits-train.csv':
        X, y = read_emnist(dataset)
        X_test, y_test = read_emnist(dataset, test=True)
    elif dataset == 'star_classification.csv':
        X, y = read_star(dataset)
        X_test, y_test = read_star(dataset, test=True)
    elif dataset == 'homo.mat':
        X, y, X_test, y_test = read_homo(dataset)
    return X, y, X_test, y_test