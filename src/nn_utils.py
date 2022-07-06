import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def load_and_standardize_data(df, X, y, test_size, seed):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    scaler = preprocessing.MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


# dataset
class DataBuilder(Dataset):
    def __init__(self, df, X, y, config, train=True):
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = load_and_standardize_data(df, X, y, config["test_size"], config["seed"])
        if train:
            self.x = torch.from_numpy(self.X_train)
            self.y = torch.from_numpy(np.array(self.y_train))
            self.len = self.x.shape[0]
        else:
            self.x = torch.from_numpy(self.X_test)
            self.y = torch.from_numpy(np.array(self.y_test))
            self.len = self.x.shape[0]
        del self.X_train, self.X_test, self.y_train, self.y_test

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len


# auto encoder model
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.conv1 = nn.Conv1d(1, 8, 3)
        self.emb = nn.Conv1d(4, 1, 3)
        self.conv2 = nn.Conv1d(1, 8, 3)
        self.out = nn.Linear(888, 900)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x_emb = F.relu(self.emb(x))
        x = F.relu(self.conv2(x_emb))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        output = self.out(x)
        return output, torch.flatten(x_emb, 1)
