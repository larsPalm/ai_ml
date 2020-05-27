#!/usr/bin/env python3
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

def scale_minMax(X):
    c = X.shape[1]
    for v in range(c):
        mi = min(X[:,v])
        X[:,v] = X[:,v]-mi
        ma = max(X[:,v])
        X[:,v] = X[:,v]/ma
    return X

def normalization(X):
    c = X.shape[1]
    for v in range(c):
        s = np.std(X[:,v])
        m = np.mean(X[:,v])
        X[:,v] = X[:,v]-m
        X[:,v] =X[:,v]/s
    return X

if __name__ == '__main__':
    X = np.asarray([[4,2,140],[3,0.1,115]])
    scaler = StandardScaler()
    scaler.fit(X)
    print('sklearn:\n',scaler.transform(X))
    print('own:\n',normalization(X))
