#!/usr/bin/env python3
import numpy as np
import random
import math as m
from MnnClassifier import MNNClassifier
from oneVsRest import OneVsRest

def scale_X(X):
    c = X.shape[1]
    for v in range(c):
        mi = min(X[:,v])
        X[:,v] = X[:,v]-mi
        ma = max(X[:,v])
        X[:,v] = X[:,v]/ma
    return X

if __name__ == '__main__':
    from sklearn.datasets import make_blobs

    X, t = make_blobs(n_samples=[400, 800, 400], centers=[[0, 0], [1, 2], [2, 3]],
                      n_features=2, random_state=2019)
    #scaler X
    X = scale_X(X)
    indices = np.arange(X.shape[0])
    random.seed(2020)
    random.shuffle(indices)
    X_train = X[indices[:800], :]
    X_val = X[indices[800:1200], :]
    X_test = X[indices[1200:], :]
    t_train = t[indices[:800]]
    t_val = t[indices[800:1200]]
    t_test = t[indices[1200:]]
    indices[:10]
    t2_train = t_train == 1
    t2_train = t2_train.astype('int')
    t2_val = (t_val == 1).astype('int')
    t2_test = (t_test == 1).astype('int')
    print("ovr:")
    diff_values = [m.pow(10,-x) for x in range(1,5)]
    ac_oneVsRest = {}
    for df in diff_values:
        """linReg = LinReg()
        linReg.fit(new_X_train,t_train,diff=df)
        ac_lin_multi[df] = linReg.accuracy(new_X_val,t2_val)"""
        ovr = OneVsRest()
        ovr.fit(X_train,t_train,diff=df)
        ac_oneVsRest[df] = ovr.accuracy(X_val,t_val)
    print("ac_ovr:",ac_oneVsRest)
    print("mnn:")
    mnn = MNNClassifier(dim_hidden=60,eta=0.001)
    mnn.fit(X_train,t_train,epochs=1000)
    print(mnn.accuracy(X_val,t_val))
