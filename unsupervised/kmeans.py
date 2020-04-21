#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import syntheticdata
import random
from sklearn.cluster import KMeans
from pca import pca
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

if __name__ == '__main__':
    X, y = syntheticdata.get_iris_data()
    _, P = pca(X, 2)
    ks = [2, 3, 4, 5]
    ys = []
    for k in ks:
        KM = KMeans(k)
        ys.append(KM.fit_predict(P))
    for i in range(len(ks)):
        plt.figure()
        plt.scatter(P[:, 0], P[:, 1], c=ys[i])
        plt.title('k= {}'.format(ks[i]))
    plt.figure()
    plt.scatter(P[:, 0], P[:, 1], c=y)
    plt.title('Original data')
    plt.show()

    acc_all = []
    for i in range(len(ks)):
        y_hat = ys[i]
        one_hot = np.zeros((P.shape[0], len(np.unique(y_hat))))
        for x in range(len(y)):
            one_hot[x, y_hat[x]] = 1
        lr = LogisticRegression(multi_class='ovr',solver='lbfgs')
        lr.fit(one_hot, y)
        pred = lr.predict(one_hot)
        acc_all.append(metrics.accuracy_score(y, pred))
    plt.plot(ks, acc_all)
    plt.scatter(ks, acc_all)
    plt.show()
