#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import syntheticdata
import random
from sklearn.cluster import KMeans
from Kmeans_own import Kmeans_own
from pca import pca
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

if __name__ == '__main__':
    X, y = syntheticdata.get_iris_data()
    _, P = pca(X, 2)
    sk_km = KMeans(3)
    sk_km.fit(P)
    sk_pred = sk_km.predict(P)
    km = Kmeans_own(3)
    km.fit2(P)
    pred = km.predict(P)
    km2 = Kmeans_own(3)
    km2.fit2(P)
    pred2 = km2.predict(P)
    print(P.shape)
    plt.figure()
    plt.scatter(P[:, 0], P[:, 1], c=pred)
    plt.title('own with k= {}'.format(3))
    plt.figure()
    plt.scatter(P[:, 0], P[:, 1], c=sk_pred)
    plt.title('sklearn with k = {}'.format(3))
    plt.show()
    for k in range(3,10,2):
        km = Kmeans_own(k)
        km.fit2(P)
        pred = km.predict(P)
        plt.figure()
        plt.scatter(P[:, 0], P[:, 1], c=pred)
        plt.title('own with k= {}'.format(k))
    plt.show()
