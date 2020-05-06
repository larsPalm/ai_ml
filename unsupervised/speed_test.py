#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import syntheticdata
import random
from sklearn.cluster import KMeans
import time
from pca import pca
from Kmeans_own import Kmeans_own
from statistics import mean


if __name__ == '__main__':
    X, y = syntheticdata.get_iris_data()
    _, P = pca(X, 2)
    tid_egen1 = []
    tid_egen2 = []
    tid_sklearn = []
    sim = 5
    ks = [x for x in range(1,16,2)]
    for k in ks:
        tid_egen1_local = []
        tid_egen2_local = []
        tid_sklearn_local = []
        for x in range(sim):
            #egen1
            egen1 = Kmeans_own(k)
            start = time.time()
            egen1.fit(P)
            egen1.predict(P)
            stop = time.time()
            tid_egen1_local.append(stop-start)
            #egen2
            egen2 = Kmeans_own(k)
            start = time.time()
            egen2.fit2(P)
            egen2.predict(P)
            stop = time.time()
            tid_egen2_local.append(stop-start)
            #sklearn
            sk = KMeans(k)
            start = time.time()
            sk.fit_predict(P)
            stop = time.time()
            tid_sklearn_local.append(stop-start)
        tid_egen1.append(mean(tid_egen1_local))
        tid_egen2.append(mean(tid_egen2_local))
        tid_sklearn.append(mean(tid_sklearn_local))
    #plot
    plt.plot(ks,tid_egen1,label='egen kmeans med fit')
    plt.plot(ks, tid_egen2, label='egen kmeans med fit2')
    plt.plot(ks, tid_sklearn, label='sklearns kmeans med fit_predict')
    plt.legend()
    plt.show()
