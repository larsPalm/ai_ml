#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import time
import random
from knn import PykNNClassifier

if __name__ == '__main__':
    from sklearn.datasets import make_blobs

    X_np, y_np = make_blobs(n_samples=200, centers=[[0, 0], [1, 2]],
                            n_features=2, random_state=2019)
    X1 = [(X_np[i, 0], X_np[i, 1]) for i in range(X_np.shape[0])]
    y1 = [y_np[i] for i in range(X_np.shape[0])]

    X_np, y_np = make_blobs(n_samples=200, centers=[[0, 0], [1, 2]],
                            n_features=2, random_state=2020)
    X2 = [(X_np[i, 0], X_np[i, 1]) for i in range(X_np.shape[0])]
    y2 = [y_np[i] for i in range(X_np.shape[0])]

    cls = PykNNClassifier(k=3)
    start = time.time()
    cls.fit(X1, y1)
    print(cls.accuracy(X2,y2))
    stop = time.time()
    print("uten:",stop-start)
    cls = PykNNClassifier(k=3)
    start = time.time()
    cls.fit(X1, y1)
    print(cls.accuracy_faster(X2, y2))
    stop = time.time()
    print("med sort_k:",stop - start)
    cls = PykNNClassifier(k=3)
    start = time.time()
    cls.fit(X1, y1)
    print(cls.accuracy_faster2(X2, y2))
    stop = time.time()
    print("med sort_k3:", stop - start)
    """x = [random.randint(0,300) for _ in range(100)]
    test_sort(x,3)"""
