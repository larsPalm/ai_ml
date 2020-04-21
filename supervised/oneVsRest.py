#!/usr/bin/env python3
import numpy as np
from MnnClassifier import add_bias
from MnnClassifier import logistic
"""brukte https://kenzotakahashi.github.io/logistic-regression-from-scratch-in-python.html
som inspirasjon for one vs rest"""


class OneVsRest:
    def fit(self, X_train, t_train, gamma=0.1, epochs=1000, diff=0.1):
        (k, m) = X_train.shape
        X_train = add_bias(X_train)
        #self.theta = theta = np.zeros(m + 1)
        self.thetas = []
        for c in np.unique(t_train):
            label = (t_train == c).astype(int)
            theta = np.zeros(m + 1)
            theta -= gamma / k * X_train.T @ (logistic(X_train @ theta) - label)
            for x in range(epochs):
                old_theta = theta.copy()
                theta -= gamma / k * X_train.T @ (logistic(X_train@theta) - label)
                new_theta = theta.copy()
                if np.linalg.norm((new_theta / np.linalg.norm(new_theta)) - (old_theta / np.linalg.norm(old_theta))) < diff:
                    break
            self.thetas.append((theta,c))

    def predict_one(self,x):
        return max((x.dot(w), c) for w, c in self.thetas)[1]

    def predict(self, X):
        return [self.predict_one(i) for i in np.insert(X, 0, 1, axis=1)]

    def accuracy(self, X_test, y_test, **kwargs):
        pred = self.predict(X_test)
        return sum(pred == y_test) / len(pred)
