#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from collections import Counter
import time
from MnnClassifier import add_bias
import random

"""koden er tatt fra løsningsforslaget til in3050 uke5
   og endret litt for å passe til oppgaven"""

def majority(a):
    counts = Counter(a)
    return counts.most_common()[0][0]

def distance_L2(a, b):
    "L2-distance using comprehension"
    s = sum((x - y) ** 2 for (x,y) in zip(a,b))
    return s ** 0.5


def sort_k(a,n,k):
    insertion(a,0,k-1)
    for i in range(k,n):
        if a[i][0] < a[k-1][0]:
            temp = a[k-1].copy()
            #print('k1f',a[k-1,:])
            a[k-1] = a[i].copy()
            #print('k1e', a[k - 1, :])
            #print('aif', a[i, :])
            a[i] = temp
            #print('aie',a[i,:])
            insertion2(a,k)


def insertion(a,v,h):
    for k in range(v,h):
        t = a[k+1].copy()
        i = k
        while i >= v and a[i][0] > t[0]:
            a[i+1] = a[i].copy()
            i -= 1
        a[i+1] = t


def insertion2(a,k):
    teller = k-1
    #print(teller)
    while teller > 0 and a[teller][0]<a[teller-1][0]:
        temp = a[teller-1].copy()
        #print(a[teller],a[teller-1])
        a[teller - 1] = a[teller].copy()
        a[teller] = temp
        #print(a[teller], a[teller - 1])
        teller = teller-1

def sort_k2(a,n,k):
    insertion1(a,0,k-1)
    for i in range(k,n):
        if a[i] < a[k-1]:
            temp = a[k - 1]
            a[k - 1] = a[i]
            a[i] = temp
            insertion21(a,k)


def insertion1(a,v,h):
    for k in range(v,h):
        t = a[k+1]
        i = k
        while i>=v and a[i]>t:
            a[i + 1] = a[i]
            i -= 1
        a[i+1] = t


def insertion21(a,k):
    teller = k-1
    #print(teller)
    while teller > 0 and a[teller]<a[teller-1]:
        temp = a[teller-1]
        #print(a[teller],a[teller-1])
        a[teller - 1] = a[teller]
        a[teller] = temp
        #print(a[teller], a[teller - 1])
        teller = teller-1


class PyClassifier():
    """Common methods to all python classifiers --- if any

    Nothing here yet"""


class PykNNClassifier(PyClassifier):
    """kNN classifier using pure python representations"""

    def __init__(self, k=3, dist=distance_L2):
        self.k = k
        self.dist = dist

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, a):
        X = self.X_train
        y = self.y_train
        distances = [(self.dist(a, b), b, c) for (b, c) in zip(X, y)]
        distances.sort()
        predictors = [c for (_, _, c) in distances[0: self.k]]
        return majority(predictors)

    def accuracy(self, X_test, y_test, **kwargs):
        """Calculate the accuracy of the classifier
        using the predict method"""
        predicted = [self.predict(a, **kwargs) for a in X_test]
        equal = len([(p, g) for (p, g) in zip(predicted, y_test) if p == g])
        return equal / len(y_test)

    def accuracy_faster(self, X_test, y_test, **kwargs):
        """Calculate the accuracy of the classifier
        using the predict method"""
        predicted = [self.predict_faster(a, **kwargs) for a in X_test]
        equal = len([(p, g) for (p, g) in zip(predicted, y_test) if p == g])
        return equal / len(y_test)

    def predict_faster(self, a):
        X = np.asarray(self.X_train)
        y = self.y_train
        distances = np.asarray([(self.dist(a, b), c) for (b, c) in zip(X, y)])
        #print(distances.shape)
        sort_k(distances,distances.shape[0],self.k)
        #print(distances[self.k+2,:])
        predictors = [c for (_,c) in distances[0: self.k]]
        #print(majority(predictors))
        return majority(predictors)

    def accuracy_faster2(self, X_test, y_test, **kwargs):
        """Calculate the accuracy of the classifier
        using the predict method"""
        predicted = [self.predict_faster2(a, **kwargs) for a in X_test]
        equal = len([(p, g) for (p, g) in zip(predicted, y_test) if p == g])
        return equal / len(y_test)

    def predict_faster2(self, a):
        X = np.asarray(self.X_train)
        y = self.y_train
        distances = [(self.dist(a, b), c) for (b, c) in zip(X, y)]
        #print(distances.shape)
        sort_k3(distances,len(distances),self.k)
        #print(distances[self.k+2,:])
        predictors = [c for (_,c) in distances[0: self.k]]
        #print(majority(predictors))
        return majority(predictors)

def test_sort(x,k):
    x1 = x.copy()
    x3 = x.copy()
    x2 = [[x[i],random.randint(1,3)] for i in range(len(x))]
    x4 = x2.copy()
    x2 = np.asarray(x2)
    print(x2.shape)
    x1.sort()
    print([x2[i,0] for i in range(k)])
    sort_k(x2,x2.shape[0],k)
    sort_k2(x3,len(x3),k)
    sort_k3(x4,len(x4),k)
    print(x)
    print('x1',[x1[i] for i in range(k)])
    print('x3',[x3[i] for i in range(k)])
    print('x2',[x2[j][0] for j in range(k)])
    print('x4', [x4[j][0] for j in range(k)])

def sort_k3(a,n,k):
    insertion3(a,0,k-1)
    for i in range(k,n):
        if a[i][0] < a[k-1][0]:
            temp = a[k-1]
            #print('k1f',a[k-1,:])
            a[k-1] = a[i]
            #print('k1e', a[k - 1, :])
            #print('aif', a[i, :])
            a[i] = temp
            #print('aie',a[i,:])
            insertion23(a,k)


def insertion3(a,v,h):
    for k in range(v,h):
        t = a[k+1]
        i = k
        while i >= v and a[i][0] > t[0]:
            a[i+1] = a[i]
            i -= 1
        a[i+1] = t


def insertion23(a,k):
    teller = k-1
    #print(teller)
    while teller > 0 and a[teller][0]<a[teller-1][0]:
        temp = a[teller-1]
        #print(a[teller],a[teller-1])
        a[teller - 1] = a[teller]
        a[teller] = temp
        #print(a[teller], a[teller - 1])
        teller = teller-1
