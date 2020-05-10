#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import syntheticdata
import random
from sklearn.cluster import KMeans
from pca import pca
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from numba import jit


class Kmeans_own:
    def __init__(self, k=3):
        self.k = k
        self.centers = [[] for _ in range(self.k)]
        self.clusters = []
        self.sample_in_cluster = []
        self.changed = True
        self.teller = 0

    def fit(self, X, epochs=50):
        """start = 0
        stop = int(X.shape[0]/self.k)
        for i in range(self.k):
            if i+1 == self.k:
                stop = X.shape[0]
            self.clusters.append(X[start:stop,:])
            start = stop
            stop = stop + int(X.shape[0]/self.k)"""
        self.clusters = [[] for _ in range(self.k)]
        for i in range(X.shape[0]):
            index = random.randint(0, self.k - 1)
            self.clusters[index].append(X[i])
            self.sample_in_cluster.append(index)
        for e in range(epochs):
            #beregn nye centers
            self.estimate_centers()
            #nullstill clusters
            self.reset_clusters()
            #legg til alle punkter på nytt i clusters
            self.make_clusters(X)
            if self.changed == False:
                break

    def fit2(self, X):
        self.clusters = [[] for _ in range(self.k)]
        for i in range(X.shape[0]):
            index = random.randint(0,self.k-1)
            self.clusters[index].append(X[i])
            self.sample_in_cluster.append(index)
        teller = 0
        while self.changed and teller < 50:
            #beregn nye centers
            self.estimate_centers()
            #nullstill clusters
            self.reset_clusters()
            #legg til alle punkter på nytt i clusters
            self.make_clusters(X)
            teller += 1
        #print(teller)

    """def make_clusters(self,X):
        for i in range(X.shape[0]):
            self.find_cluster(X[i])"""

    def make_clusters(self,X):
        self.changed = False
        for i in range(X.shape[0]):
            self.find_cluster(X[i],i)

    """def find_cluster(self,x):
        dist = [self.distance_L2(x, np.asarray(self.centers[ks])) for ks in range(self.k)]
        cluster_index = dist.index(min(dist))
        self.clusters[cluster_index].append(x)"""

    def find_cluster(self, x,i):
        dist = [self.distance_L2(x, np.asarray(self.centers[ks])) for ks in range(self.k)]
        cluster_index = dist.index(min(dist))
        if cluster_index != self.sample_in_cluster[i]:
            self.changed = True
            self.sample_in_cluster[i] = cluster_index
        self.clusters[cluster_index].append(x)

    def estimate_cluster(self,x):
        dist = [self.distance_L2(x, np.asarray(self.centers[ks])) for ks in range(self.k)]
        cluster_index = dist.index(min(dist))
        return cluster_index

    def estimate_centers(self):
        for x in range(self.k):
            cluster = np.asarray(self.clusters[x])
            if cluster.shape[0]>0:
                self.centers[x] = [np.mean(cluster[:, i]) for i in range(cluster.shape[1])]
        self.teller += 1

    def reset_clusters(self):
        self.clusters = [[] for _ in range(self.k)]

    def distance_L2(self, a, b):
        "L2-distance using comprehension"
        s = sum((x - y) ** 2 for (x, y) in zip(a, b))
        return s ** 0.5

    def predict(self, X):
        return [self.estimate_cluster(X[i]) for i in range(X.shape[0])]
