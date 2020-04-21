#!/usr/bin/env python3
import numpy as np

def add_bias(X):
    # Put bias in position 0
    sh = X.shape
    if len(sh) == 1:
        #X is a vector
        return np.concatenate([np.array([1]), X])
    else:
        # X is a matrix
        m = sh[0]
        bias = np.ones((m,1)) # Makes a m*1 matrix of 1-s
        return np.concatenate([bias, X], axis  = 1)

def logistic(x):
    return 1/(1+np.exp(-x))


class MNNClassifier():
    """A multi-layer neural network with one hidden layer"""

    def __init__(self, eta=0.001, dim_hidden=60):
        """Intialize the hyperparameters"""
        self.eta = eta
        self.dim_hidden = dim_hidden
        self.dim_out = 0
        self.dim_in = 0
        self.weights1 = None
        self.weights2 = None

        # Should you put additional code here?

    def fit(self, X_train, t_train, epochs=1000):
        """Intialize the weights. Train *epochs* many epochs."""
        self.dim_in = X_train.shape[1]
        self.dim_out = len(np.unique(t_train))
        self.weights1 = np.random.uniform(-1, 1, (self.dim_in + 1) * self.dim_hidden).reshape((self.dim_in + 1, self.dim_hidden))
        self.weights1[0,:] = 1
        self.weights2 = np.random.uniform(-1, 1, (self.dim_hidden + 1) * self.dim_out).reshape((self.dim_hidden + 1, self.dim_out))
        self.weights2[0,:] = 1
        fasit = np.zeros((len(t_train), self.dim_out))
        for x in range(len(t_train)):
            fasit[x, t_train[x]] = 1
        # Initilaization
        # Fill in code for initalization
        """print(self.weights2)
        dummy = self.weights1.copy()"""
        for e in range(epochs):
            # Run one epoch of forward-backward
            # Fill in the code
            (hidden_output,output) = self.forward(X_train)
            self.update(hidden_output,output,X_train,fasit)
        """print(self.weights2)
        print(dummy)
        print(self.weights1)"""

    def forward(self, X):
        """Perform one forward step.
        Return a pair consisting of the outputs of the hidden_layer
        and the outputs on the final layer"""
        X_plus = add_bias(X)
        hidden_output = np.asarray(logistic(X_plus@self.weights1))
        hidden_output_plus = add_bias(hidden_output)
        output = np.asarray(logistic(hidden_output_plus@self.weights2))
        return hidden_output,output

    def update(self, hidden_output, output,X, fasit):
        error_output = np.asarray([(fasit[:,x]-output[:,x])*output[:,x]*(1-(output[:,x])) for x in range(self.dim_out)]).T
        hidden_output_plus = add_bias(hidden_output)
        X_plus = add_bias(X)
        # updating the weights on the output nodes
        for i in range(1,self.weights2.shape[0]):
            for j in range(error_output.shape[1]):
                self.weights2[i,j] += self.eta*sum(hidden_output_plus[:,i]*error_output[:,j])
        # updating the weights on the hidden layer
        for i in range(1,self.weights1.shape[0]):
            delta_js = np.asarray(error_output)
            for l in range(hidden_output.shape[1]):
                sum_delta = sum([self.weights2[l,j]*delta_js[:,j] for j in range(error_output.shape[1])])
                delta_l = hidden_output[:,l]*(1-hidden_output[:,l])*sum_delta
                self.weights1[i,l] += self.eta*delta_l@X_plus[:,i]

    def accuracy(self, X_test, t_test):
        """Calculate the accuracy of the classifier on the pair (X_test, t_test)
        Return the accuracy"""
        # Fill in the code
        hidden,output = self.forward(X_test)

        pred = [np.argmax(output[x,:],axis=0) for x in range(output.shape[0])]
        return sum(pred == t_test) / len(pred)

    def predict(self,X):
        hidden,output = self.forward(X)
        return [np.argmax(output[x,:],axis=0) for x in range(output.shape[0])]
