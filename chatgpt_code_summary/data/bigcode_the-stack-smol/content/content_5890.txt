#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Single NN

@author: xuping
"""
import numpy as np
import scipy.io
#from threeNN import sigmoid

def layer_sizes(X, Y):
    n_in = X.shape[0]
    n_out = Y.shape[0]
    return(n_in, n_out)

def initialize_parameters(dim):
    np.random.seed(3)
    
    W = np.random.randn(dim, dim)*0.01
    b = np.zeros((dim, 1))
    return W,b

def prop(W,b,X,Y,lambd):
    m = X.shape[1]
    #forward
    A = sigmoid(np.dot(W, X) + b)
    cost = 1./m*np.sum(np.sum(np.square(A-Y),axis=0,keepdims=True)) + lambd/(2*m)*np.sum(np.sum(W*W))
    #cost = 1./m*np.sum(np.sum(np.square(A-Y)))
    #backward
    Z = np.dot(W, X) + b
    dZ = 2*(A-Y)*sigmoid(Z)*(1-sigmoid(Z))
    dW = 1./m*np.dot(dZ, X.T) + lambd/m*W
    #dW = 1./m*np.dot(dZ, X.T)
    db = 1./m*np.sum(dZ,axis=1,keepdims=True)
    
    grads = {"dW":dW, "db":db}
    return grads, cost

def nn_model(X,Y,num_iterations, lambd, learning_rate, print_cost=True):
    #np.random.seed(3)
    costs = []
    
    W, b = initialize_parameters(X.shape[0])
    
    for i in range(num_iterations):
        
        grads, cost = prop(W,b,X,Y,lambd)
        dW = grads["dW"]
        db = grads["db"]
        
        W = W-learning_rate*dW
        b = b-learning_rate*db
        
        if print_cost and i%1000==0:
            print("cost after iteration %i: %f" %(i, cost))
            costs.append(cost)
            
    parameters={"W":W, "b":b}
    grads={"dW":dW, "db":db}
    
    return parameters, costs

def predict(parameters, X):
    W=parameters["W"]
    b=parameters["b"]
    A = sigmoid(np.dot(W, X) + b)
    return A

def load_data():
    data=scipy.io.loadmat('U_Train.mat')
    X = data['ud']
    Y10 = data['tauR10']
    Y5 = data['tauR5']
    Y6 = data['tauR6']
    
    return X, Y5, Y6, Y10

if __name__ == "__main__":
    
    #load data
    X, Y5, Y6, Y10 = load_data()
    X5 = X[:5, :]
    X6 = X[:6, :]
    X10 = X[:10, :]
    
    num_iterations = 30000
    lambd = 10
    learning_rate = 3
    """
    X=X6
    Y=Y6
    np.random.seed(3)
    dim=X.shape[0]
    W = np.random.randn(dim, dim)*0.01
    b = np.zeros((dim, 1))
    Z = np.dot(W, X) + b
    A = sigmoid(Z)
    cost = A-Y
    #dZ = 2*(A-Y)*sigmoid(Z)*(1-sigmoid(Z))
    #dW = 1/m*np.dot(dZ, X.T)
    #db = 1/m*np.sum(dZ,axis=1,keepdims=True)
    """
    #parameters5, cost5 = nn_model(X5, Y5, num_iterations, lambd, learning_rate, print_cost=True)
    parameters6, cost6 = nn_model(X6, Y6, num_iterations, lambd, learning_rate, print_cost=True)
    #parameters10, cost10 = nn_model(X10, Y10, num_iterations, lambd, learning_rate, print_cost=True)
    
    #W5=parameters5["W"]
    #b5=parameters5["b"]
    W6=parameters6["W"]
    b6=parameters6["b"]
    #W10=parameters10["W"]
    #b10=parameters10["b"]
    
    #scipy.io.savemat('weights6.mat',{'W6':W6})
    #scipy.io.savemat('bias.mat',{'b6':b6})
    
    
