# coding: utf-8
import numpy as np
from frequent_direction import FrequentDirection
from sklearn.preprocessing import normalize

from sklearn.metrics.pairwise import pairwise_kernels


def laplacian_sketch(X,ell,k,do_normalize_feature,normed,callback,**args):
    fd = FrequentDirection(ell,k)

    D = np.array([np.sum(callback(X,i,**args)) for i in range(len(X))])
    if normed:
        D = np.sqrt(D)
    isolation_mask = D==0

    if do_normalize_feature:
        # normalize original feature (for cosine distance)
        X[-isolation_mask] = normalize(X[-isolation_mask],norm='l2', axis=1, copy=False)
        D[:] = 1 # set 1 even to D==0 samples to avoid 0 division.

    for i,isolation in enumerate(isolation_mask):
        A_i = -1 * callback(X,i,**args)
        if normed:
            A_i /= D[i]
            A_i /= D
            A_i[i] = 1 - isolation  # set 0 to isolated node.
        else:
            A_i[i] = D[i]
        fd.add_sample(-A_i)
    return fd.get_result().T, D

def laplacian_sketch_rbf_kernel(X,ell,k,normed=True,gamma=None):
    return laplacian_sketch(X,ell,k,False,normed,one_row_rbf_kernel,gamma=None)

def laplacian_sketch_cosine_similarity(X,ell,k,normed=True):
    return laplacian_sketch(X,ell,k,True,normed,one_row_cosine_similarity)

def one_row_rbf_kernel(X,i,gamma=None):
    """
    X : array of shape (n_samples_X, n_features)
    i : target sample in X (X[i])
    gamma : float, default None
        If None, defaults to 1.0 / n_samples_X
    K(x, y) = exp(-gamma ||x-xi||^2)
    Returns
    -------
    kernel_matrix : array of shape (n_samples_X, n_samples_Y)
    """
    if gamma is None:
        gamma = 1.0 / X.shape[0]
    d = np.sum(np.power(X-X[i],2),axis=1)
    return np.array(np.exp(-gamma * d))


def one_row_cosine_similarity(X,i):
    """
    X : normalized matrix
    i : target sample in X
    """
    a = (np.dot(X,X[i].T)+1)/2
    a[a<0]=0
    return a

def debug_one_row_rbf_kernel(X,gamma=None):
    W = np.zeros((X.shape[0],X.shape[0]))
    W_gt = pairwise_kernels(X, metric='rbf',
                            filter_params=True,
                            gamma=gamma)
    for i,row in enumerate(X):
        W[i] = one_row_rbf_kernel(X,i,gamma=gamma)
    #print(W)
    #print(W_gt)
    #print(np.sum(W-W_gt))

def debug_one_row_cosine_similarity(X):
    W = np.zeros((X.shape[0],X.shape[0]))
    W_gt = pairwise_kernels(X, metric='cosine',
                            filter_params=True)
    for i,row in enumerate(X):
        W[i] = one_row_cosine_similarity(X,i)
    print(W)
    print(W_gt)
    print(np.sum(W-W_gt))
