""" converted from Matlab code
source: http://www.robots.ox.ac.uk/~fwood/teaching/AIMS_CDT_ML_2015/homework/HW_2_em/
"""

import numpy as np

def m_step_gaussian_mixture(data, gamma):
    """% Performs the M-step of the EM algorithm for gaussain mixture model.
    %
    % @param data   : n x d matrix with rows as d dimensional data points
    % @param gamma  : n x k matrix of resposibilities
    %
    % @return pi    : k x 1 array
    % @return mu    : k x d matrix of maximized cluster centers
    % @return sigma : cell array of maximized 
    %
    """
    
    n = np.shape(data)[0]
    d = np.shape(data)[1]
    k = np.shape(gamma)[1]
    
    pi = np.zeros(k)
    mu = np.zeros((k, d))
    sigma = np.zeros((k, d, d))
    
    for kk in range(k):
        Nkk = np.sum(gamma[:, kk])
        pi[kk] = Nkk / n
        for dd in range(d):
            mu[kk, dd] = np.sum(gamma[:, kk] * data[:, dd], axis=0) / Nkk
        
    for kk in range(k):
        Nkk = np.sum(gamma[:, kk])
        centered_data = data - mu[kk, :]
        for nn in range(n):
            sigma[kk] += gamma[nn, kk] * np.dot(centered_data[nn, None].T, centered_data[nn, None])
        sigma[kk] /= Nkk
        
    return [mu, sigma, pi]
