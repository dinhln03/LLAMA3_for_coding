# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:25:43 2020

@author: greg6
"""

import numpy as np

t = [i for i in range(3)]
lam = [100+i*10 for i in range(2)]
com = ["A","B","C"]

S = dict()
for l in lam:
    for u,c in enumerate(com):
        S[(l,c)] = l+0.1*u

C = dict()
for i in t:
    for u,c in enumerate(com):
        C[(i,c)] = (i+0.1*u)

nt = len(t)
nw = len(lam)
nc = len(com)
nparams = 2

nd = nw*nt
ntheta = nc*(nw+nt)+nparams

B_matrix = np.zeros((ntheta,nw*nt))
for i, t in enumerate(t):
    for j, l in enumerate(lam):
        for k, c in enumerate(com):
            # r_idx1 = k*nt+i
            r_idx1 = i * nc + k
            r_idx2 = j * nc + k + nc * nt
            # r_idx2 = j * nc + k + nc * nw
            # c_idx = i+j*nt
            c_idx = i * nw + j
            # print(j, k, r_idx2)
            B_matrix[r_idx1, c_idx] = S[l, c]
            # try:
            B_matrix[r_idx2, c_idx] = C[t, c]