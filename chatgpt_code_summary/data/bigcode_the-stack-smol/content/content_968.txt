# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 08:32:13 2019

@author: Thiago
"""

import numpy as np
import pylab as pl

#%%

#Simulação de uma va
def va_estoque():
    p=np.array([0.1, 0.2, 0.6, 0.1])
    x=np.random.rand()

    if 0 < x <= p[0]:
        return 1
    elif p[0] < x <= p[0]+p[1]:
        return 2
    elif p[0]+p[1] < x <= p[0]+p[1]+p[2]:
        return 3
    elif p[0]+p[1]+p[2] < x <= 1.0:
        return 4
        
v = [va_estoque() for i in range(100000)]
pl.hist(v,)
pl.show()


#%%

#simulação estoque
M, T, estoque, lucro = 3, 3, 10, 0
R = 10000

for i in range(R):
    Y=va_estoque()
    lucro += 20*min(estoque, Y)
    estoque -= max(0, estoque-Y)
    lucro -= 5*estoque
    
    if estoque<M:
        estoque += T
        lucro -= 10*T
        
lucro /= R
print(M, T, lucro, estoque)


#%%

#simulação Urna de Ehrenfest

N, s = 100, []

for j in range(1000):
    v = [True for i in range(N)]
    for i in range(1000):
        k=np.random.choice(N)
        v[k] = not v[k]
        
    x = sum(v) / N
    s.append(x)
    
pl.hist(s)


#%%

#Lei dos grandes números

np.random.seed(0)

S = [1, 2, 3, 4, 5, 6]

n_vals = np.logspace(1, 5, num=200)
s=[]
for val in n_vals:
    np.random.seed(0)
    n = int(val)
    x = np.random.choice(S,n)
    p=sum(x==3)/n
    s.append([n,p])
    
s=np.array(s)
pl.semilogx(s[:,1])
pl.axhline(1./len(S),c='r')


#%%

#processos ergodicos

#%%

'''
s = 3000

for n in [1,2,3,5,10,50,100,200,400,1000]:
    z=np.zeros(s)
    for k in range(n):
        x = np.random.uniform(-1, 1, s)
        z+=x
        
    x = z/np.sqrt(n)
    pl.figure(n)
    sns.distplot(y, bins=12, rug=True)
    pl.title('N = ' + str())
'''