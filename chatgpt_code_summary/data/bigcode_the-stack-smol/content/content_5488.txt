import random as rd
import time as t
def Seed():
    rd.seed(int(str(t.time()).split(".")[0]))
Seed()
def Random():
    return rd.gauss(0,0.01)
def RandomZeroMask(Prob=0.1):
    r= rd.random()
    if r<Prob:
        return 0.0
    else:
        return 1.0