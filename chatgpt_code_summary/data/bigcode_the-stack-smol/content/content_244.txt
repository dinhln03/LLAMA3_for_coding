# module for distance computation;
import numpy as np

def dist(arraya, arrayb, mode):
    if mode == 0:
        dis = np.sum(np.abs(np.subtract(arraya, arrayb)))
    elif mode == 1:
        dis = np.sqrt(np.sum(np.power(np.subtract(arraya, arrayb), 2)))
    else:
        dis = 1 - np.dot(arraya, arrayb) / np.sqrt(np.sum(np.power(arraya, 2)) * np.sum(np.power(arrayb, 2)))
    return dis


def corr(arraya, arrayb, show):
    a = np.subtract(arraya, np.mean(arraya))
    b = np.subtract(arrayb, np.mean(arrayb))
    corr = np.sum(np.multiply(a, b)) / np.sqrt(np.multiply(np.sum(np.power(a, 2)), np.sum(np.power(b, 2)))) 
    return corr