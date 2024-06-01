import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

def updateParams(k, alpha, N,sum_log_di, x, h):

    div_xByAlpha = np.divide(x,alpha)
    powK_div_xByAlpha = np.power(div_xByAlpha, k)
    log_div_xByAlpha = np.log(div_xByAlpha)

    sum_powK_div_diByAlpha = np.sum(np.multiply(powK_div_xByAlpha, h))
    sum_prod_OF_powK_div_diByAlpha_AND_log_div_diByAlpha = np.sum(np.multiply(np.multiply(powK_div_xByAlpha,log_div_xByAlpha),h))
    sum_prod_OF_powK_div_diByAlpha_AND_logP2_div_diByAlpha = np.sum(np.multiply(np.multiply(powK_div_xByAlpha,np.power(log_div_xByAlpha,2)),h))

    #N = d.shape[0]

    hessian = np.zeros((2,2))
    hessian[0,0] = -1.0 * ((N/(k*k)) + sum_prod_OF_powK_div_diByAlpha_AND_logP2_div_diByAlpha)
    hessian[1,1] = (k/(alpha*alpha)) * (N-(k+1)*sum_powK_div_diByAlpha)
    hessian[0,1] = hessian[1,0] = (1.0/alpha)*sum_powK_div_diByAlpha + (k/alpha)*sum_prod_OF_powK_div_diByAlpha_AND_log_div_diByAlpha - N/alpha

    vec = np.zeros((2,1))
    vec[0] = -1.0 *( N/k - N*np.log(alpha) + sum_log_di - sum_prod_OF_powK_div_diByAlpha_AND_log_div_diByAlpha)
    vec[1] = -1.0 *(k/alpha * (sum_powK_div_diByAlpha - N))

    param = np.linalg.inv(hessian).dot(vec)
    return k+param[0], alpha+param[1]

if __name__ == "__main__":
    #loading histograms
    data = np.loadtxt('myspace.csv',dtype=np.object,comments='#',delimiter=',')
    h = data[:,1].astype(np.int)
    h = np.array([x for x in h if x>0])
    x = np.array([num for num in range(1, h.shape[0]+1)])

    k = 1.0
    alpha = 1.0

    N = np.sum(h)
    sum_log_di = np.sum(np.multiply(np.log(x), h))
    for i in range(0,20):
        k,alpha = updateParams(k, alpha, N, sum_log_di, x, h)
        print i
        print k
        print alpha
        print "________"

    x_1 = np.linspace(1,500,2500)
    fig = plt.figure()
    axs = fig.add_subplot(111)
    y = N * (k/alpha) * np.multiply(np.power(np.divide(x_1,alpha), k-1), np.exp(-1.0* np.power(np.divide(x_1,alpha), k)))
    axs.plot(x_1,y, 'b')
    axs.plot(x, h, 'g')
    plt.show()
