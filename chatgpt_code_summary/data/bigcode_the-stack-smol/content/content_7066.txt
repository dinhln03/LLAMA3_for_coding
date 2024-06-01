import theano.tensor as T
import theano
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
import random
import time

'''
    Sample code to reproduce our results for the Bayesian neural network example.
    Our settings are almost the same as Hernandez-Lobato and Adams (ICML15) https://jmhldotorg.files.wordpress.com/2015/05/pbp-icml2015.pdf
    Our implementation is also based on their Python code.

    p(y | W, X, \gamma) = \prod_i^N  N(y_i | f(x_i; W), \gamma^{-1})
    p(W | \lambda) = \prod_i N(w_i | 0, \lambda^{-1})
    p(\gamma) = Gamma(\gamma | a0, b0)
    p(\lambda) = Gamma(\lambda | a0, b0)

    The posterior distribution is as follows:
    p(W, \gamma, \lambda) = p(y | W, X, \gamma) p(W | \lambda) p(\gamma) p(\lambda)
    To avoid negative values of \gamma and \lambda, we update loggamma and loglambda instead.

    Copyright (c) 2016,  Qiang Liu & Dilin Wang
    All rights reserved.
'''

class svgd_bayesnn:

    '''
        We define a one-hidden-layer-neural-network specifically. We leave extension of deep neural network as our future work.

        Input
            -- X_train: training dataset, features
            -- y_train: training labels
            -- batch_size: sub-sampling batch size
            -- max_iter: maximum iterations for the training procedure
            -- M: number of particles are used to fit the posterior distribution
            -- n_hidden: number of hidden units
            -- a0, b0: hyper-parameters of Gamma distribution
            -- master_stepsize, auto_corr: parameters of adgrad
    '''
    def __init__(self, X_train, y_train,  X_test, y_text, batch_size = 100, max_iter = 1000, M = 20, n_hidden = 50,
        a0 = 1, b0 = 0.1, master_stepsize = 1e-3, auto_corr = 0.9, h=-1, alpha = 0.9,
        method = 'none',m=5, cf = False, uStat = True, regCoeff = 0.1, adver = False, adverMaxIter = 5,
        maxTime = 20, numTimeSteps = 20):
        self.n_hidden = n_hidden
        self.d = X_train.shape[1]   # number of data, dimension
        self.M = M

        num_vars = self.d * n_hidden + n_hidden * 2 + 3  # w1: d*n_hidden; b1: n_hidden; w2 = n_hidden; b2 = 1; 2 variances
        self.theta = np.zeros([self.M, num_vars])  # particles, will be initialized later

        '''
            We keep the last 10% (maximum 500) of training data points for model developing
        '''
        size_dev = min(int(np.round(0.1 * X_train.shape[0])), 500)
        X_dev, y_dev = X_train[-size_dev:], y_train[-size_dev:]
        X_train, y_train = X_train[:-size_dev], y_train[:-size_dev]

        '''
            The data sets are normalized so that the input features and the targets have zero mean and unit variance
        '''
        self.std_X_train = np.std(X_train, 0)
        self.std_X_train[ self.std_X_train == 0 ] = 1
        self.mean_X_train = np.mean(X_train, 0)

        self.mean_y_train = np.mean(y_train)
        self.std_y_train = np.std(y_train)

        '''
            Theano symbolic variables
            Define the neural network here
        '''
        X = T.matrix('X') # Feature matrix
        y = T.vector('y') # labels

        w_1 = T.matrix('w_1') # weights between input layer and hidden layer
        b_1 = T.vector('b_1') # bias vector of hidden layer
        w_2 = T.vector('w_2') # weights between hidden layer and output layer
        b_2 = T.scalar('b_2') # bias of output

        N = T.scalar('N') # number of observations

        log_gamma = T.scalar('log_gamma')   # variances related parameters
        log_lambda = T.scalar('log_lambda')

        ###
        prediction = T.dot(T.nnet.relu(T.dot(X, w_1)+b_1), w_2) + b_2

        ''' define the log posterior distribution '''
        log_lik_data = -0.5 * X.shape[0] * (T.log(2*np.pi) - log_gamma) - (T.exp(log_gamma)/2) * T.sum(T.power(prediction - y, 2))
        log_prior_data = (a0 - 1) * log_gamma - b0 * T.exp(log_gamma) + log_gamma
        log_prior_w = -0.5 * (num_vars-2) * (T.log(2*np.pi)-log_lambda) - (T.exp(log_lambda)/2)*((w_1**2).sum() + (w_2**2).sum() + (b_1**2).sum() + b_2**2)  \
                       + (a0-1) * log_lambda - b0 * T.exp(log_lambda) + log_lambda

        # sub-sampling mini-batches of data, where (X, y) is the batch data, and N is the number of whole observations
        log_posterior = (log_lik_data * N / X.shape[0] + log_prior_data + log_prior_w)
        dw_1, db_1, dw_2, db_2, d_log_gamma, d_log_lambda = T.grad(log_posterior, [w_1, b_1, w_2, b_2, log_gamma, log_lambda])

        # automatic gradient
        logp_gradient = theano.function(
             inputs = [X, y, w_1, b_1, w_2, b_2, log_gamma, log_lambda, N],
             outputs = [dw_1, db_1, dw_2, db_2, d_log_gamma, d_log_lambda]
        )

        # prediction function
        self.nn_predict = theano.function(inputs = [X, w_1, b_1, w_2, b_2], outputs = prediction)

        '''
            Training with SVGD
        '''
        # normalization
        X_train, y_train = self.normalization(X_train, y_train)
        N0 = X_train.shape[0]  # number of observations

        ''' initializing all particles '''
        for i in range(self.M):
            w1, b1, w2, b2, loggamma, loglambda = self.init_weights(a0, b0)
            # use better initialization for gamma
            ridx = np.random.choice(range(X_train.shape[0]), \
                                           np.min([X_train.shape[0], 1000]), replace = False)
            y_hat = self.nn_predict(X_train[ridx,:], w1, b1, w2, b2)
            loggamma = -np.log(np.mean(np.power(y_hat - y_train[ridx], 2)))
            self.theta[i,:] = self.pack_weights(w1, b1, w2, b2, loggamma, loglambda)

        grad_theta = np.zeros([self.M, num_vars])  # gradient
        # adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = 0
        self.y_historical_grad = 0
        self.h_historical_grad = 0

        self.rmse_overTime = np.zeros(numTimeSteps)  # RMSE
        self.llh_overTime = np.zeros(numTimeSteps)  # LLH
        self.iter_overTime = np.zeros(numTimeSteps)  # LLH
        timeStepUnit = maxTime / numTimeSteps # Time to check every iteration
        timeInd = 0;

        start_time = time.time()
        for iter in range(max_iter):
            if method == 'subparticles':
                self.Sqy = np.zeros([m, num_vars]) # Sqy
            elif method == 'inducedPoints' or method == 'none':
                self.Sqx = np.zeros([self.M, num_vars]) # Sqx
            h = -1;
            # sub-sampling
            batch = [ i % N0 for i in range(iter * batch_size, (iter + 1) * batch_size) ]

            if method == 'none' or method =='inducedPoints':
                for i in range(self.M):
                    w1, b1, w2, b2, loggamma, loglambda = self.unpack_weights(self.theta[i,:])
                    dw1, db1, dw2, db2, dloggamma, dloglambda = logp_gradient(X_train[batch,:], y_train[batch], w1, b1, w2, b2, loggamma, loglambda, N0)
                    self.Sqx[i,:] = self.pack_weights(dw1, db1, dw2, db2, dloggamma, dloglambda)

                if method == 'none':
                    grad_theta = self.svgd_kernel(h=h)
                elif method == 'inducedPoints':
                    self.yInd = np.random.choice(self.theta.shape[0], m, replace=False)
                    self.y = self.theta[self.yInd]
                    grad_theta = self.svgd_kernel_inducedPoints(h=h, uStat = uStat, regCoeff = regCoeff, adver=adver, adverMaxIter = adverMaxIter)

            elif method == 'subparticles':
                self.yInd = np.random.choice(self.theta.shape[0], m, replace=False)
                self.y = self.theta[self.yInd]

                for i in range(m):
                    w1, b1, w2, b2, loggamma, loglambda = self.unpack_weights(self.y[i,:])
                    dw1, db1, dw2, db2, dloggamma, dloglambda = logp_gradient(X_train[batch,:], y_train[batch], w1, b1, w2, b2, loggamma, loglambda, N0)
                    self.Sqy[i,:] = self.pack_weights(dw1, db1, dw2, db2, dloggamma, dloglambda)

                grad_theta = self.svgd_kernel_subset(h=-1, cf=cf)

            [adj_grad, historical_grad] = self.get_adamUpdate(iter, grad_theta, historical_grad,master_stepsize, alpha, fudge_factor)
            self.theta = self.theta + adj_grad;
            elapsed_time = time.time() - start_time

            if elapsed_time > timeStepUnit:
                self.thetaCopy = np.copy(self.theta)

                # Evaluate and save
                '''
                    Model selection by using a development set
                '''
                X_dev = self.normalization(X_dev)
                for i in range(self.M):
                    w1, b1, w2, b2, loggamma, loglambda = self.unpack_weights(self.thetaCopy[i, :])
                    pred_y_dev = self.nn_predict(X_dev, w1, b1, w2, b2) * self.std_y_train + self.mean_y_train
                    # likelihood
                    def f_log_lik(loggamma): return np.sum(  np.log(np.sqrt(np.exp(loggamma)) /np.sqrt(2*np.pi) * np.exp( -1 * (np.power(pred_y_dev - y_dev, 2) / 2) * np.exp(loggamma) )) )
                    # The higher probability is better
                    lik1 = f_log_lik(loggamma)
                    # one heuristic setting
                    loggamma = -np.log(np.mean(np.power(pred_y_dev - y_dev, 2)))
                    lik2 = f_log_lik(loggamma)
                    if lik2 > lik1:
                        self.thetaCopy[i,-2] = loggamma  # update loggamma

                svgd_rmse, svgd_ll = self.evaluation(X_test, y_test)
                self.rmse_overTime[timeInd] = svgd_rmse
                self.llh_overTime[timeInd] = svgd_ll
                self.iter_overTime[timeInd] = iter

                start_time = time.time()
                timeInd = timeInd + 1


                # Break after maxTime
                if timeInd >= numTimeSteps:
                    print('Reached ', iter, 'iterations\n')
                    break


    def normalization(self, X, y = None):
        X = (X - np.full(X.shape, self.mean_X_train)) / \
            np.full(X.shape, self.std_X_train)

        if y is not None:
            y = (y - self.mean_y_train) / self.std_y_train
            return (X, y)
        else:
            return X

    '''
        Initialize all particles
    '''
    def init_weights(self, a0, b0):
        w1 = 1.0 / np.sqrt(self.d + 1) * np.random.randn(self.d, self.n_hidden)
        b1 = np.zeros((self.n_hidden,))
        w2 = 1.0 / np.sqrt(self.n_hidden + 1) * np.random.randn(self.n_hidden)
        b2 = 0.
        loggamma = np.log(np.random.gamma(a0, b0))
        loglambda = np.log(np.random.gamma(a0, b0))
        return (w1, b1, w2, b2, loggamma, loglambda)

    '''
        Returns control functional weights
    '''
    def getWeights(self, KpMat):
        condNumber = self.getConditionNumber(KpMat)
        z = KpMat.shape[0]

        # Get weights
        KPrime = KpMat + condNumber * z * np.identity(z)
        num = np.matmul(np.ones(z),np.linalg.inv(KPrime))
        denom = 1 + np.matmul(num,np.ones(z))
        weights = num / denom

        weights = weights / sum(weights)

        return (weights)

    '''
        Given a kernel matrix K, let lambda be smallest power of 10 such that
        kernel matrix K0 + lamba*I has condition number lower than 10^10
        Note we use 2-norm for computing condition number
    '''
    def getConditionNumber(self, K):
        condNumber = 10e-10
        condA = 10e11
        matSize = K.shape[0]
        while condA > 10e10:
            condNumber = condNumber * 10
            A = K + condNumber * np.identity(matSize)
            condA = np.linalg.norm(A, ord=2) * np.linalg.norm(np.linalg.inv(A), ord=2)
        return (condNumber)

    '''
        Calculate kernel matrix and its gradient: K, \nabla_x k
    '''
    def svgd_kernel(self, h = -1):
        n,d = self.theta.shape
        sq_dist = pdist(self.theta)
        pairwise_dists = squareform(sq_dist)**2
        if h < 0: # if h < 0, using median trick
            h = np.median(pairwise_dists)
            h = np.sqrt(0.5 * h / np.log(n+1))

        # compute the rbf kernel
        Kxy = np.exp( -pairwise_dists / h**2 / 2)

        dxkxy = -np.matmul(Kxy, self.theta)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(d):
            dxkxy[:, i] = dxkxy[:,i] + np.multiply(self.theta[:,i],sumkxy)
        dxkxy = dxkxy / (h**2)

        grad_theta = (np.matmul(Kxy, self.Sqx) + dxkxy) / n

        return grad_theta

    '''
        Compute gradient update for theta using svgd random subset (with optional control functional)
    '''
    def svgd_kernel_subset(self, h=-1, cf = False):
        n,d = self.theta.shape
        m = self.y.shape[0]



        pairwise_dists = cdist(self.theta, self.y)**2

        if h < 0: # if h < 0, using median trick
            h = np.median(pairwise_dists)
            h = np.sqrt(0.5 * h / np.log(n+1))

        # compute the rbf kernel
        Kxy = np.exp( -pairwise_dists / h**2 / 2)

        if cf == True : # Using control functional
            sqxdy_part = np.array([np.sum(np.multiply(self.Sqy,self.y),axis=1),]*m).T
            sqxdy = -(np.matmul(self.Sqy,self.y.T)- sqxdy_part)/ h**2
            dxsqy = sqxdy.T
            dxdy = -pairwise_dists[self.yInd]/h**4 +d/h**2
            KxySub = Kxy[self.yInd]

            KpMat = (np.matmul(self.Sqy, self.Sqy.T) + sqxdy + dxsqy + dxdy)
            KpMat = np.multiply(KpMat, KxySub)

            weights = self.getWeights(KpMat)
            Kxy = np.multiply(Kxy, np.matlib.repmat(weights, n, 1))

        dxkxy = -np.matmul(Kxy, self.y)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(d):
            dxkxy[:, i] = dxkxy[:,i] + np.multiply(self.theta[:,i],sumkxy)
        dxkxy = dxkxy / (h**2)

        grad_theta = (np.matmul(Kxy, self.Sqy) + dxkxy)
        if cf == False:
            grad_theta = grad_theta / m

        return grad_theta

    '''
        Perform a step of adam update
    '''
    def get_adamUpdate(self, iterInd, ori_grad, hist_grad, stepsize = 1e-3, alpha = 0.9, fudge_factor = 1e-6):
        if iterInd == 0:
            hist_grad = hist_grad + ori_grad ** 2
        else:
            hist_grad = alpha * hist_grad + (1 - alpha) * (ori_grad ** 2)

        adj_grad = np.divide(ori_grad, fudge_factor+np.sqrt(hist_grad))

        return (stepsize * adj_grad, hist_grad)

    '''
        Compute gradient update for y
    '''
    def svgd_kernel_grady(self, h=-1, uStat=True, regCoeff=0.1):
        m = self.y.shape[0]
        xAdverSubsetInd = np.random.choice(self.theta.shape[0], m, replace=False)
        self.thetaSubset = self.theta[xAdverSubsetInd,:]
        self.SqxSubset = self.Sqx[xAdverSubsetInd,:]

        #self.thetaSubset = np.copy(self.theta)
        #self.SqxSubset = np.copy(self.Sqx)
        n,d = self.thetaSubset.shape


        pairwise_dists = cdist(self.thetaSubset, self.y)**2

        if h < 0: # if h < 0, using median trick
            h = np.median(pairwise_dists)
            h = np.sqrt(0.5 * h / np.log(n+1))

        # compute the rbf kernel
        Kxy = np.exp( -pairwise_dists / h**2 / 2)

        yGrad = np.zeros((m,d));

        # Compute gradient
        for yInd in range(m):
            Kxy_cur = Kxy[:,yInd];
            xmy = (self.thetaSubset - np.tile(self.y[yInd,:],[n,1]))/h**2
            Sqxxmy = self.SqxSubset - xmy;
            back = np.tile(np.array([Kxy_cur]).T,(1,d)) * Sqxxmy
            inner = np.tile(np.array([np.sum(np.matmul(back, back.T),axis=1)]).T,[1,d])
            yGrad[yInd,:] = np.sum(xmy * inner,axis=0) + np.sum(back,axis=0) * np.sum(Kxy_cur)/h**2

            # For U-statistic
            if uStat:
                front_u = np.tile(np.array([(Kxy_cur**2) * np.sum(Sqxxmy **2,axis=1)]).T,[1,d]) * xmy;
                back_u = np.tile(np.array([Kxy_cur**2 / h**2]).T,[1,d]) * Sqxxmy

                yGrad[yInd,:] = yGrad[yInd,:] - np.sum(front_u + back_u,axis=0)

        if uStat:
            yGrad = yGrad * 2 / (n*(n-1)*m);
        else:
            yGrad = yGrad * 2 / (n**2 * m);

        if regCoeff > 0 :
            H_y = cdist(self.y, self.y)**2
            Kxy_y = np.exp( -H_y / h**2 / 2)
            sumKxy_y = np.sum(Kxy_y,axis=1)
            yReg = (self.y * np.tile(np.array([sumKxy_y]).T,[1,d]) - np.matmul(Kxy_y,self.y))/(h**2 * m)

        yGrad = yGrad + regCoeff * yReg
        return (yGrad)

    '''
        Compute gradient update for h
    '''
    def svgd_kernel_gradh(self, h=-1, uStat=True):
        n,d = self.thetaSubset.shape
        m = self.y.shape[0]

        H = cdist(self.thetaSubset, self.y)**2

        if h < 0: # if h < 0, using median trick
            h = np.median(H)
            h = np.sqrt(0.5 * h / np.log(n+1))

        # compute the rbf kernel
        Kxy = np.exp( -H / h**2 / 2)

        hGrad = 0;

        # For each induced point
        for yInd in range(m):
            Kxy_cur = Kxy[:,yInd]
            H_cur = H[:,yInd]
            xmy = (self.thetaSubset - np.tile(self.y[yInd,:],[n,1]))/h**2
            Sqxxmy = self.SqxSubset - xmy

            part2 = np.tile(np.array([Kxy_cur]).T,[1,d]) * Sqxxmy
            part1_1 = np.tile(np.array([H_cur/h**3]).T,[1,d]) * part2
            part1_2 = np.tile(np.array([Kxy_cur]).T,[1,d]) * (2*xmy / h**3)
            part = np.matmul(part1_1 + part1_2, part2.T)
            hGrad = hGrad + np.sum(np.sum(part,axis=1))

            if uStat:
                front_u = (Kxy_cur**2) * (H_cur/h**3) * np.sum(Sqxxmy**2, axis=1)
                back_u = np.sum((2*xmy/h**3) * Sqxxmy,axis=1)
                hGrad = hGrad - np.sum(Kxy_cur**2 * (front_u + back_u),axis=0)

        if uStat:
            hGrad = hGrad * 2 / (n*(n-1)*m);
        else:
            hGrad = hGrad * 2 / (n**2 * m);

        return (hGrad)

    '''
        Induced Points Method
    '''
    def svgd_kernel_inducedPoints(self, h=-1, uStat=True, regCoeff=0.1, adver = False, adverMaxIter = 5, stepsize = 1e-3, alpha = 0.9):

        n,d = self.theta.shape
        m = self.y.shape[0]

        # If we want to perform EM
        if adver == True:
            # Perform update emMaxIter number of times
            fudge_factor = 1e-6

            for adverIter in range(0,adverMaxIter):
                grad_y = self.svgd_kernel_grady(h=h,uStat=uStat, regCoeff=regCoeff)
                [update_y,hist_grad] = self.get_adamUpdate(adverIter, grad_y, self.y_historical_grad,stepsize = stepsize, alpha = alpha)
                self.y = self.y + update_y
                self.y_historical_grad = hist_grad

                grad_h = self.svgd_kernel_gradh(h=h,uStat=uStat)
                [update_h, hist_grad] = self.get_adamUpdate(adverIter, grad_h, self.h_historical_grad,stepsize = stepsize, alpha = alpha)
                h = h + update_h
                self.h_historical_grad = hist_grad

        pairwise_dists = cdist(self.theta, self.y)**2

        # compute the rbf kernel
        Kxy = np.exp( -pairwise_dists / h**2 / 2)

        innerTerm_1 = np.matmul(Kxy.T, (self.Sqx - self.theta/ h**2))
        sumkxy = np.sum(Kxy, axis=0)
        innerTerm_2 = np.multiply(np.tile(np.array([sumkxy]).T,(1,d)), self.y/h**2)
        innerTerm = (innerTerm_1 + innerTerm_2)/n

        gradTheta = np.matmul(Kxy, innerTerm)/m
        return (gradTheta)

    '''
        Pack all parameters in our model
    '''
    def pack_weights(self, w1, b1, w2, b2, loggamma, loglambda):
        params = np.concatenate([w1.flatten(), b1, w2, [b2], [loggamma],[loglambda]])
        return params

    '''
        Unpack all parameters in our model
    '''
    def unpack_weights(self, z):
        w = z
        w1 = np.reshape(w[:self.d*self.n_hidden], [self.d, self.n_hidden])
        b1 = w[self.d*self.n_hidden:(self.d+1)*self.n_hidden]

        w = w[(self.d+1)*self.n_hidden:]
        w2, b2 = w[:self.n_hidden], w[-3]

        # the last two parameters are log variance
        loggamma, loglambda= w[-2], w[-1]

        return (w1, b1, w2, b2, loggamma, loglambda)


    '''
        Evaluating testing rmse and log-likelihood, which is the same as in PBP
        Input:
            -- X_test: unnormalized testing feature set
            -- y_test: unnormalized testing labels
    '''
    def evaluation(self, X_test, y_test):
        # normalization
        X_test = self.normalization(X_test)

        # average over the output
        pred_y_test = np.zeros([self.M, len(y_test)])
        prob = np.zeros([self.M, len(y_test)])

        '''
            Since we have M particles, we use a Bayesian view to calculate rmse and log-likelihood
        '''
        for i in range(self.M):
            w1, b1, w2, b2, loggamma, loglambda = self.unpack_weights(self.thetaCopy[i, :])
            pred_y_test[i, :] = self.nn_predict(X_test, w1, b1, w2, b2) * self.std_y_train + self.mean_y_train
            prob[i, :] = np.sqrt(np.exp(loggamma)) /np.sqrt(2*np.pi) * np.exp( -1 * (np.power(pred_y_test[i, :] - y_test, 2) / 2) * np.exp(loggamma) )
        pred = np.mean(pred_y_test, axis=0)

        # evaluation
        svgd_rmse = np.sqrt(np.mean((pred - y_test)**2))
        svgd_ll = np.mean(np.log(np.mean(prob, axis = 0)))

        return (svgd_rmse, svgd_ll)

    '''
        Returns the result of the iterations
    '''
    def getResults(self):
        return (self.rmse_overTime, self.llh_overTime, self.iter_overTime)

if __name__ == '__main__':

    print ('Theano', theano.version.version)    #our implementation is based on theano 0.8.2

    np.random.seed(1)
    ''' load data file '''

    for dataInd in range(0,1):
        if dataInd == 0:
            data = np.loadtxt('../data/boston_housing')
            datasetName = 'Boston Housing'
        elif dataInd == 1:
            data = np.loadtxt(open("../data/Concrete.csv", "rb"), delimiter=",", skiprows=1) # Concrete
            datasetName = 'Concrete'
        elif dataInd == 2:
            data = np.loadtxt(open("../data/Energy.csv", "rb"), delimiter=",", skiprows=1) # Energy
            datasetName = 'Energy'
        elif dataInd == 3:
            data = np.loadtxt(open("../data/kin8nm.csv", "rb"), delimiter=",", skiprows=0) # Kin8nm Dataset
            datasetName = 'Kin8nm'
        print('-------------------',datasetName,'-------------------')

        if dataInd == 2:
            X_input = data[ :, range(data.shape[ 1 ] - 2) ]
            y_input = data[ :, data.shape[ 1 ] - 2 ]
        else:
            # Please make sure that the last column is the label and the other columns are features
            X_input = data[ :, range(data.shape[ 1 ] - 1) ]
            y_input = data[ :, data.shape[ 1 ] - 1 ]

        ''' build the training and testing data set'''
        train_ratio = 0.9 # We create the train and test sets with 90% and 10% of the data
        permutation = np.arange(X_input.shape[0])
        random.shuffle(permutation)

        size_train = int(np.round(X_input.shape[ 0 ] * train_ratio))
        index_train = permutation[ 0 : size_train]
        index_test = permutation[ size_train : ]

        X_train, y_train = X_input[ index_train, : ], y_input[ index_train ]
        X_test, y_test = X_input[ index_test, : ], y_input[ index_test ]

        #names = ['Base','Subset','Subset-CF','Induced Points'];
        names = ['Base','Subset','Subset-CF','Induced Points','Adversarial Induced Points'];
        #names = ['Base','Induced Points','Adversarial Induced Points'];
        numIter = 10
        maxTime = 100
        numTimeSteps = 20
        modelNum = len(names);

        svgd_rmse_final = np.zeros((modelNum, numTimeSteps))
        svgd_ll_final = np.zeros((modelNum, numTimeSteps))
        svgd_iter_final = np.zeros((modelNum, numTimeSteps))

        ''' Training Bayesian neural network with SVGD '''
        #batch_size, n_hidden, max_iter, numParticles = 100, 50, 2000, 30  # max_iter is a trade-off between running time and performance
        batch_size, n_hidden, max_iter, numParticles = 100, 50, 100000, 20  # max_iter is a trade-off between running time and performance
        max_iterRS = 100000
        max_iterRSCF = 100000
        max_iterIP = 100000
        max_iterAIP = 100000
        m, adverMaxIter = 10,1
        max_iters = [max_iter, max_iterRS, max_iterRSCF, max_iterIP];

        np.set_printoptions(precision=4)
        for modelInd in range(0,5):
            for t in range(0,numIter):
                np.random.seed(t)
                print(names[modelInd], ': Iteration ', t+1, '/', numIter)
                start = time.time()

                if modelInd == 0 :# base
                    svgd = svgd_bayesnn(X_train, y_train, X_test, y_test, numTimeSteps = numTimeSteps, maxTime = maxTime,
                        batch_size = batch_size, n_hidden = n_hidden, M=numParticles, max_iter = max_iter,
                        method = 'none')
                elif modelInd == 1 : # Subset
                    svgd = svgd_bayesnn(X_train, y_train, X_test, y_test, numTimeSteps = numTimeSteps, maxTime = maxTime,
                        batch_size = batch_size, n_hidden = n_hidden, M=numParticles, max_iter = max_iterRS,
                        method = 'subparticles',m=m,cf=False)
                elif modelInd == 2 : # Subset (CF)
                    svgd = svgd_bayesnn(X_train, y_train, X_test, y_test, numTimeSteps = numTimeSteps, maxTime = maxTime,
                        batch_size = batch_size, n_hidden = n_hidden, M=numParticles, max_iter = max_iterRSCF,
                        method = 'subparticles',m=m,cf=True)
                elif modelInd == 3 : # Induced Points
                    svgd = svgd_bayesnn(X_train, y_train, X_test, y_test, numTimeSteps = numTimeSteps, maxTime = maxTime,
                        batch_size = batch_size, n_hidden = n_hidden, M=numParticles, max_iter = max_iterIP,
                        method = 'inducedPoints',m=m, uStat = True, adver=False)
                elif modelInd == 4 : # Induced Points (Adver)
                    svgd = svgd_bayesnn(X_train, y_train, X_test, y_test, numTimeSteps = numTimeSteps, maxTime = maxTime,
                        batch_size = batch_size, n_hidden = n_hidden, M=numParticles, max_iter = max_iterAIP,
                        method = 'inducedPoints',m=m, uStat = True, adver=True, adverMaxIter = adverMaxIter)

                [rmseResult, llResult, iterResult] = svgd.getResults()

                svgd_rmse_final[modelInd,:] = svgd_rmse_final[modelInd,:] + rmseResult / numIter
                svgd_ll_final[modelInd,:] = svgd_ll_final[modelInd,:] + llResult / numIter
                svgd_iter_final[modelInd,:] = svgd_iter_final[modelInd,:] + np.round(iterResult / numIter)


                np.save('./subset_1adver_rmseResult_'+datasetName,svgd_rmse_final)
                np.save('./subset_1adver_llResult_'+datasetName,svgd_ll_final)
                np.save('./subset_1adver_iterResult_'+datasetName,svgd_iter_final)

        #print('--------------------------------------------------------------------------------')
        #print('Dataset : ', datasetName)
        #print('[Options] : M=',numParticles, ', m=',m, ', max_iter=', max_iter, ', n_hidden=',n_hidden, ', batch_size=',batch_size)
        #print('--------------------------------------------------------------------------------')
        #for modelInd in range(0,modelNum):
        #    print (names[modelInd],' [Average of', numIter, 'runs] : ', max_iters[modelInd], ' iterations')
        #    print ('[rmse] Mean : ', "%.4f" % np.mean(svgd_rmse_final[modelInd,]), ' st.dev : ', "%.4f" % np.std(svgd_rmse_final[modelInd,]) )
        #    print ('[llik] Mean : ', "%.4f" % np.mean(svgd_ll_final[modelInd,]), ' st.dev : ', "%.4f" % np.std(svgd_ll_final[modelInd,]) )
        #    print ('[time] Mean : ', "%.2f" % np.mean(svgd_time_final[modelInd,]), ' st.dev : ', "%.2f" % np.std(svgd_time_final[modelInd,]), '\n')
